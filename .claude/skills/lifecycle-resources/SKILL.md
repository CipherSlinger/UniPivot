---
name: lifecycle-resources
description: Request lifecycle and resource ownership — disconnects, stream start reclamation, timeouts, singleflight backpressure, concurrency semaphores. Use when working on failover.py's stream wrapper, disconnect_guard.py, singleflight.py, or any code that acquires a resource before the response body exists.
---

# 模块：请求生命周期与资源归属

## 你拥有的文件
- `providers/disconnect_guard.py`、`providers/singleflight.py`、`providers/stream_keepalive.py`
- `providers/failover.py` 中 `ManagedStream`、`execute_chat` 的资源获取/释放、超时
- `server.py` 的 `ReclaimingStreamingResponse`
- 对应测试：`tests/test_disconnect.py`、`tests/test_lifecycle.py`、`tests/test_stream_*.py`

## 你必须守住的不变量
1. **每个资源只有一个所有者**，且释放是幂等的。
2. 从未启动的 async generator 上 `aclose()` 是 **no-op** —— 它的 `finally` 永远不会跑。需要显式回收路径。
3. 在已取消的 task 里做清理，每个 `await` 都会被立刻取消 —— 清理必须派成独立 task 并 shield。
4. 流式路径的资源在 response body 存在**之前**就已获取，释放却写在 body 的 `finally` 里 —— 这个缺口必须显式补。
5. 超时 ≠ 风控。超时不得写入风控账本。

## 工作方法（硬性）
- 泄漏是**构造上不可见**的，测试必须断言**计数器**（信号量、LB 配额、上游 close 次数），不能断言"没抛异常"。
- 先写红，修复后跑全套。

## 已被证伪，不要再提
- **chat 路径的断开计数偏低**：与合成上游完成速度有竞态；用慢上游测是对的，且"已经正常完成的不计入断开"是正确行为。

## 已测量：`GATEWAY_HANDSHAKE_BUDGET` 的资源归属（第 59 节）

**结论：预算耗尽这条出口本身不持有任何资源，它之前的每个失败候选也都正常释放。**
没有发现泄漏，因此没有为"泄漏"制造修复。测量方法（`tests/test_budget_resource_reclaim.py`
就是它的固化版本，保留下来钉住不变量）：

- 五个候选全部 `chat()` 里 `await asyncio.Event().wait()`（连着但永不吐首包），
  `finally` 里 `closed += 1`；`chat()` 入口 `opened += 1`。
- `GATEWAY_HANDSHAKE_BUDGET` 取 0.2 / 0.6 / 1.2 / 0.9（跨 4 个数量级的份额），
  `GATEWAY_FIRST_CHUNK_TIMEOUT=30`，`GATEWAY_STREAM_IDLE_TIMEOUT=5`。
- 断言三组**计数器**：`opened == closed`（逐节点）；`get_semaphore(k)._value`
  与驱动前逐值相等；经真实 ASGI 栈驱动 `/v1/chat/completions`、`/v1/messages`、
  `/v1/responses` 后 `load_balancer.get_metrics(k).active_concurrency == 0`。

逐分支结论（读代码确认，非推测）：

| 失败分支 | 信号量 | 上游流 |
|---|---|---|
| 握手超时（`wait_for` 抛 `TimeoutError`） | `wrapped_stream` 的 `finally` → `ManagedStream.release_sync()` | `wait_for` 取消 `__anext__`，生成器收到 `GeneratorExit`，`finally` 跑 |
| 上游异常 / `ProviderError` | 同上（`sem_held` 判定） | 同上 |
| 候选被闸门拦下（禁用/租约/长避让） | 未获取（在 `try` 之前 redirect） | 未打开 |
| 预算耗尽 raise（循环顶部、`try` 之外） | 未获取（前一个候选已在上一轮 `finally` 归还） | 未打开 |
| 预算耗尽 raise（`try` 内、占用资源之前） | 未获取（判在 pacing/信号量之前） | 未打开 |
| 成功返回 `(stream, event, provider)` | `sem_handed_off=True`，交给 body 生成器 | 交给 `ReclaimingStreamingResponse`（C2） |

**健在的缺陷（本轮修掉的两个，都属于"把我们的决定写成上游的失败"）**：
1. 被总预算**裁短**的超时曾被按节点超时记账。实测份额 0.1s、节点 0.4s 健康应答 →
   qwen 冷却 60.0s、成因 `timeout`、`failure_count 1`。现用 `BudgetCappedTimeout`
   区分：继续故障转移，但不入冷却、不增失败计数。
2. 份额低于 `0.05s` 地板时曾按地板**派发**（`max(share, 0.05)`）。实测预算 0.12s /
   五候选 → 5 次注定失败的上游调用 + 5 份假的 60s 超时冷却，而请求结局与零上游拒绝
   完全相同。现判 `share < _MIN_HANDSHAKE_ATTEMPT_SECONDS` 即抛
   `BudgetExhaustedRefusal`（`zero_upstream`，实测上游调用 0 次、冷却全 0）。

**注意 `_count_untried_candidates` 的除数陷阱**：`get_fallback_candidates` 恒排除
初选节点，所以"当前节点是否已进 `tried_providers`"会让返回值差一，导致循环顶部与
`try` 内部按不同分母算份额（1/5 vs 1/4）。`current_provider` 参数就是为了让两处
只差时钟、不差分母。

## 待办队列（PM 维护，从上往下做）
1. 等待新缺陷（当前模块已知问题均已修复并有回归测试）。
2. `_MIN_HANDSHAKE_ATTEMPT_SECONDS = 0.05` 是经验地板（这五家首包以秒计）。若将来
   接入首包在毫秒级的本地模型，需要重新校准，否则会拒绝本可以成功的尝试。
3. 预算按"剩余候选数"分摊，但候选的真实耗时分布未知。默认 180s / 五节点 = 36s 一份，
   对慢推理模型偏紧；若现场出现健康节点被预算裁短（日志关键词「本次上限由握手总预算
   裁剪得出」），应调大 `GATEWAY_HANDSHAKE_BUDGET` 而非调小单次上限。