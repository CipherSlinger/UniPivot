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

## 已测量：真实并发下的资源归属（`tests/test_concurrency_soak.py`）

前面的套件把每条路径都钉住了，但几乎都是 N=1 的定点用例——定点用例回答"这条路径会
回收"，回答不了"同时有十几条路径时每条都回收"，更回答不了"信号量真的是并发上界"
（而不是恰好没有并发）。本套件全部经**真实 ASGI 栈**驱动，上游一律脚本化。

**复现配方**（照抄即可，坑都在注释里）：

- `send` 抛 `OSError` 就是"客户端消失"的忠实模型。Starlette 1.9.0 在
  `spec_version >= 2.4` 下把 send 的 `OSError` 转成 `ClientDisconnect`；低版本走
  另一条分支，测的就不是同一条路径了。`send_frames=0` = 响应头之前消失。
- **错开启动**：`select_route` 读的是 `LB.active_concurrency`，而配额是在 `gen()`
  里 acquire 的（`execute_chat` 之后）。同一 tick 起来的请求**全都看到 0 并发**，
  必然全压到同一个节点上，环根本不会分摊——实测第一版 14 条流全落在 qwen。
  按 0.12s 铺开才看得到整环分摊。
- **每个用例一个事件循环**：`asyncio.Semaphore` 在首次使用时绑定循环，全局 router
  的信号量会在第一个 `asyncio.run` 里绑死，第二个用例再用就是
  `is bound to a different event loop`（会被包成 502 并给节点记冷却，污染测量）。
  重建 `tracker._semaphores` 即可。这是测试隔离问题，不是产品缺陷。
- 计数必须放在**共享节点账本**上：真实 `_make_provider` 每次派发都新建 provider
  实例，挂在实例上每个请求都是 1/1，等于没测。
- 断言结束后必须给独立收尾任务留调度机会（50~80 × `sleep(0.01)`）。

实测（`n=14` 并发长流）：

| 场景 | 结果 |
|---|---|
| 整环浸泡 | 逐节点峰值 `qwen 3 / doubao 5 / glm 4 / deepseek 1 / kimi 1`，**全部 ≤ 各自上限**，分摊到 5 个节点 |
| 单节点浸泡（其余四家禁用，qwen 上限 3） | 峰值 **恰好 3**（<3 说明并发没发生，>3 说明治理失效） |
| 断连风暴（3 轮 × 12 客户端） | 36 条上游流，每轮结束后 `sem` 逐值归位、`LB=0`、`inflight=0`，**三轮不累积** |
| 心跳覆盖长沉默（沉默 0.35s = 7 个心跳周期） | 12 帧 `: keepalive`，内容 `['a','b','c']` 一个不少、顺序不变 |
| SF 扇出背压（Leader + 4 跟随者，其一 0.08s/帧） | 慢跟随者被摘除 1 个，快跟随者全收，总耗时 0.62s（未被拖住） |

**结论：并发治理本身是站得住的**——信号量确实是整段流的上界，断连风暴不累积泄漏。
但浸泡**当场抓到一个真实缺陷**，见下条。

## 已修复 [C3]：中途断连时上游生成器从没被关闭

`wrapped_stream` 的 `finally` 原先只调 `_release_sem()`，**从不 `aclose()` `raw_stream`**。
槽位回来了、LB 配额也归零了，唯独 provider 生成器停在 `yield` 上还活着——它要等到
CPython 回收这个对象时才跑 `finally`。

实测（把生成器引用一直攥在手里、禁止 GC）：`opened=1 closed=0`；放手让 GC 跑完才变
1/1。真实进程里那条流会带着上游 HTTP 连接一直挂到垃圾回收，正是"客户端已走、上游仍在
生成"的空耗。

**为什么全套件一直是绿的**：`raw_stream` 只被 `wrapped_stream` 的闭包引用，包装流被
丢弃时引用计数瞬间归零，async generator 的终结器顺手跑了 `finally`——**是 GC 替代码
完成了回收**。所以任何"驱动完再断言 closed"的用例都测不到它。

修复：`finally` 里先 `await _close_raw_stream()`（独立任务 + shield，断连时本任务已在
取消中），**再** `_release_sem()`。顺序不能反：先放槽位的话，下一个拿到槽位的请求会与
这条尚未关闭的流并存，节点并发上界当场被突破。

回归用例 `tests/test_stream_start_reclaim.py::test_mid_stream_disconnect_closes_the_upstream_generator`
用 `_RefHoldingProvider` 把生成器对象**本身**也攥住，让终结器无法替代码收尾——修复前
三条协议全部 RED（`1 开 / 0 关`），修复后全绿。**写这类断言时，先问一句"GC 会不会替我
把它做掉"**，否则测的是解释器而不是代码。

## 待办队列（PM 维护，从上往下做）
1. 等待新缺陷（当前模块已知问题均已修复并有回归测试）。
2. ~~`_MIN_HANDSHAKE_ATTEMPT_SECONDS = 0.05` 是经验地板~~ **已测（保留为观察项）**：
   网关自身握手开销 `n=30` → min 0.445ms / median 0.479ms / p95 0.574ms / max 0.712ms，
   对 50ms 地板有 **70× 余量**。五家上游首包以秒计，因此维持 0.05s 正确。
   **重校准触发条件**：接入首包在毫秒级的本地模型（那时 70× 的余量就不再是余量）。
3. ~~预算按"剩余候选数"分摊对慢推理模型偏紧~~ **已测（保留为观察项）**：
   预算 0.6s / 三候选 → qwen 份额 0.20s，而 qwen 需 0.40s。实测四轮：
   `turn1/2` 由健康候选 glm 正常服务，qwen 冷却 0、失败计数 0（**没有诬告**）；
   `turn3` 连续第 3 次触发升级 → 冷却 60.0s、成因 `timeout`、**failure_count 0**
   （不是 `waf`，不会让运维去点验证码），streak 饱和在 3；`turn4` 仍每轮碰一次 qwen
   （这是 OPT-021"短冷却保留最后一次尝试"的既定语义，不是缺陷）。
   即：升级语义正确、不会永久冻结、也不会误报成因。**唯一代价是每轮仍白付一份份额**，
   而共享同一环的健康候选会让总耗时稳定在 0.31s。
   **处置**：若现场出现健康节点被预算裁短（日志关键词「本次上限由握手总预算裁剪得出」），
   调大 `GATEWAY_HANDSHAKE_BUDGET`（默认 180s / 五节点 = 36s 一份），不要调小单次上限。
4. **下一步（具体）**：`tests/test_concurrency_soak.py` 目前只压 `/v1/chat/completions`
   的浸泡，`/v1/messages` 与 `/v1/responses` 只在断连风暴里各出现两次。把浸泡扩到三协议
   各自 `n>=12`，重点看 `/v1/responses`——它的 LB 配额是在 `execute_chat` **之前**
   acquire 的（server.py:3926），与另两条的获取时机不同，最可能藏着不同的泄漏形状。