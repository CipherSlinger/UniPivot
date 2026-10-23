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

## 待办队列（PM 维护，从上往下做）
1. 等待新缺陷（当前模块已知问题均已修复并有回归测试）。