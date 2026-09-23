# 自适应负载均衡与提供方临时熔断实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将自适应负载均衡器 `AdaptiveLoadBalancer` 深度装配至网关主请求链路，并实现提供方主动熔断机制 `DISABLED_PROVIDERS`，确保被封禁节点（如 DeepSeek）零延迟避让与平滑降级。

**Architecture:** 
1. 在 `CooldownTracker` 与 `FailoverRouter` 中支持 `DISABLED_PROVIDERS` 配置识别与自动节点剔除；
2. 在 `server.py` 的 `/v1/chat/completions`、`/v1/messages` 和 `/v1/responses` 入口接入 `AdaptiveLoadBalancer.select_route(...)`；
3. 全生命周期注入 `acquire()`、`release()`、`record_ttft()` 并暴露至 `/v1/diagnostics`；
4. 当初选节点处于被禁状态时，立即在同等能力环中推选最优健康节点承接。

**Tech Stack:** Python 3.10+, FastAPI, Starlette, Pydantic, HTTPX, asyncio

**Spec:** `docs/superpowers/specs/2026-09-23-adaptive-load-balancer-and-provider-disabling-design.md`

## Global Constraints

- 不可引入额外的第三方重量级依赖；
- 保持对现有多协议（OpenAI Chat / Anthropic Messages / OpenAI Responses）的完全向下兼容；
- 必须通过已有全部单测（tests/*.py），不产生任何回归问题；
- 针对 disabled 节点的重定向耗时必须在本地计算完成（< 1ms），不能发起任何无效的外部网络 IO。

---

### Task 1: 在 CooldownTracker、FailoverRouter 与 AdaptiveLoadBalancer 中实现 DISABLED_PROVIDERS 过滤

**Files:**
- Modify: `providers/failover.py:100-388`
- Modify: `providers/load_balancer.py:256-420`
- Test: `tests/test_failover.py`
- Test: `tests/test_load_balancer.py`

**Interfaces:**
- Consumes: `os.getenv("DISABLED_PROVIDERS", "")`
- Produces: 
  - `CooldownTracker.disabled_providers: Set[str]`
  - `CooldownTracker.is_disabled(provider_key: str) -> bool`
  - `FailoverRouter.get_fallback_candidates(...)` (自动排除 disabled 节点)
  - `AdaptiveLoadBalancer.select_route(...)` (自动将 disabled 节点判定为不可用并寻找同环备用)

- [ ] **Step 1: 在 tests/test_failover.py 中编写测试用例验证 disabled_providers**

```python
def test_disabled_providers_exclusion(self):
    os.environ["DISABLED_PROVIDERS"] = "deepseek,glm"
    tracker = CooldownTracker()
    self.assertTrue(tracker.is_disabled("deepseek"))
    self.assertTrue(tracker.is_disabled("glm"))
    self.assertFalse(tracker.is_disabled("qwen"))

    router = FailoverRouter()
    candidates = router.get_fallback_candidates("qwen", "Qwen3.7-Max")
    keys = [c.provider_key for c in candidates]
    self.assertNotIn("deepseek", keys)
    self.assertNotIn("glm", keys)
```

- [ ] **Step 2: 运行单测确认失败**

Run: `.venv/bin/python -m unittest tests/test_failover.py`
Expected: FAIL (AttributeError: 'CooldownTracker' object has no attribute 'is_disabled')

- [ ] **Step 3: 在 providers/failover.py 和 providers/load_balancer.py 中实现 disabled_providers**

在 `CooldownTracker.__init__` 读取 `os.getenv("DISABLED_PROVIDERS", "")` 并解析为集合，提供 `is_disabled(provider_key: str) -> bool`。在 `get_fallback_candidates` 中跳过 `self.cooldown_tracker.is_disabled(p_key)`。
在 `AdaptiveLoadBalancer.is_high_load` 与 `select_route` 中增加对 `is_disabled` 的判断。

- [ ] **Step 4: 运行单测确认通过**

Run: `.venv/bin/python -m unittest tests/test_failover.py tests/test_load_balancer.py`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add providers/failover.py providers/load_balancer.py tests/test_failover.py tests/test_load_balancer.py
git commit -m "feat(failover): support DISABLED_PROVIDERS exclusion in CooldownTracker and FailoverRouter"
```

---

### Task 2: 在 server.py 主请求管线中装配 AdaptiveLoadBalancer 与首选被禁即刻转移

**Files:**
- Modify: `server.py:120-1650`
- Modify: `providers/responses_compat.py` (如需)
- Test: `tests/test_server.py`
- Test: `tests/test_responses.py`

**Interfaces:**
- Consumes: `load_balancer = AdaptiveLoadBalancer(cooldown_tracker=failover_router.cooldown_tracker)`
- Produces: 
  - 全局自动降级、打散决策与请求生命周期度量跟踪 (`acquire`, `release`, `record_ttft`)
  - 响应头：`x-load-balancing-action`, `x-load-balancing-reason`, `x-dispatched-provider`, `x-dispatched-model`

- [ ] **Step 1: 在 tests/test_server.py 中添加自适应降级与被禁转移单测**

测试直接请求 `deepseek-chat`（当 `DISABLED_PROVIDERS=deepseek` 时），验证其被即刻替换为同能力环中的 `qwen` 或 `doubao`，且返回 `x-load-balancing-action` 响应头。

- [ ] **Step 2: 运行单测确认失败**

Run: `.venv/bin/python tests/test_server.py`
Expected: FAIL (缺少响应头或未即刻转移)

- [ ] **Step 3: 在 server.py 中实现调度装配**

在请求进入 `chat_completions`、`anthropic_messages` 和 `responses` 时：
1. 估算 prompt tokens；
2. 检查初选 provider 是否被禁，若被禁从同能力环挑选首个可用节点；
3. 调用 `load_balancer.select_route(...)`；
4. 包装生成器调用 `load_balancer.acquire(...)`，首包计算 TTFT 并调用 `record_ttft`，流退出在 `finally` 中调用 `load_balancer.release(...)`；
5. 在响应头透传调度决策。

- [ ] **Step 4: 运行单测确认通过**

Run: `.venv/bin/python tests/test_server.py && .venv/bin/python tests/test_responses.py`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add server.py tests/test_server.py tests/test_responses.py
git commit -m "feat(server): wire AdaptiveLoadBalancer and zero-latency disabled provider switch into request pipeline"
```

---

### Task 3: 诊断监控接口 (/v1/diagnostics) 与健康检查 (/health) 数据升级

**Files:**
- Modify: `server.py:600-840`
- Modify: `healing/monitor.py:30-150`
- Test: `tests/test_healing.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `load_balancer.get_load_stats()`
- Produces: `/v1/diagnostics` 返回 `load_balancer` 统计字段与 `disabled_providers` 列表；`HealthMonitor` 跳过被禁用 provider。

- [ ] **Step 1: 在 tests/test_server.py 中增加 /v1/diagnostics load_balancer 字段校验**

断言 `/v1/diagnostics` 响应包含 `load_balancer`，且包含 `providers`、`total_requests`、`high_watermark_ratio` 等度量字段。

- [ ] **Step 2: 运行单测确认失败**

Run: `.venv/bin/python tests/test_server.py`
Expected: FAIL (KeyError: 'load_balancer')

- [ ] **Step 3: 在 server.py 与 healing/monitor.py 中实现暴露与跳过探针**

更新 `get_diagnostics` 聚合 `load_balancer.get_load_stats()`；在 `HealthMonitor.check_provider` 中若 provider 被禁用则直接设置 `status = "disabled"` 并跳过网络测试。

- [ ] **Step 4: 运行单测确认通过**

Run: `.venv/bin/python tests/test_server.py && .venv/bin/python tests/test_healing.py`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add server.py healing/monitor.py tests/test_server.py tests/test_healing.py
git commit -m "feat(diagnostics): expose real-time load balancer metrics and mark disabled providers in health monitor"
```

---

### Task 4: 端到端全套单测回归验证与优化报告��新

**Files:**
- Modify: `reports/optimization_proposal.json`
- Modify: `reports/optimization_recommendations.md`
- Test: 全部 10 个测试文件

- [ ] **Step 1: 运行全量 10 个测试套件**

Run:
```bash
.venv/bin/python tests/test_providers.py && \
.venv/bin/python tests/test_server.py && \
.venv/bin/python tests/test_session.py && \
.venv/bin/python tests/test_healing.py && \
.venv/bin/python tests/test_responses.py && \
.venv/bin/python tests/test_failover.py && \
.venv/bin/python tests/test_load_balancer.py && \
.venv/bin/python tests/test_prompt_cache.py && \
.venv/bin/python tests/test_prompt_optimizer.py && \
.venv/bin/python tests/test_dispatcher.py
```
Expected: PASS (所有 10 个套件全部通过，0 错误)

- [ ] **Step 2: 更新优化进度报告**

更新 `reports/optimization_proposal.json` 与 `reports/optimization_recommendations.md`，将 `OPT-003` 标记为 `COMPLETED`。

- [ ] **Step 3: 最终提交并推送到 GitHub 远程仓库**

```bash
git add reports/
git commit -m "chore: update optimization reports marking OPT-003 as implemented"
git push origin main
```
