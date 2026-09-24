# 风控拦截避让与自适应流量整形 (Risk Control Avoidance & Adaptive Traffic Shaping) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建网关层风控避让与流量整形引擎（TrafficPacer 与 RiskGuard），通过节点级并发信号量控制、随机拟人抖动、全量 WAF/人机验证指纹智能识别与指数级冷却熔断，彻底规避因突发高频调用导致的阿里 RGV587 滑块、字节盾风控与 Cloudflare 封禁，并在触发风险时实现 0.03ms 内同能力环透明故障转移。

**Architecture:** 
1. 在 `providers/failover.py` 中引入 `TrafficPacer`，维护各 Provider 的严格并发信号量 (`Semaphore`) 与带 Jitter 的拟人调用间隔防突发。
2. 扩充 `is_risk_interception` 模式库，全面拦截 Alibaba RGV587、ByteDance Acrawler、Cloudflare Turnstile、Moonshot Rate Limit 与 Token 过期特征，触发阶梯式指数冷却回退 (`cooldown * (2 ** fails)`)。
3. 在 `server.py` 的请求流（OpenAI Chat / Anthropic Messages / OpenAI Responses）中接入并发整形与风险防护包装，并向客户端回传 `x-risk-pacing-delay-ms` 与 `x-failover-from` 等排障响应头。
4. 在 `/v1/diagnostics` 中暴露全节点并发占用、平滑间隔等待延迟及风险冷却倒计时。

**Tech Stack:** Python 3.10+, FastAPI / Starlette, asyncio.Semaphore, httpx, pytest / unittest.

**Spec:** `docs/superpowers/specs/2026-09-24-risk-control-avoidance-design.md`

## Global Constraints

- **Zero-Latency Bypass**: 处于冷却或风控封禁状态的节点必须在 `< 0.03ms` 内被绕过，严禁发起外部网络 IO。
- **No Half-Packet Corruption**: 首个 Token Chunk 产生后绝不切换节点；故障转移必须在首字发射前的物理连接/握手阶段完成。
- **HTTP Header RFC 规范**: 所有注入的自定义响应头必须严格符合 ASCII 编码规范，严禁抛出 Starlette latin-1 异常。
- **Zero-Disruption Fallback**: 当所有同环节点均受阻时，必须返回标准协议错误 JSON，不得抛出未捕获的 Python 内部异常。

---

### Task 1: 流量整形器 TrafficPacer 与风控指纹识别增强

**Files:**
- Modify: `providers/failover.py:75-130`
- Test: `tests/test_risk_avoidance.py`

**Interfaces:**
- Produces: 
  - `class TrafficPacer`: 提供 `async with pacer.acquire(provider_name):` 异步上下文管理器，控制单节点最大并发并平滑请求间隔。
  - `def is_risk_interception(exc_or_text: Any) -> Tuple[bool, str]`: 识别错误信息是否属于 WAF 滑块、风控验证码、CF 封禁或 Token 失效。
  - `get_traffic_pacer() -> TrafficPacer` 单例方法。

- [ ] **Step 1: 编写测试用例 `tests/test_risk_avoidance.py` 验证并发限制与指纹识别**

```python
import asyncio
import time
import pytest
from providers.failover import (
    TrafficPacer,
    is_risk_interception,
    PROVIDER_RISK_SPECS,
    RiskLevel,
)

def test_risk_interception_matching():
    # 测试阿里 RGV587
    is_risk, reason = is_risk_interception("Error: RGV587 FAIL_SYS_USER_VALIDATE")
    assert is_risk is True
    assert "Alibaba" in reason

    # 测试字节盾风控
    is_risk, reason = is_risk_interception("ProviderError: 豆包触发风控/人机验证，请运行 .venv/bin/python login.py doubao")
    assert is_risk is True
    assert "ByteDance" in reason

    # 测试 DeepSeek 封禁
    is_risk, reason = is_risk_interception("ProviderError: DeepSeek 接口错误: Authorization Failed (invalid token)")
    assert is_risk is True
    assert "DeepSeek" in reason

    # 测试常规无关报错
    is_risk, _ = is_risk_interception("KeyError: 'choices'")
    assert is_risk is False

@pytest.mark.asyncio
async def test_traffic_pacer_concurrency_and_pacing():
    pacer = TrafficPacer()
    # 模拟 qwen 严格 max_concurrency = 1
    t0 = time.time()
    order = []

    async def worker(w_id: int):
        async with pacer.acquire("qwen"):
            order.append((w_id, time.time() - t0))
            await asyncio.sleep(0.05)

    await asyncio.gather(worker(1), worker(2))
    assert len(order) == 2
    # 串行执行，第二个 worker 的启动时间必须滞后
    assert order[1][1] >= order[0][1] + 0.04
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_risk_avoidance.py`
Expected: FAIL with "ImportError: cannot import name 'TrafficPacer'"

- [ ] **Step 3: 在 `providers/failover.py` 中实现 `TrafficPacer` 与增强的 `is_risk_interception`**

在 `providers/failover.py` 中：
1. 细化 `PROVIDER_RISK_SPECS`：为敏感节点配置严格 `max_concurrency: 1` 和拟人调用间隔 `min_call_interval`。
2. 实现 `TrafficPacer` 类：
   - 包含字典 `_semaphores: Dict[str, asyncio.Semaphore]`
   - 包含字典 `_last_call_time: Dict[str, float]`
   - 提供 `async acquire(provider: str) -> AsyncIterator[float]`，计算并注入 jitter 延迟，同时通过信号量限制并发。
3. 实现 `is_risk_interception`：匹配已知各平台 WAF 关键词。

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_risk_avoidance.py`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add providers/failover.py tests/test_risk_avoidance.py
git commit -m "feat(failover): add TrafficPacer and enhanced WAF risk interception matcher"
```

---

### Task 2: 接入网关请求流水线与自适应避让熔断

**Files:**
- Modify: `server.py:125-145, 1090-1160`
- Modify: `providers/failover.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `TrafficPacer`, `is_risk_interception`, `CooldownTracker`
- Produces: 
  - `x-risk-pacing-delay-ms`: 响应头透传被流量整形器平滑等待的毫秒数。
  - 自动将捕获到 WAF 拦截的节点置入冷却期并无缝分流至同环备选节点。

- [ ] **Step 1: 在 `tests/test_server.py` 中添加流量整形响应头与避让测试**

```python
def test_risk_avoidance_headers_and_cooldown(client):
    # 发送请求，验证响应头包含流量整形指标
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert resp.status_code == 200
    assert "x-risk-pacing-delay-ms" in resp.headers
```

- [ ] **Step 2: 运行测试验证缺失响应头失败**

Run: `.venv/bin/python -m pytest tests/test_server.py -k test_risk_avoidance_headers`
Expected: FAIL

- [ ] **Step 3: 在 `server.py` 中集成 `TrafficPacer`**

1. 初始化全局 `traffic_pacer = get_traffic_pacer()`。
2. 在调用底层 Provider 前，使用 `async with traffic_pacer.acquire(provider_key) as delay_ms:` 保护调用。
3. 将 `delay_ms` 格式化为响应头 `x-risk-pacing-delay-ms`。
4. 在捕获到 `ProviderError` 时，通过 `is_risk_interception` 判定，若为真则立即触发 `cooldown_tracker.record_failure(provider_key, is_waf=True)`，进入长效冷却。

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_server.py -k test_risk_avoidance_headers`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add server.py tests/test_server.py
git commit -m "feat(server): integrate TrafficPacer and risk avoidance headers into request pipeline"
```

---

### Task 3: 诊断指标暴露与主动会话健康度监控

**Files:**
- Modify: `server.py:820-890`
- Modify: `healing/monitor.py`
- Test: `tests/test_healing.py`

**Interfaces:**
- Produces: `/v1/diagnostics` 接口中的 `traffic_pacing` 与 `risk_avoidance` 状态字典。

- [ ] **Step 1: 编写测试验证 `/v1/diagnostics` 包含风控监控度量**

在 `tests/test_server.py:test_diagnostics_endpoint` 中增加断言：
```python
assert "risk_avoidance" in data
assert "active_semaphores" in data["risk_avoidance"]
assert "cooldown_nodes" in data["risk_avoidance"]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `.venv/bin/python -m pytest tests/test_server.py -k test_diagnostics_endpoint`
Expected: FAIL

- [ ] **Step 3: 在 `server.py:get_diagnostics` 中挂载状态数据**

将 `traffic_pacer.get_stats()` 与 `cooldown_tracker.get_all_cooldowns()` 组装至 `risk_avoidance` 字段。

- [ ] **Step 4: 运行测试验证通过**

Run: `.venv/bin/python -m pytest tests/test_server.py -k test_diagnostics_endpoint`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add server.py tests/test_server.py
git commit -m "feat(diagnostics): expose traffic pacing and risk avoidance telemetry"
```

---

### Task 4: 全套回归测试与优化建议书更新

**Files:**
- Modify: `healing/task_dispatcher.py`
- Modify: `reports/optimization_proposal.json`
- Modify: `reports/optimization_recommendations.md`

- [ ] **Step 1: 运行全量 11 套离线测试套件**

```bash
.venv/bin/python tests/test_providers.py && \
.venv/bin/python tests/test_server.py && \
.venv/bin/python tests/test_session.py && \
.venv/bin/python tests/test_healing.py && \
.venv/bin/python tests/test_failover.py && \
.venv/bin/python tests/test_load_balancer.py && \
.venv/bin/python tests/test_prompt_cache.py && \
.venv/bin/python tests/test_prompt_optimizer.py && \
.venv/bin/python tests/test_dispatcher.py && \
.venv/bin/python tests/test_responses.py && \
.venv/bin/python tests/test_risk_avoidance.py
```
Expected: 全部测试 100% PASS

- [ ] **Step 2: 更新任务调度与建议书**

在 `healing/task_dispatcher.py` 中将新增的风控避让与自适应整形机制标注，更新报告。

- [ ] **Step 3: 提交代码**

```bash
git add healing/task_dispatcher.py reports/
git commit -m "chore: record risk control avoidance implementation and refresh reports"
```
