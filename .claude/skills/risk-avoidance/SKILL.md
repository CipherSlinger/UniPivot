---
name: risk-avoidance
description: Upstream risk-control avoidance — cooldowns, quarantine leases, credential watches, ban inference, pacing, dispatch gates. Use when working on risk_ledger.py, credential_watch.py, credential_renewal.py, quarantine.py, risk_dossier.py, risk_correlation.py, device_identity.py, or the dispatch gate in failover.py.
---

# 模块：上游风控避让

## 你拥有的文件
- `providers/risk_ledger.py`、`providers/credential_watch.py`、`providers/credential_renewal.py`、`providers/quarantine.py`
- `providers/risk_dossier.py`、`risk_correlation.py`、`device_identity.py`
- `providers/failover.py` 中 `evaluate_dispatch_gate`、`record_failure`、`_set_cooldown_cause`、`touch_avoidance_reason`、`note_budget_capped`
- `server.py` 只有 `_refusal_headers` / `_is_deterministic_refusal` 与确定性拒绝的抛出点
- 对应测试：`tests/test_risk_*.py`、`tests/test_*quarantine*.py`、`tests/test_ban_inference.py`、`tests/test_waf_escalation.py`、`tests/test_dispatch_*.py`、`tests/test_unconfigured_credentials.py` 等

## 你必须守住的不变量
1. **绝不触碰被封禁/撤回的节点**：零上游调用，包括作为故障转移后备。
2. **一处配置只有一个判读者**：`DISABLED_PROVIDERS` 不得在多处各自解析；冷却成因只能由 `_set_cooldown_cause` 写。
3. **成因必须诚实**：`auth` → 重登有效；`waf` → 需要人工完成验证；`ratelimit` → 自愈，重登无意义；`ban` → 重登无效，只能等；`timeout`/`truncated` → 上游故障，建议指向上游健康/出口质量；`unconfigured` → **本机配置缺口**，重登有效（与 `auth` 同一处置，与 `disabled` 相反）。诊断绝不能给出无效的处置建议。
4. **预演 == 派发**：`audit_dispatch` 的预测必须与 `execute_chat` 的真实行为一致，且是**结构性**一致（同一个闸门定义）。
5. **拒绝要能被客户端读懂**：确定性拒绝必须带 `x-should-retry: false`（**唯一**真正抑制重试的头，Claude Code 在状态码判断之前读它）+ `Retry-After`（标准字段，值取抛出点知道的真实窗口，不能用常量）。

## 工作方法（硬性）
- 每个新机制必须说明它与已有六种避让语义（软冷却 / 运维租约 / 自动租约 / 整环凭证失效 / 长效避让 / **未配置**）的**区别**，否则很可能是在重复已有的东西。
- 聚合新的"一致性"断言时，优先写成**对全部状态的性质测试**，而不是逐例枚举。
- **测一个头 / 一个开关的效果时，唯一变量只能是它**。曾经比较两个端口的两个服务端得出「Retry-After 抑制重试」，受控复测（同一服务端，只切响应头）推翻了它 —— 真实信源是 `x-should-retry`。
- **新增一种冷却成因时，要同时问四个读成因的地方**：`release_auth_avoidance_locked` 的 `causes`、`risk_dossier._SEVERITY_TO_VERDICT` / `_LIVE_ONLY_CAUSES` / `AVOIDANCE_MATRIX`、`touch_avoidance_reason`、以及 `tests/test_risk_dossier.py` 里那条**钉住矩阵成员**的断言。漏掉任何一处，运维读到的建议就会指向错误的处置（`unconfigured` 漏在 `_SEVERITY_TO_VERDICT` 里时，档案会说「无需人工介入」）。
- 一条避免"自锁"的判据：**任何自动避让，都必须存在一条不依赖它自身状态的解除路径**。OPT-026/027/029 与 DAY-ZERO 都是同一个形状——避让制造的状态把唯一能修复它的机制锁死了。

## 已被证伪，不要再提
- **`Retry-After` 能抑制 Claude Code 对 503 的重试**：不能。CLI 只在 429 上解析它。测出来 7 次（无头）、7 次（只带 Retry-After）、1 次（带 `x-should-retry: false`）。别再把它当成重试抑制手段。

## 确定性拒绝的状态码：实测结论（决定：保持 503）
需求是「确定性拒绝（零上游：credential_outage / disabled / quarantined / ring_exhausted / upstream_timeout）到底该回什么状态码」。**结论：保持 503 不变**，因为这个状态码不是我们能选的，是客户端决定的。

受控测量（`/tmp/ra_probe/refuse`，同一台服务端、同一个假上游、只切状态码；`E2E_STRIP=1` 剥掉我们自己的两个响应头，模拟第三方 SDK 看到的样子；`timeout 60`）：

| 状态码 | codex 请求数 | claude 请求数 | 用户看到 |
|---|---|---|---|
| 503（现状） | 30 | 7（剥头）/ 1（带头） | codex 无限重试；claude 因 `x-should-retry` 立即停 |
| 400 | **1** | 7 | codex 停，但把网关的原始 JSON 当正文吐给用户 |
| 404 | 30 | — | 同 5xx |
| 409 | 6 | 7 | 稍好，仍无意义重试 |
| 422 | 30 | — | 同 5xx |
| 429 | 1 | 7 | codex 止于 429 规则（**不是** `Retry-After`），但我们从未用过 429 表达拒绝 |

三点结论：
1. **codex 0.145.0 只按状态码分流**：400/429 → 1 次；5xx/404/422 → 30 次；409 → 6 次。它**不读** `x-should-retry`，也**不读**非 429 的 `Retry-After`。所以选 400 能压住 codex，但代价是把结构化错误当正文打印（`ERROR: {"error":{...}}` 原样出现在用户终端），且与我们自己的 "502/503 才是上游故障" 语义冲突。
2. **Claude Code 在状态码判断之前读 `x-should-retry: false`**，所以它在 400 / 409 / 429 / 503 上一律 1 次——我们现有的头已经是它最优的解，换状态码只会让第三方 SDK（读不到我们的头）更差。
3. **没有任何一个状态码能让两个客户端同时「只请求一次且看到真正原因」**。503 + `x-should-retry: false` 是唯一同时满足「Claude Code 一次即止」与「codex 至少能看到上游原因」（它 30 次后仍会打印 `ERROR: {...}`）的组合。**因此本模块的状态码不需要改；要改的从来只有响应头。**

遗留的可改进点（未做，因为改的是客户端行为而不是我们的契约）：给 codex 这类只认状态码的客户端提供一个 4xx 通道（例如 `400` + 保留 `x-should-retry`），需要先确认不会让现有 SDK 的 `is_retryable` 判定反向。**做之前必须先测量，不能用"看起来更对"决定。**

## 队列（PM 维护，从上往下做）
1. **`[BUDGET-7]` 预算裁短证据与冷却窗口的交叉**：`note_budget_capped` 现在在达到阈值时写一次 `timeout` 冷却，并在 `record_success` 时清零 streak。但**冷却期间到达的请求**仍会走「短软冷却只降级」的最后一搏路径（load balancer 之上才有拦截）。需要在真实 HTTP 路径上量：一个「挂死但偶尔答一次」的节点，在 streak 饱和 + 冷却生效的组合下，每 N 次请求仍被触碰多少次？如果成功一次就清零，那么「偶尔答一次」的节点永远在 `3 次白等 → 1 次冷却 → 1 次成功 → 3 次白等` 的循环里，等价于每 4 次请求付一次 1s。可考虑改为**衰减**而非清零（`streak = max(0, streak - 1)`），但必须先测出交替成功率下的稳态开销。
2. **`[REFUSAL-2]` 状态码的 4xx 通道**（见上节遗留点）：先测 400 是否会让 `claude` 的 `x-should-retry` 语义或其他 SDK 的 `is_retryable` 反向，再决定。
3. **`record_failure` 的窗口/成因错配**：`_set_cooldown_cause` 已经做到「非风控成因不抹掉仍在生效的风控成因」，但**反向**没有约束——一次风控成因的短冷却可以抹掉一个更长的 `timeout` 成因，于是档案把一次上游超时说成人机验证。`_RISK_CAUSES` 之间的互相改写目前是"后到者赢"，需要判断是否也该按「窗口是否被取代」判定。