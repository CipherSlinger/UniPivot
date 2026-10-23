---
name: risk-avoidance
description: Upstream risk-control avoidance — cooldowns, quarantine leases, credential watches, ban inference, pacing, dispatch gates. Use when working on risk_ledger.py, credential_watch.py, credential_renewal.py, quarantine.py, risk_dossier.py, risk_correlation.py, device_identity.py, or the dispatch gate in failover.py.
---

# 模块：上游风控避让

## 你拥有的文件
- `providers/risk_ledger.py`、`credentials_watch.py`、`credential_renewal.py`、`quarantine.py`
- `providers/risk_dossier.py`、`risk_correlation.py`、`device_identity.py`
- `providers/failover.py` 中 `evaluate_dispatch_gate`、`record_failure`、`touch_avoidance_reason`
- 对应测试：`tests/test_risk_*.py`、`tests/test_*quarantine*.py`、`tests/test_ban_inference.py`、`tests/test_waf_escalation.py`、`tests/test_dispatch_*.py` 等

## 你必须守住的不变量
1. **绝不触碰被封禁/撤回的节点**：零上游调用，包括作为故障转移后备。
2. **一处配置只有一个判读者**：`DISABLED_PROVIDERS` 不得在多处各自解析。
3. **成因必须诚实**：`auth` → 重登有效；`waf` → 需要人工完成验证；`ratelimit` → 自愈，重登无意义；`ban` → 重登无效，只能等。诊断绝不能给出无效的处置建议。
4. **预演 == 派发**：`audit_dispatch` 的预测必须与 `execute_chat` 的真实行为一致，且是**结构性**一致（同一个闸门定义）。
5. **拒绝要能被客户端读懂**：确定性拒绝必须带 `x-should-retry: false`（**唯一**真正抑制重试的头，Claude Code 在状态码判断之前读它）+ `Retry-After`（标准字段，值取抛出点知道的真实窗口，不能用常量）。

## 工作方法（硬性）
- 每个新机制必须说明它与已有五种避让语义（软冷却 / 运维租约 / 自动租约 / 整环凭证失效 / 长效避让）的**区别**，否则很可能是在重复已有的东西。
- 聚合新的"一致性"断言时，优先写成**对全部状态的性质测试**，而不是逐例枚举。
- **测一个头 / 一个开关的效果时，唯一变量只能是它**。曾经比较两个端口的两个服务端得出「Retry-After 抑制重试」，受控复测（同一服务端，只切响应头）推翻了它 —— 真实信源是 `x-should-retry`。

## 已被证伪，不要再提
- **`Retry-After` 能抑制 Claude Code 对 503 的重试**：不能。CLI 只在 429 上解析它。测出来 7 次（无头）、7 次（只带 Retry-After）、1 次（带 `x-should-retry: false`）。别再把它当成重试抑制手段。

## 待办队列（PM 维护，从上往下做）
1. 等待新缺陷。