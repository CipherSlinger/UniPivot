# 架构设计说明书：自适应负载均衡管线接入与提供方临时熔断机制

- **设计时间**: 2026-09-23
- **版本**: v1.0.0
- **需求代号**: OPT-003-V2 / ITERATION-01
- **状态**: 已批准 / 实施中

---

## 1. 背景与业务痛点

在当前同枢 (UniPivot) 本地多协议网关中：
1. **真实风控与不可用挑战**：DeepSeek 上游触发了真实风控封禁，数天内无法提供服务。若网关缺少主动屏蔽配置，所有指向 DeepSeek 或将 DeepSeek 作为首选/候选的请求都会经历 401/429 报错重试，造成长达数秒甚至超时的 TTFT 首字时延惩罚。
2. **调度脱节**：底层已经实现优秀的秒级滑动窗口度量 `SlidingWindowMetrics` 与自适应负载均衡器 `AdaptiveLoadBalancer`，但尚未在 `server.py` 的主请求链路中全量装配，轻量请求自动降级（Query Complexity Degradation）与高水位同环分流（Same-Ring Shedding）未在生产流量中生效。
3. **运行时指标未闭环**：请求的并发进入、成功/失败、首包 TTFT、风控拦截没有实时反馈到滑动窗口桶中，使得系统无法根据真实流量动态自我调整。

---

## 2. 架构设计与核心组件

### 2.1 节点主动熔断与屏蔽机制 (Disabled Providers)
- **配置感知**：读取环境变量 `DISABLED_PROVIDERS`（支持逗号分隔，如 `DISABLED_PROVIDERS=deepseek`）。
- **避让注入**：
  - `CooldownTracker` 统一维护 `_disabled_providers: Set[str]`。
  - `FailoverRouter.get_fallback_candidates()` 自动将处于 disabled 状态的节点完全剔除。
  - 主入口 `server.py:resolve_model()` 之后，如果初选 provider 被禁用，调度器在同能力环中以 **0ms 内部瞬时重定向**至首个健康低风险节点，完全杜绝无效网络请求。
  - `HealthMonitor` 对 disabled 的 provider 标记状态为 `disabled`，跳过后台探针与多余的自愈修补任务。

### 2.2 请求生命周期与负载均衡流水线 (Request Lifecycle & Pipeline)
在 `/v1/chat/completions`、`/v1/messages`、`/v1/responses` 的请求生命周期中接入以下标准阶段：
```
1. 预处理 (Preprocessing):
   - 计算估算输入 Prompt Tokens；
   - 提取 is_thinking 推理诉求与 tools 列表。

2. 路由与分流决策 (Routing Decision):
   - 若 primary_provider 处于 disabled，直接在对应能力环中选出健康接替节点；
   - 调用 load_balancer.select_route(...) 做出智能决策：
     * degrade: 短 Prompt (< 300 tokens) 且无思考需求时，若旗舰环繁忙，自动降级至 SPEED 环 (如 Qwen3.6-Flash, doubao-lite)；
     * shed: 目标节点接近并发高水位或近期错误率上升，同环打散；
     * normal: 正常通过。

3. 执行与度量采集 (Execution & Metrics Tracking):
   - 进入前调用 load_balancer.acquire(provider_key)；
   - 在首包到达时记录首包耗时 load_balancer.record_ttft(provider_key, ttft_ms)；
   - 流结束或请求异常时调用 load_balancer.release(provider_key, success, is_rate_limit)；
   - 保证在异常或客户端断开连接时，并发计数必定安全释放。

4. 响应透传 (Response Headers):
   - 透传 x-load-balancing-action, x-load-balancing-reason, x-dispatched-provider, x-dispatched-model。
```

### 2.3 诊断与监控接口升级 (/v1/diagnostics)
- 暴露 `disabled_providers` 列表与节点屏蔽状态；
- 将 `load_balancer.get_load_stats()` 聚合输出到 `/v1/diagnostics` 中的 `load_balancer` 字段，展示各节点的 QPS、活跃并发、平均 TTFT、降级次数与打散次数。

---

## 3. 测试与验证策略

1. **单元测试与边界覆盖**：
   - 验证 `DISABLED_PROVIDERS` 环境变量对 `CooldownTracker` 与 `FailoverRouter` 的过滤生效；
   - 验证 `server.py` 对禁用节点的零网络开销即时转移；
   - 验证 `acquire` / `release` 在正常完成与异常熔断下的计数量平衡；
2. **端到端集成测试**：
   - 模拟指定 `deepseek-chat` 请求，断言其无痛转移到 `qwen` / `doubao` 并附带正确的负载审计响应头；
   - 运行全部现有 10 个测试套件，确保 100% 回归通过。
