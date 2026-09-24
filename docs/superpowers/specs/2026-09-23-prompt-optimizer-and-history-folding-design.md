# 提示词自适应压缩与历史折叠算法设计规范 (OPT-002)

**文档状态**: 已通过评审 / 设计落地阶段  
**设计负责人**: Product Manager & Lead Architect  
**目标模块**: `providers/prompt_optimizer.py`, `server.py`, `tests/test_prompt_optimizer.py`, `tests/test_server.py`  
**关联优化项**: OPT-002 (Token 经济性与上下文窗口管理)

---

## 1. 业务背景与设计目标

### 1.1 背景痛点
UniPivot 作为多协议 AI 网关，为 Claude Code CLI、OpenAI Codex CLI、Aider 以及各类 Agent 客户端提供长会话中继。在大型代码库重构、日志分析或复杂测试循环中：
1. **上下文急剧膨胀**: 单个会话在经历数十轮 Bash 执行输出、全量文件读取 (`cat` / `Read`) 后，Prompt Tokens 轻易突破 30,000 ~ 100,000 Tokens。
2. **上游网页模型载荷超限与超时**: 各国内提供方（通义千问、豆包、Kimi、GLM）对 Web 端单次请求载荷、传输延迟均有敏感阈值，超大上下文极易诱发 400 Payload Too Large 或 504 Gateway Timeout。
3. **冗余数据浪费**: 开发者关注的是最近若干轮的操作与当下的执行状态，10 轮之前的临时巨量测试日志对当前决策并无增量价值，但全量重传消耗大量网络带宽与计算资源。
4. **粗暴截断的灾难性后果**: 若直接裁掉历史消息，会破坏系统提示词（CLI 规则丢失）、丢失初始任务目标，或者破坏 `tool_calls` 与 `tool_result` 之间的对应关系，导致上游模型报错（例如 400 "tool_call_id not found"）。

### 1.2 设计目标
1. **分级自适应折叠 (Two-Phase Adaptive Folding)**:
   - **阶段一 (Phase 1: Tool & Code Block Truncation)**: 针对中间轮次的超长工具输出与巨型代码块，就地替换为保留元数据的折叠标记：`[已折叠历史执行结果: 原 {orig_len} 字符, 状态正常]`。
   - **阶段二 (Phase 2: Conversation Convergence)**: 若阶段一折叠后仍超过上限，将老旧中间轮次收敛压缩为单一紧凑提示：`{"role": "user", "content": "[前序对话历史已由网关自适应折叠，保留核心结论与��下文]"}`。
2. **关键状态与语义严格保护 (Strict Invariants)**:
   - **头部保护**: 严格保留开头所有连续的 `system` 消息（CLI 规则与 Tool 定制 Prompt），以及首条 `user` 消息（初始任务核心诉求）。
   - **尾部保护**: 严格保留最近 `preserve_recent_turns`（默认 4 轮）的上下文完整性。
   - **工具状态机事务闭环 (Tool Transaction Balance)**: 绝不在 `tool_calls` 与其对应的 `tool` / `tool_result` 中间切割，必须成对保留或成对处理。
3. **全协议全链路透明装配**:
   - 覆盖 `/v1/chat/completions`、`/v1/messages` 与 `/v1/responses`。
   - 透传标准审计响应头：`x-prompt-folding-applied`, `x-prompt-folding-saved-tokens`, `x-prompt-folding-original-tokens`, `x-prompt-folding-final-tokens`。
4. **配置化与运行时诊断**:
   - 支持环境变量开关 `GATEWAY_DISABLE_PROMPT_FOLDING=1`、容量阈值 `GATEWAY_MAX_CONTEXT_TOKENS` (默认 24000) 与最近保留轮次 `GATEWAY_PRESERVE_RECENT_TURNS` (默认 4)。
   - 在 `/v1/diagnostics` 与 `/health` 中暴露全局折叠度量（检查总数、折叠触发数、累计节约 Tokens 等）。

---

## 2. 架构设计与状态机

### 2.1 提示词折叠管线流转图

```
                Incoming Request (/chat/completions | /messages | /responses)
                                      │
                                      ▼
                      Normalize to Gateway Messages List
                                      │
                                      ▼
                        PromptOptimizer.optimize(...)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   Tokens <= max_tokens or Disabled          Tokens > max_tokens
                 │                                         │
        Bypass (applied=False)               Phase 1: Fold Intermediate Tool/Code
                 │                                         │
                 │                            ┌────────────┴────────────┐
                 │                            ▼                         ▼
                 │                    Tokens <= max_tokens    Tokens > max_tokens
                 │                            │                         │
                 │                    Done (Phase 1)             Phase 2: Converge
                 │                            │                         │
                 └────────────────────────────┬─────────────────────────┘
                                              ▼
                             Assemble Headers & Update Metrics
                              (x-prompt-folding-applied, ...)
                                              │
                                              ▼
                             AdaptiveLoadBalancer Route Selection
                                              │
                                              ▼
                                Upstream Failover Execution
```

### 2.2 消息保留与折叠区间划分

在 `messages` 数组中，划分三个明确区域：

```
[ Head (Preserved) ]           [ Intermediate (Foldable) ]         [ Tail (Preserved) ]
┌────────────────────┐        ┌───────────────────────────┐       ┌────────────────────┐
│ Index 0: system    │        │ Turn 2: assistant         │       │ Turn N-3: user     │
│ Index 1: user (1st)│ ───��>  │ Turn 2: tool (oversized)  │ ────> │ Turn N-2: assistant│
└────────────────────┘        │ Turn 3: assistant         │       │ Turn N-1: tool     │
                              │ Turn 3: tool (oversized)  │       │ Turn N: user       │
                              └───────────────────────────┘       └────────────────────┘
                                            │                                │
                                  Stage 1: Truncate Tool              Tool-pair Balance:
                                  Stage 2: Single Summary             Never cut orphan!
```

---

## 3. 详细接口设计

### 3.1 `PromptOptimizer` 类 (`providers/prompt_optimizer.py`)

```python
class PromptOptimizer:
    """线程安全的提示词自适应压缩与历史折叠全局管理器。"""

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        preserve_recent_turns: Optional[int] = None,
        disabled: Optional[bool] = None,
    ):
        ...

    def optimize_chat_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """执行消息列表优化。
        
        返回: (optimized_messages, meta)
        meta 包含:
          - applied: bool
          - original_tokens: int
          - final_tokens: int
          - saved_tokens: int
          - phase: int (0: none, 1: tool/code fold, 2: summary convergence)
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """返回度量指标：
        {
            "total_inspections": int,
            "total_folded_requests": int,
            "total_saved_tokens": int,
            "total_original_tokens": int,
            "total_final_tokens": int,
            "max_tokens": int,
            "preserve_recent_turns": int,
            "folding_enabled": bool
        }
        """
        ...
```

### 3.2 审计响应头规范 (`_make_prompt_folding_headers`)

```python
def make_prompt_folding_headers(meta: Dict[str, Any]) -> Dict[str, str]:
    """生成符合 RFC 规范的折叠审计响应头。"""
    applied = meta.get("applied", False)
    return {
        "x-prompt-folding-applied": "true" if applied else "false",
        "x-prompt-folding-saved-tokens": str(meta.get("saved_tokens", 0)),
        "x-prompt-folding-original-tokens": str(meta.get("original_tokens", 0)),
        "x-prompt-folding-final-tokens": str(meta.get("final_tokens", 0)),
    }
```

### 3.3 主服务装配点 (`server.py`)

1. **初始化**:
   - `prompt_optimizer = PromptOptimizer()` 全局实例。
2. **`/v1/chat/completions`**:
   - 在估算 prompt tokens 之前，执行 `messages, fold_meta = prompt_optimizer.optimize_chat_messages(messages)`。
   - 生成 `fold_headers = make_prompt_folding_headers(fold_meta)`，合并至流式与非流式响应头。
3. **`/v1/messages`**:
   - 在 `convert_anthropic_to_gateway_messages` 之后，执行 `gateway_messages, fold_meta = prompt_optimizer.optimize_chat_messages(gateway_messages)`。
   - 合并 `fold_headers` 至流式与非流式响应头。
4. **`/v1/responses`**:
   - 在 `convert_responses_to_gateway_messages` 之后，执行 `gateway_messages, fold_meta = prompt_optimizer.optimize_chat_messages(gateway_messages)`。
   - 合并 `fold_headers` 至流式与非流式响应头。
5. **`/v1/diagnostics` 与 `/health`**:
   - 增加 `prompt_optimizer` 状态字段输出。

---

## 4. 测试与验证标准

1. **算法单测 (`tests/test_prompt_optimizer.py`)**:
   - 覆盖短文本未超限穿透。
   - 覆盖阶段一超长工具输出与长代码块截断。
   - 覆盖阶段二极限收敛为统一摘要标记。
   - 覆盖首部与尾部工具配对平衡不变性。
   - 覆盖度量指标统计累加与 `GATEWAY_DISABLE_PROMPT_FOLDING` 禁用逻辑。
2. **端到端集成测试 (`tests/test_server.py`)**:
   - 请求超长会话（模拟多轮 CLI 执行），验证 `x-prompt-folding-applied: true` 与 `x-prompt-folding-saved-tokens > 0`。
   - 验证 `/v1/messages` 与 `/v1/responses` 均能正确触发折叠与响应头透传。
   - 验证 `/v1/diagnostics` 正确返回 `prompt_optimizer` 统计结构。
3. **回归验证**:
   - 确保全套 10 个测试套件全部 PASS，无任何功能与兼容性回归。
