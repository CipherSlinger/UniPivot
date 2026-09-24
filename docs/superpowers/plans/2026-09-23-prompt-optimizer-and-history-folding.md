# 提示词自适应压缩与历史折叠算法实现计划 (OPT-002)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 增强并装配 `PromptOptimizer` 自适应折叠管线至网关全协议链路（OpenAI Chat / Anthropic Messages / OpenAI Responses），在超长会话中安全修剪历史工具输出并透传审计响应头，大幅降低 Token 消耗并规避上游载荷超限。

**Architecture:** 
1. 在 `providers/prompt_optimizer.py` 中封装 `PromptOptimizer` 类，提供线程安全度量统计 (`get_stats`)、分级折叠入口 (`optimize_chat_messages`) 及响应头生成函数 (`make_prompt_folding_headers`)；
2. 在 `server.py` 中将 `prompt_optimizer` 接入 `/v1/chat/completions`、`/v1/messages` 与 `/v1/responses` 入口，在模型调度和调用前执行优化并透传响应头；
3. 在 `/v1/diagnostics` 与 `/health` 接口中输出 `prompt_optimizer` 状态与累计节约数据；
4. 保证头部系统指令、首条用户任务与尾部工具配对绝对安全不变性。

**Tech Stack:** Python 3.10+, FastAPI, Starlette, Pydantic, HTTPX, asyncio

**Spec:** `docs/superpowers/specs/2026-09-23-prompt-optimizer-and-history-folding-design.md`

## Global Constraints

- 不可引入额外的第三方重量级依赖；
- 保持对现有多协议（OpenAI Chat / Anthropic Messages / OpenAI Responses）的完全向下兼容；
- 必须通过已有全部单测（tests/*.py），不产生任何回归问题；
- 严格遵循 RFC 规范，所有自定义 HTTP 响应头值必须为 ASCII 兼容格式；
- 遇到 `GATEWAY_DISABLE_PROMPT_FOLDING=1` 时应以零损耗穿透跳过。

---

### Task 1: 完善 PromptOptimizer 类、统计度量与响应头构建器

**Files:**
- Modify: `providers/prompt_optimizer.py`
- Test: `tests/test_prompt_optimizer.py`

**Interfaces:**
- Consumes: `fold_history`, `estimate_history_tokens`, `fold_single_message`
- Produces:
  - `PromptOptimizer` 类 (支持 `optimize_chat_messages(messages, max_tokens=None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]`)
  - `PromptOptimizer.get_stats() -> Dict[str, Any]`
  - `make_prompt_folding_headers(meta: Dict[str, Any]) -> Dict[str, str]`
  - 全局默认单例 `prompt_optimizer = PromptOptimizer.get_instance()`

- [ ] **Step 1: 在 tests/test_prompt_optimizer.py 中编写测试用例验证 PromptOptimizer 与响应头构建**

```python
def test_prompt_optimizer_class_and_headers(self):
    from providers.prompt_optimizer import PromptOptimizer, make_prompt_folding_headers
    opt = PromptOptimizer(max_tokens=1000, preserve_recent_turns=2)
    messages = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is 1+1?"},
    ]
    optimized, meta = opt.optimize_chat_messages(messages)
    self.assertFalse(meta["applied"])
    self.assertEqual(meta["saved_tokens"], 0)
    
    headers = make_prompt_folding_headers(meta)
    self.assertEqual(headers["x-prompt-folding-applied"], "false")
    self.assertEqual(headers["x-prompt-folding-saved-tokens"], "0")
    
    stats = opt.get_stats()
    self.assertIn("total_inspections", stats)
    self.assertEqual(stats["total_inspections"], 1)
```

- [ ] **Step 2: 运行单测确认失败**

Run: `.venv/bin/python -m unittest tests/test_prompt_optimizer.py`
Expected: FAIL (ImportError: cannot import name 'PromptOptimizer')

- [ ] **Step 3: 在 providers/prompt_optimizer.py 中实现 PromptOptimizer 与 make_prompt_folding_headers**

在 `providers/prompt_optimizer.py` 中实现:
1. `PromptOptimizer` 类，拥有 `optimize_chat_messages`、`get_stats`、`reset_stats`、`get_instance` 等方法；
2. 内部维护线程安全的累计指标（`total_inspections`, `total_folded_requests`, `total_saved_tokens`, `total_original_tokens`, `total_final_tokens`）；
3. `make_prompt_folding_headers` 函数构造 ASCII 规范的响应头字典；
4. 导出全局单例 `prompt_optimizer`。

- [ ] **Step 4: 运行单测确认通过**

Run: `.venv/bin/python -m unittest tests/test_prompt_optimizer.py`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add providers/prompt_optimizer.py tests/test_prompt_optimizer.py
git commit -m "feat(optimizer): introduce PromptOptimizer class with telemetry and header formatting"
```

---

### Task 2: 在 server.py 主链路中装配 PromptOptimizer 与审计响应头

**Files:**
- Modify: `server.py:1090-2050`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `prompt_optimizer = PromptOptimizer.get_instance()`, `make_prompt_folding_headers`
- Produces: 
  - `/v1/chat/completions`、`/v1/messages` 与 `/v1/responses` 均在调度前自适应折叠；
  - 响应包含 `x-prompt-folding-applied`, `x-prompt-folding-saved-tokens`, `x-prompt-folding-original-tokens`, `x-prompt-folding-final-tokens` 头。

- [ ] **Step 1: 在 tests/test_server.py 中添加折叠响应头与长会话优化单测**

构造带有超长工具调用日志的消息请求，验证返回的响应头包含 `x-prompt-folding-applied`。

- [ ] **Step 2: 运行单测确认失败**

Run: `.venv/bin/python tests/test_server.py`
Expected: FAIL (缺少响应头或未应用折叠)

- [ ] **Step 3: 在 server.py 中实现三协议链路装配**

在 `server.py` 中:
1. 在 `chat_completions` 中调用 `messages, fold_meta = prompt_optimizer.optimize_chat_messages(messages)`，并将 `make_prompt_folding_headers(fold_meta)` 合并到响应头中；
2. 在 `anthropic_messages` 中对转化后的 `gateway_messages` 执行优化并透传响应头；
3. 在 `responses_endpoint` 中对转化后的 `gateway_messages` 执行优化并透传响应头。

- [ ] **Step 4: 运行单测确认通过**

Run: `.venv/bin/python tests/test_server.py`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add server.py tests/test_server.py
git commit -m "feat(server): wire PromptOptimizer and folding headers into chat, messages, and responses endpoints"
```

---

### Task 3: 诊断监控接口 (/v1/diagnostics) 与健康检查 (/health) 数据升级

**Files:**
- Modify: `server.py:840-910`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `prompt_optimizer.get_stats()`
- Produces: `/v1/diagnostics` 返回 `prompt_optimizer` 统计结构与折叠状态。

- [ ] **Step 1: 在 tests/test_server.py 中增加 /v1/diagnostics prompt_optimizer 字段断言**

断言 `/v1/diagnostics` 响应包含 `prompt_optimizer` 字段，且包含 `total_inspections`、`total_folded_requests` 等度量字段。

- [ ] **Step 2: 运行单测确认失败**

Run: `.venv/bin/python tests/test_server.py`
Expected: FAIL (KeyError: 'prompt_optimizer')

- [ ] **Step 3: 在 server.py 中暴露 prompt_optimizer 指标**

在 `get_diagnostics` 函数中，调用 `prompt_optimizer.get_stats()` 并写入返回数据对象。

- [ ] **Step 4: 运行单测确认通过**

Run: `.venv/bin/python tests/test_server.py`
Expected: PASS

- [ ] **Step 5: 提交代码**

```bash
git add server.py tests/test_server.py
git commit -m "feat(diagnostics): expose prompt_optimizer telemetry in diagnostics endpoint"
```

---

### Task 4: 端到端全套单测回归验证与优化报告更新

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

更新 `reports/optimization_proposal.json` 与 `reports/optimization_recommendations.md`，将 `OPT-002` 标记为 `COMPLETED`。

- [ ] **Step 3: 最终提交并推送到 GitHub 远程仓库**

```bash
git add reports/
git commit -m "chore: update optimization reports marking OPT-002 as implemented"
git push origin main
```
