"""客户端要求结构化输出时，解析端必须兑现 (PROTOCOL)

OpenAI 的 `response_format`（Responses 里的 `text.format` 是同一个东西）是**硬的**
输出契约：`{"type": "json_object"}` 承诺 `message.content` 本身就是一个 JSON 对象，
客户端拿到之后直接 `JSON.parse`。第 35 节只把它翻译成一条提示词——解析端什么也没做。
实测（真实 HTTP 路径，脚本化 provider 返回
`Sure! Here is the data you asked for:\\n\\n```json\\n{"name": "Ada"}\\n```\\nHope this helps!`）：

    /v1/chat/completions -> 200，content 就是上面那段大白话
    /v1/responses        -> 200，output_text 同样是那段大白话

客户端 `JSON.parse` 当场抛错，整轮报废，而且没有任何可读的成因。与 `tool_choice` /
`stop` / `parallel_tool_calls` 完全同一条纪律：**提示词负责降低发生率，解析端负责
保证结果**。

兑现分两步，顺序不能反：

1. **回收**：模型把 JSON 包在围栏或寒暄里时，把那个值取出来。这一步是确定性的，不
   改变模型说了什么——只是把它已经说出的值交给客户端，而不是让客户端对着寒暄崩溃。
2. **诚实**：完全取不出合法 JSON 时，不把大白话伪装成「结构化输出」。客户端明确
   声明了它要 `JSON.parse`，递过去一段非 JSON 等于让它一行之后崩掉；这里直接给出
   `invalid_output` 错误，并带上模型原文摘录，让客户端与运维都能看到真实原因。

## 范围（诚实说明，不假装比自己能做的更多）

* 只处理**明确**声明了 `json_object` / `json_schema` 的请求；`{"type": "text"}` 与
  一切无法识别的形状逐字节不变（失败开放，零回归）。
* 只校验**顶层类型**，不做 JSON Schema 的嵌套校验（不引入 `jsonschema` 依赖）。
  声明了 `schema.type` 时按它校验顶层；`required` 缺键**不**在这里检查——半套
  schema 校验只会制造「验过了」的错觉。
* **截断修复不参与**：`repair_truncated_json` 会给缺失字段补 `null`，那是替模型
  编数据。被截断的答案必须表现为失败，而不是一个看起来完整的对象（第 52 节）。
* 回收只在**顶层**取值，不会去猜「模型想说的其实是哪个对象」：先严格解析，再按
  围栏 / 平衡括号逐个候选严格解析，最后才走**非截断式**的自愈管道。
"""

from __future__ import annotations

import json
from typing import Any, Iterator, List, Optional, Tuple

from .anthropic_compat import response_format_top_level_type
from .json_repair import safe_loads_with_repair, strip_markdown_fence

#: 明确要求结构化输出的 `response_format.type` 取值。
_ENFORCED_MODES = frozenset({"json_object", "json_schema"})


def response_format_mode(response_format: Any) -> Optional[str]:
    """客户端是否**明确**要求 JSON：返回 `json_object` / `json_schema`，否则 `None`。

    未声明、`{"type": "text"}`、以及任何无法识别的形状都返回 `None`——这类请求
    必须逐字节保持既有行为（失败开放）。
    """
    if not isinstance(response_format, dict):
        return None
    rf_type = response_format.get("type")
    return rf_type if rf_type in _ENFORCED_MODES else None


def expected_top_level_type(response_format: Any) -> Optional[str]:
    """`json_schema` 里声明的顶层类型（没有声明或形状无法认识时返回 `None`）。

    实现就在 `anthropic_compat`：提示词也要读同一个字段。定义只能有一份，否则
    「提示词让模型输出 object」和「解析端接受 array」会各自演进（见第 70/78 节）。
    """
    return response_format_top_level_type(response_format)


def _shape_ok(value: Any, mode: str, expected: Optional[str]) -> bool:
    """顶层类型判定。`json_object` 的契约就是「一个对象」，别的一律不算兑现。"""
    if mode == "json_object":
        return isinstance(value, dict)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _strict_loads(text: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


_OPENERS = {"{": "}", "[": "]"}


def _balanced_json_values(text: str) -> Iterator[str]:
    """文本里所有括号平衡的最外层 JSON **值**（对象或数组），按出现顺序产出。

    为什么不复用 `extract_balanced_json_objects`：它只从 `{` 起扫。于是
    `json_schema` 声明 `type: "array"` 的请求——一个完全合法的请求——哪怕模型把
    `[1, 2, 3]` 原样放在围栏里，也一个候选都取不到（实测 `payload is None`，最后
    报 `invalid_output`，把一个正确的回答判成违约）。工具调用那条路径只关心对象，
    语义保持不变，所以这里是本模块自己的一个扫描器，而不是去改它。
    """
    n = len(text or "")
    i = 0
    while i < n:
        opener = text[i]
        closer = _OPENERS.get(opener)
        if closer is None:
            i += 1
            continue
        stack: List[str] = []
        in_string = False
        escape = False
        j = i
        while j < n:
            c = text[j]
            if escape:
                escape = False
            elif c == "\\" and in_string:
                escape = True
            elif c == '"':
                in_string = not in_string
            elif not in_string:
                if c in _OPENERS:
                    stack.append(_OPENERS[c])
                elif c in ("}", "]"):
                    if not stack or stack[-1] != c:
                        # 括号不配对：这一段不可能是同一个 JSON 值，重新起扫
                        break
                    stack.pop()
                    if not stack:
                        yield text[i : j + 1]
                        i = j
                        break
            j += 1
        i += 1


def _candidates(text: str) -> Iterator[str]:
    """「模型可能把 JSON 放在了哪里」的全部候选，按从紧到松排列。"""
    raw = text or ""
    yield raw
    inner = strip_markdown_fence(raw)
    if inner != raw:
        yield inner
    for value in _balanced_json_values(raw):
        yield value


def extract_json_payload(text: Any, response_format: Any) -> Optional[str]:
    """从模型原文里取出一个**客户端 `JSON.parse` 一定成功**的 JSON 文本。

    返回 `None` 表示取不出来（调用方必须据此报错，而不是把原文递过去）。

    两步解析，顺序是刻意的：先把**所有**候选按严格 JSON 试一遍，全部失败后才允许
    自愈管道介入。否则一段严格合法的 JSON 可能输给前面某个「修复出来的」候选。
    严格命中时原样返回候选文本（不改写模型已经给对的字节）；自愈命中时重新序列化，
    因为管道产物本身未必是严格 JSON。
    """
    mode = response_format_mode(response_format)
    if mode is None:
        return None
    expected = expected_top_level_type(response_format)

    # 1. 严格解析：命中的候选原样交付。
    for cand in _candidates(text or ""):
        ok, value = _strict_loads(cand)
        if ok and _shape_ok(value, mode, expected):
            return cand

    # 2. 自愈管道（**不补截断**）：命中的值重新序列化成严格 JSON 再交付。
    for cand in _candidates(text or ""):
        stripped = cand.strip()
        # `safe_loads_with_repair` 把裸 `null` / `None` 当成「没有参数」返回 `{}`，
        # 那是工具调用语义。结构化输出里 `null` 是一个合法的 JSON 值，不能被改写。
        if stripped in ("null", "None"):
            continue
        value, _repaired = safe_loads_with_repair(cand, allow_truncation_repair=False)
        if value is not None and _shape_ok(value, mode, expected):
            try:
                return json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                continue
    return None


def invalid_output_message(text: Any, response_format: Any, limit: int = 200) -> str:
    """「模型没有兑现结构化输出契约」的错误文案，带模型原文摘录。

    摘录是这里唯一有用的证据：客户端/运维看到的是模型**真的**说了什么，而不是
    一句「上游异常」。
    """
    mode = response_format_mode(response_format) or "json"
    excerpt = " ".join(str(text or "").split())
    if not excerpt:
        excerpt = "(空回复)"
    elif len(excerpt) > limit:
        excerpt = excerpt[:limit] + "…"
    return (
        f"上游模型没有按 `response_format` 返回合法 JSON（要求: {mode}）。"
        f"模型原文: {excerpt!r}"
    )


def enforce_json_output(
    text: Any, response_format: Any
) -> Tuple[Optional[str], str]:
    """`(可交付的 JSON 文本, "")` 或 `(None, 错误文案)`。

    调用点只需判第一个返回值：非 `None` 时替换正文，`None` 时按各自的协议报错
    （OpenAI 502 `invalid_output`，Responses `response.failed`）。
    """
    if response_format_mode(response_format) is None:
        return (text if isinstance(text, str) else ""), ""
    payload = extract_json_payload(text, response_format)
    if payload is not None:
        return payload, ""
    return None, invalid_output_message(text, response_format)
