"""JSON 自动容错修复引擎 (Self-Healing JSON Engine)

专门针对大模型输出的非标准、截断、转义缺失、全角标点及 Python 字面量 JSON
进行无损或自愈式修复，保证工具调用 (Tool Call) 与结构化输出的高可用解析。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, List, Optional, Tuple


# 全角引号/标点替换表
_FULLWIDTH_QUOTES = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "：": ":",
    "，": ",",
})

# 尾部多余逗号正则: {"a": 1,} -> {"a": 1}
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")

# 未加引号的键正则: {foo: "bar", _bar_1: 123} -> {"foo": "bar", "_bar_1": 123}
_UNQUOTED_KEY_RE = re.compile(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_.-]*)\s*:', re.MULTILINE)

# Markdown 代码块包裹匹配
_CODE_BLOCK_RE = re.compile(r"^```+(?:[a-zA-Z0-9_.-]*\s*\n)?([\s\S]*?)(?:\n?```+)?$", re.DOTALL)


def strip_markdown_fence(s: str) -> str:
    """剥离字符串外层的 ``` 或 ```json / ```tool_call 代码块包裹。"""
    s_clean = s.strip()
    match = _CODE_BLOCK_RE.match(s_clean)
    if match:
        inner = match.group(1).strip()
        if inner:
            return inner
    return s_clean


def repair_truncated_json(s: str) -> str:
    """修复因 Token 上限截断导致的未闭合引号、括号与花括号。

    扫描括号栈与字符串状态，在末尾依次补全闭合符号。
    """
    stack: List[str] = []
    in_string = False
    escape = False
    quote_char = '"'

    for char in s:
        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if in_string:
            if char == quote_char:
                in_string = False
        else:
            if char in ('"', "'"):
                in_string = True
                quote_char = char
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in ("}", "]"):
                if stack and stack[-1] == char:
                    stack.pop()

    repaired = s
    if in_string:
        repaired += quote_char

    # 补充未闭合的括号
    while stack:
        repaired += stack.pop()

    return repaired


def clean_trailing_commas(s: str) -> str:
    """循环剥离 JSON 对象与数组中尾部多余的逗号：{"a": 1,} -> {"a": 1}。"""
    prev = ""
    curr = s
    while prev != curr:
        prev = curr
        curr = _TRAILING_COMMA_RE.sub(r"\1", curr)
    return curr


def normalize_python_literals(s: str) -> str:
    """安全替换 Python 特有字面量为 JSON 标准字面量 (True/False/None -> true/false/null)。"""
    # 仅在非字符串区间或通过词边界替换
    tokens = re.split(r'("(?:\\.|[^"\\])*")', s)
    for i in range(0, len(tokens), 2):
        chunk = tokens[i]
        chunk = re.sub(r"\bTrue\b", "true", chunk)
        chunk = re.sub(r"\bFalse\b", "false", chunk)
        chunk = re.sub(r"\bNone\b", "null", chunk)
        tokens[i] = chunk
    return "".join(tokens)


def quote_unquoted_keys(s: str) -> str:
    """将未带引号的对象键转换为带引号的标准 JSON 键。"""
    tokens = re.split(r'("(?:\\.|[^"\\])*")', s)
    for i in range(0, len(tokens), 2):
        tokens[i] = _UNQUOTED_KEY_RE.sub(r'\1"\2":', tokens[i])
    return "".join(tokens)


def escape_raw_newlines_in_strings(s: str) -> str:
    """对字符串内部的未转义物理换行符进行 \\n 转义。"""
    out: List[str] = []
    in_string = False
    escape = False

    for char in s:
        if escape:
            escape = False
            out.append(char)
            continue

        if char == "\\":
            escape = True
            out.append(char)
            continue

        if char == '"':
            in_string = not in_string
            out.append(char)
            continue

        if in_string and char == "\n":
            out.append("\\n")
        elif in_string and char == "\r":
            out.append("\\r")
        elif in_string and char == "\t":
            out.append("\\t")
        else:
            out.append(char)

    return "".join(out)


def repair_json_string(raw: str) -> Tuple[str, bool]:
    """对非标准或破损的 JSON 字符串执行渐进式管道修复。

    返回 (修复后的字符串, 是否触发了修复)。
    """
    if not raw or not isinstance(raw, str):
        return "", False

    cleaned = raw.strip()
    cleaned = strip_markdown_fence(cleaned)

    # 1. 如果原生标准合法，零开销直接返回
    try:
        json.loads(cleaned)
        return cleaned, False
    except Exception:
        pass

    # 2. 管道修复
    modified = cleaned

    # 全角标点替换
    if any(q in modified for q in ("“", "”", "‘", "’", "：", "，")):
        modified = modified.translate(_FULLWIDTH_QUOTES)

    # 换行与控制字符转义
    modified = escape_raw_newlines_in_strings(modified)

    # 移除尾部逗号
    modified = clean_trailing_commas(modified)

    # Python 字面量标准化
    modified = normalize_python_literals(modified)

    # 未加引号的键修复
    modified = quote_unquoted_keys(modified)

    # 截断补全
    modified = repair_truncated_json(modified)

    # 再次清理可能补全后残留的逗号
    modified = clean_trailing_commas(modified)

    return modified, True


def safe_loads_with_repair(raw: Any) -> Tuple[Optional[Any], bool]:
    """鲁棒的 JSON 解析与自愈。

    返回 (解析后的 Python 对象, 是否经历过修复自愈)。
    """
    if raw is None:
        return None, False
    if isinstance(raw, (dict, list)):
        return raw, False
    if not isinstance(raw, str):
        return None, False

    s = raw.strip()
    if not s or s in ("{}", "null", "None"):
        return {}, False

    # 快速路径：原生合法 JSON
    try:
        return json.loads(s, strict=False), False
    except Exception:
        pass

    # 尝试管道修复
    repaired_str, _ = repair_json_string(s)
    try:
        return json.loads(repaired_str, strict=False), True
    except Exception:
        pass

    # 尝试 Python AST 安全字面量求值 (针对单引号字典等)
    try:
        evaled = ast.literal_eval(repaired_str)
        if isinstance(evaled, (dict, list)):
            return evaled, True
    except Exception:
        pass

    # 尝试直接对原字符串进行 AST 求值
    try:
        evaled = ast.literal_eval(s)
        if isinstance(evaled, (dict, list)):
            return evaled, True
    except Exception:
        pass

    return None, False
