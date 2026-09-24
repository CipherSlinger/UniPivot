"""工具调用与自愈 JSON 引擎单测 (OPT-006)

运行: .venv/bin/python tests/test_tools.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient
import server as srv
from providers.json_repair import (
    clean_trailing_commas,
    escape_raw_newlines_in_strings,
    normalize_python_literals,
    quote_unquoted_keys,
    repair_json_string,
    repair_truncated_json,
    safe_loads_with_repair,
    strip_markdown_fence,
)
from providers.tool_streamer import (
    SpeculativeToolStreamer,
    parse_tool_calls_with_healing,
)


def test_json_repair_unit():
    """测试各类 LLM 常见残损/变体 JSON 的容错修复。"""
    # 1. 干净的标准 JSON，零修复直接解析
    data, healed = safe_loads_with_repair('{"name": "exec_cmd", "cmd": "ls -la"}')
    assert data == {"name": "exec_cmd", "cmd": "ls -la"}
    assert not healed

    # 2. 中文全角符号转换
    cn_json = '{\n  “name”： “query_db”，\n  “params”： {“limit”： 10}\n}'
    data, healed = safe_loads_with_repair(cn_json)
    assert data == {"name": "query_db", "params": {"limit": 10}}
    assert healed

    # 3. 尾部逗号容错
    trailing_json = '{"items": [1, 2, 3,], "status": "ok",}'
    data, healed = safe_loads_with_repair(trailing_json)
    assert data == {"items": [1, 2, 3], "status": "ok"}
    assert healed

    # 4. Python 字面量 (True/False/None)
    py_json = '{"is_active": True, "has_error": False, "extra": None}'
    data, healed = safe_loads_with_repair(py_json)
    assert data == {"is_active": True, "has_error": False, "extra": None}
    assert healed

    # 5. 未加双引号的健名
    unquoted_json = '{name: "create_file", path: "/tmp/a.txt", overwrite: true}'
    data, healed = safe_loads_with_repair(unquoted_json)
    assert data == {"name": "create_file", "path": "/tmp/a.txt", "overwrite": True}
    assert healed

    # 6. 字符串内部裸物理换行符
    raw_newlines = '{"code": "def hello():\n    print(\'hi\')\n"}'
    data, healed = safe_loads_with_repair(raw_newlines)
    assert "def hello():" in data["code"]

    # 7. 截断 JSON 补全闭合
    truncated = '{"name": "eval", "arguments": {"expr": "1 + 1"'
    data, healed = safe_loads_with_repair(truncated)
    assert data["name"] == "eval"
    assert data["arguments"]["expr"] == "1 + 1"
    assert healed

    # 8. Markdown 代码块包裹
    md_wrapped = "```json\n{\"action\": \"view\", \"id\": 123}\n```"
    data, healed = safe_loads_with_repair(md_wrapped)
    assert data == {"action": "view", "id": 123}

    print("[PASS] test_json_repair_unit: JSON 容错修复引擎各项自愈规则验证通过")


def test_tool_streamer_logic():
    """测试推测式流式分发状态机与零延迟思考流。"""
    # 场景 1: 纯文本流，安全发射
    streamer = SpeculativeToolStreamer(tools_enabled=True)
    actions = []
    for chunk in ["今天", "的天气", "非常晴朗，适合户外运动，", "建议去公园散步。"]:
        actions.extend(streamer.feed_chunk(chunk))
    rem, tools, healed = streamer.finalize()
    assert tools is None
    all_text = "".join(a.content for a in actions if a.action_type == "text_delta") + rem
    assert "今天的天气非常晴朗" in all_text

    # 场景 2: 思考流零延迟透传 + 随后输出工具调用
    streamer2 = SpeculativeToolStreamer(tools_enabled=True)
    # 思考流立即以 reasoning_delta 发射
    r_act = streamer2.feed_chunk("用户想查询文件，我应该调用 Read 工具。", is_reasoning=True)
    assert len(r_act) == 1
    assert r_act[0].action_type == "reasoning_delta"
    assert r_act[0].content == "用户想查询文件，我应该调用 Read 工具。"

    # 文本前缀 + ```tool_call
    streamer2.feed_chunk("我为您查询：\n```tool_call\n")
    streamer2.feed_chunk('{"name": "Read", "arguments": {"file_path": "/etc/hosts"}}\n```')
    rem, tools, healed = streamer2.finalize()
    assert tools is not None
    assert len(tools) == 1
    assert tools[0][0] == "Read"
    assert tools[0][1] == {"file_path": "/etc/hosts"}

    # 场景 3: 触发包含截断自愈的工具调用
    broken_tool_text = '我正在执行操作：\n```json\n{"name": "deploy", "arguments": {"env": "prod", "version": "v1.2"'
    parsed_res, was_healed = parse_tool_calls_with_healing(broken_tool_text)
    assert parsed_res is not None
    prefix, tool_list = parsed_res
    assert prefix == "我正在执行操作："
    assert tool_list[0][0] == "deploy"
    assert tool_list[0][1]["version"] == "v1.2"
    assert was_healed

    print("[PASS] test_tool_streamer_logic: 推测式流式分发与思考流透传验证通过")


class MockToolProvider:
    """提供标准/破损工具调用回显的假 Provider。"""

    def __init__(self, mode: str = "clean"):
        self.mode = mode

    async def chat(self, *a, **kw):
        if self.mode == "clean":
            yield "我将帮您查询文件内容。\n", {}
            yield "```tool_call\n", {}
            yield '{"name": "ReadFile", "arguments": {"path": "/app/config.json"}}\n', {}
            yield "```", {}
        elif self.mode == "healed":
            # 带全角符号和尾部逗号以及截断的劣质输出
            yield "执行修复诊断：\n", {}
            yield "<tool_call>\n", {}
            yield '{\n  “name”： “Diagnose”，\n  “params”： {“level”： “deep”，}\n', {}
        elif self.mode == "reasoning_then_tool":
            yield "思考中...", {"reasoning": True}
            yield "好的，调用工具中：\n", {}
            yield '```tool_call\n{"name": "Bash", "arguments": {"command": "ls"}}\n```', {}


def test_server_tool_streaming_and_telemetry():
    """测试服务端全协议流式工具调用、增量输出与响应头遥测。"""
    client = TestClient(srv.app)
    orig_factory = srv._make_provider

    try:
        # 1. OpenAI 协议流式工具调用
        srv._make_provider = lambda k: MockToolProvider("clean")
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "读文件"}],
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "ReadFile"}}],
            },
        ) as resp:
            assert resp.status_code == 200
            content = "".join(resp.iter_text())
            assert "tool_calls" in content
            assert "ReadFile" in content

        # 2. Anthropic 协议非流式工具调用与自愈响应头
        srv._make_provider = lambda k: MockToolProvider("healed")
        resp_ant = client.post(
            "/v1/messages",
            headers={"x-api-key": "test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "诊断"}],
                "tools": [{"name": "Diagnose", "description": "run diagnose"}],
            },
        )
        assert resp_ant.status_code == 200
        ant_json = resp_ant.json()
        assert resp_ant.headers.get("x-tool-calls-count") == "1"
        assert resp_ant.headers.get("x-tool-healing") == "1"
        tool_blocks = [c for c in ant_json["content"] if c["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "Diagnose"

        # 3. Responses 协议非流式工具调用
        srv._make_provider = lambda k: MockToolProvider("clean")
        resp_res = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4o",
                "input": "读取配置",
                "tools": [{"type": "function", "name": "ReadFile"}],
            },
        )
        assert resp_res.status_code == 200
        res_json = resp_res.json()
        assert resp_res.headers.get("x-tool-calls-count") == "1"
        assert any(item.get("type") == "function_call" for item in res_json.get("output", []))

        # 4. Diagnostics 接口指标反映
        diag_resp = client.get("/v1/diagnostics")
        assert diag_resp.status_code == 200
        diag = diag_resp.json()
        assert "tool_calling" in diag
        assert diag["tool_calling"]["total_tool_calls"] >= 2
        assert diag["tool_calling"]["healed_tool_calls"] >= 1

        print("[PASS] test_server_tool_streaming_and_telemetry: 全协议流式工具调用与自愈遥测验证通过")
    finally:
        srv._make_provider = orig_factory


if __name__ == "__main__":
    test_json_repair_unit()
    test_tool_streamer_logic()
    test_server_tool_streaming_and_telemetry()
    print("\n[ALL PASS] OPT-006 工具调用增量分发与自愈引擎测试全部通过！")
