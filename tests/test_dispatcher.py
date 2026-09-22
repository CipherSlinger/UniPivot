"""智能体研发任务分发与成果验收评估系统单元测试 (tests/test_dispatcher.py)。

覆盖验证：
1. 智能体环境探测别名与命令构造 (find_agent_cli, build_command, prepare_environment)。
2. 环境变量注入 (Claude: ANTHROPIC_BASE_URL / Codex: OPENAI_BASE_URL)。
3. 成果物检查与回归测试 (inspect_artifacts, run_regression_tests)。
4. 任务调度与后台子进程执行 (dispatch_task, logs/tasks 日志记录, exit_code 状态跃迁)。
5. 优化空间评估器 (evaluate_optimization_opportunities, JSON + Markdown 报告生成)。
6. 服务端路由 (/v1/tasks/dispatch, /v1/tasks/{task_id}, /v1/tasks/evaluation 及无 /v1 前缀别名)。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from healing.agent_runner import AgentInfo, find_agent_cli
from healing.task_dispatcher import TaskDispatcher, TaskEvaluator, TaskResult
import server as srv


def test_agent_cli_and_dispatcher_env():
    """验证智能体 CLI 探测及针对 Claude/Codex 的环境变量注入与命令构造。"""
    # 1. 验证 find_agent_cli 别名
    with patch("healing.agent_runner.find_agent", return_value=AgentInfo("claude", "/mock/bin/claude", "2.1")):
        cli = find_agent_cli()
        assert cli is not None
        assert cli.name == "claude"
        assert cli.path == "/mock/bin/claude"

    dispatcher = TaskDispatcher(port=8088, gateway_api_key="rnd-secret-key")

    # 2. Claude 环境注入与命令构造
    claude_env = dispatcher.prepare_environment("claude")
    assert claude_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8088"
    assert claude_env["ANTHROPIC_API_KEY"] == "rnd-secret-key"
    assert claude_env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    claude_cmd = dispatcher.build_command(
        AgentInfo("claude", "/bin/claude"),
        instruction="优化代码",
        model_alias="qwen3.7-max",
    )
    assert claude_cmd == ["/bin/claude", "-p", "优化代码", "--model", "qwen3.7-max"]

    # 3. Codex 环境注入与命令构造
    codex_env = dispatcher.prepare_environment("codex")
    assert codex_env["OPENAI_BASE_URL"] == "http://127.0.0.1:8088/v1"
    assert codex_env["OPENAI_API_KEY"] == "rnd-secret-key"

    codex_cmd = dispatcher.build_command(
        AgentInfo("codex", "/bin/codex"),
        instruction="优化代码",
        model_alias="default",
    )
    assert codex_cmd == ["/bin/codex", "exec", "优化代码"]


def test_evaluator_artifacts_and_tests():
    """验证成果物检测与本地回归测试机制。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        evaluator = TaskEvaluator(project_root=tmp_path)

        # 创建测试文件与目录
        f1 = tmp_path / "report.md"
        f1.write_text("Hello Report", encoding="utf-8")
        d1 = tmp_path / "subdir"
        d1.mkdir()
        (d1 / "item.txt").write_text("item", encoding="utf-8")

        # 检查成果物
        artifacts = evaluator.inspect_artifacts(["report.md", "subdir", "non_existent.txt"])
        assert artifacts["report.md"]["exists"] is True
        assert artifacts["report.md"]["size_bytes"] > 0
        assert artifacts["subdir"]["exists"] is True
        assert artifacts["subdir"]["file_count"] == 1
        assert artifacts["non_existent.txt"]["exists"] is False

        # 测试回归测试执行 (Mock subprocess)
        async def _test_async():
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                # 场景 1: 回归测试全部通过
                mock_proc = AsyncMock()
                mock_proc.communicate.return_value = (b"[PASS] All passed", b"")
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                passed, out = await evaluator.run_regression_tests()
                assert passed is True
                assert "[PASS]" in out

                # 场景 2: 任务综合评估
                task = TaskResult(
                    task_id="t1",
                    instruction="测试指令",
                    agent_name="claude",
                    model_alias="qwen",
                    status="completed",
                    created_at=time.time(),
                    exit_code=0,
                )
                res = await evaluator.evaluate_task(
                    task=task,
                    run_tests=True,
                    expected_artifacts=["report.md"],
                )
                assert res["overall_success"] is True
                assert res["artifacts_passed"] is True
                assert res["tests_passed"] is True

                # 场景 3: 产出物缺失导致评估失败
                res_fail = await evaluator.evaluate_task(
                    task=task,
                    run_tests=False,
                    expected_artifacts=["missing_file.txt"],
                )
                assert res_fail["overall_success"] is False
                assert res_fail["artifacts_passed"] is False

        asyncio.run(_test_async())


def test_optimization_opportunities_evaluation():
    """验证优化空间建议书生成（JSON + Markdown）。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        reports_dir = tmp_path / "reports"
        with patch("healing.task_dispatcher.REPORTS_DIR", reports_dir):
            evaluator = TaskEvaluator(project_root=tmp_path)
            report = evaluator.evaluate_optimization_opportunities(tasks=[])

            assert report["title"] == "项目优化空间与迭代需求建议书"
            assert "model_metrics" in report
            assert len(report["optimization_opportunities"]) >= 4

            # 验证报告文件是否被持久化写入
            json_file = reports_dir / "optimization_recommendations.json"
            md_file = reports_dir / "optimization_recommendations.md"
            assert json_file.exists()
            assert md_file.exists()

            with open(json_file, "r", encoding="utf-8") as f:
                saved_json = json.load(f)
                assert saved_json["title"] == report["title"]

            with open(md_file, "r", encoding="utf-8") as f:
                md_text = f.read()
                assert "# 项目优化空间与迭代需求建议书" in md_text
                assert "OPT-001" in md_text


def test_task_dispatcher_lifecycle():
    """验证任务调度器的完整生命周期（创建、后台运行、日志捕获、验收评估）。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        logs_tasks = tmp_path / "logs" / "tasks"
        with patch("healing.task_dispatcher.TASKS_LOG_DIR", logs_tasks), \
             patch("healing.task_dispatcher.LOGS_DIR", tmp_path / "logs"):

            dispatcher = TaskDispatcher(port=9000, project_root=tmp_path)

            async def _test():
                # 场景 1: 未找到智能体 CLI
                with patch("healing.task_dispatcher.find_agent_cli", return_value=None):
                    t_failed = await dispatcher.dispatch_task(instruction="写代码")
                    assert t_failed.status == "failed"
                    assert "未探测到" in (t_failed.error_message or "")

                # 场景 2: 正常执行
                mock_agent = AgentInfo("claude", "/bin/claude", "2.1")
                with patch("healing.task_dispatcher.find_agent_cli", return_value=mock_agent), \
                     patch.object(dispatcher, "execute_task_subprocess", AsyncMock(return_value=0)), \
                     patch.object(dispatcher.evaluator, "run_regression_tests", AsyncMock(return_value=(True, "Tests OK"))):

                    task = await dispatcher.dispatch_task(
                        instruction="优化系统架构",
                        model_alias="claude-3-7-sonnet",
                        run_evaluation_tests=True,
                    )
                    assert task.status in ("pending", "running")

                    # 等待微任务协程执行
                    await asyncio.sleep(0.05)

                    assert task.status == "completed"
                    assert task.exit_code == 0
                    assert task.evaluation is not None
                    assert task.evaluation["overall_success"] is True

                    # 验证查询接口
                    t_info = dispatcher.get_task(task.task_id)
                    assert t_info is not None
                    assert t_info["task_id"] == task.task_id
                    assert "log_content" in t_info

                    all_tasks = dispatcher.list_tasks()
                    assert len(all_tasks) >= 2

            asyncio.run(_test())


def test_server_tasks_endpoints():
    """验证服务端的 /v1/tasks/dispatch, /v1/tasks/{id}, /v1/tasks/evaluation 接口。"""
    client = TestClient(srv.app)

    # 1. 提交空指令报错 400
    r_empty = client.post("/v1/tasks/dispatch", json={"instruction": "   "})
    assert r_empty.status_code == 400
    assert r_empty.json()["error"]["type"] == "invalid_request_error"

    # 2. 正常提交任务（Mock 智能体执行）
    mock_agent = AgentInfo("claude", "/bin/claude", "2.1")
    with patch("healing.task_dispatcher.find_agent_cli", return_value=mock_agent), \
         patch.object(srv.task_dispatcher, "execute_task_subprocess", AsyncMock(return_value=0)), \
         patch.object(srv.task_dispatcher.evaluator, "run_regression_tests", AsyncMock(return_value=(True, "OK"))):

        r_dispatch = client.post(
            "/v1/tasks/dispatch",
            json={
                "instruction": "审查 providers/ 下五大模型的重试机制",
                "model": "qwen3.7-max",
                "run_tests": False,
            },
        )
        assert r_dispatch.status_code == 200
        task_data = r_dispatch.json()
        task_id = task_data["task_id"]
        assert task_id.startswith("task_")

        # 3. 查询任务详情
        r_get = client.get(f"/v1/tasks/{task_id}")
        assert r_get.status_code == 200
        assert r_get.json()["task_id"] == task_id

        # 4. 查询不存在的任务 404
        r_404 = client.get("/v1/tasks/task_not_exist_xyz")
        assert r_404.status_code == 404
        assert r_404.json()["error"]["type"] == "task_not_found"

        # 5. 获取最新成果验收与优化空间评估报告
        r_eval = client.get("/v1/tasks/evaluation")
        assert r_eval.status_code == 200
        eval_data = r_eval.json()
        assert eval_data["title"] == "项目优化空间与迭代需求建议书"
        assert "optimization_opportunities" in eval_data
        assert "markdown_report" in eval_data

        # 6. 验证无 /v1 前缀别名路由
        r_alias_dispatch = client.post(
            "/tasks/dispatch",
            json={"instruction": "别名提交任务", "run_tests": False},
        )
        assert r_alias_dispatch.status_code == 200
        alias_task_id = r_alias_dispatch.json()["task_id"]

        r_alias_get = client.get(f"/tasks/{alias_task_id}")
        assert r_alias_get.status_code == 200

        r_alias_eval = client.get("/tasks/evaluation")
        assert r_alias_eval.status_code == 200


if __name__ == "__main__":
    test_agent_cli_and_dispatcher_env()
    print("[PASS] 智能体 CLI 探测、环境注入与命令构造")
    test_evaluator_artifacts_and_tests()
    print("[PASS] 成果物检查与回归测试评估")
    test_optimization_opportunities_evaluation()
    print("[PASS] 优化空间建议书生成 (JSON + Markdown)")
    test_task_dispatcher_lifecycle()
    print("[PASS] 任务调度生命周期与后台日志捕获")
    test_server_tasks_endpoints()
    print("[PASS] 服务端任务管理端点集成测试")
    print("\n研发任务分发与成果验收系统全部单测通过！")
