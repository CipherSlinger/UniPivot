"""远端临时会话删除的持久化重试队列 (OPT-016) 测试套件

验证：
1. 入队即落盘（POSIX 原子写，文件 0o600）
2. 删除成功后出队
3. 删除失败后保留在队列并累计尝试次数
4. 达到最大尝试次数后放弃（有界重试，队列不无界增长）
5. 进程重启后从磁盘恢复队列（跨重启不丢失待删除会话）
6. drain_once 用 resolver 重建 provider 并重试删除
7. GATEWAY_DISABLE_CONVERSATION_GC=1 时退化为历史 fire-and-forget 行为
8. 无事件循环时 schedule 仅入队不炸（等待后台 drain）
9. server 侧接线：dispatched_provider 路由到队列、resolver 已注入、诊断暴露

运行: .venv/bin/python tests/test_conversation_gc.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ.pop("GATEWAY_DISABLE_CONVERSATION_GC", None)

import providers.conversation_gc as cg_mod  # noqa: E402
from providers.conversation_gc import ConversationGC  # noqa: E402


class FakeProvider:
    """可控删除结果的假 Provider。"""

    def __init__(self, result=True, raises=None):
        self.result = result
        self.raises = raises
        self.deleted = []

    async def delete_conversation(self, conversation_id):
        if self.raises is not None:
            raise self.raises
        self.deleted.append(conversation_id)
        return self.result


def _fresh(tmp: Path) -> ConversationGC:
    return ConversationGC(directory=tmp)


# ------------------------------------------------------------------ 1. 落盘

def test_enqueue_persists_atomically():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    gc.enqueue("qwen", "conv-abc")

    assert gc.queue_path.exists(), "入队后必须落盘"
    mode = stat.S_IMODE(gc.queue_path.stat().st_mode)
    assert mode == 0o600, f"队列文件权限必须为 0o600，实际 {oct(mode)}"
    dirmode = stat.S_IMODE(tmp.stat().st_mode)
    assert dirmode == 0o700, f"队列目录权限必须为 0o700，实际 {oct(dirmode)}"

    data = json.loads(gc.queue_path.read_text(encoding="utf-8"))
    assert len(data["pending"]) == 1
    assert data["pending"][0]["conversation_id"] == "conv-abc"
    assert data["pending"][0]["provider"] == "qwen"
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_enqueue_persists_atomically: PASSED")


def test_enqueue_is_idempotent():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    gc.enqueue("qwen", "dup")
    gc.enqueue("qwen", "dup")
    assert gc.get_diagnostics()["pending_count"] == 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_enqueue_is_idempotent: PASSED")


# ------------------------------------------------------------------ 2/3. 成功与失败

def test_success_removes_from_queue():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    provider = FakeProvider(result=True)
    gc.enqueue("kimi", "conv-ok")
    asyncio.run(gc._attempt(provider, "kimi", "conv-ok"))

    assert gc.get_diagnostics()["pending_count"] == 0, "删除成功后必须出队"
    assert provider.deleted == ["conv-ok"]
    assert gc.get_diagnostics()["stats"]["deleted"] == 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_success_removes_from_queue: PASSED")


def test_failure_keeps_entry_with_attempts():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    provider = FakeProvider(result=False)
    gc.enqueue("glm", "conv-fail")
    asyncio.run(gc._attempt(provider, "glm", "conv-fail"))

    diag = gc.get_diagnostics()
    assert diag["pending_count"] == 1, "删除失败必须保留在队列等待重试"
    assert diag["pending"][0]["attempts"] == 1
    assert "falsy" in diag["pending"][0]["last_error"]
    assert diag["stats"]["retried"] == 1

    # 异常同样计入失败
    tmp2 = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc2 = _fresh(tmp2)
    provider2 = FakeProvider(raises=RuntimeError("boom"))
    gc2.enqueue("glm", "conv-exc")
    asyncio.run(gc2._attempt(provider2, "glm", "conv-exc"))
    diag2 = gc2.get_diagnostics()
    assert diag2["pending_count"] == 1
    assert "RuntimeError" in diag2["pending"][0]["last_error"]
    shutil.rmtree(tmp2, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_failure_keeps_entry_with_attempts: PASSED")


def test_dropped_after_max_attempts():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    os.environ["GATEWAY_CONVERSATION_GC_MAX_ATTEMPTS"] = "3"
    try:
        gc = _fresh(tmp)
        assert gc.max_attempts == 3
        provider = FakeProvider(result=False)
        gc.enqueue("doubao", "conv-drop")
        for _ in range(3):
            asyncio.run(gc._attempt(provider, "doubao", "conv-drop"))
        diag = gc.get_diagnostics()
        assert diag["pending_count"] == 0, "达到上限后必须放弃，杜绝无界重试"
        assert diag["stats"]["dropped"] == 1
        assert diag["stats"]["failed"] == 3
    finally:
        os.environ.pop("GATEWAY_CONVERSATION_GC_MAX_ATTEMPTS", None)
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_dropped_after_max_attempts: PASSED")


# ------------------------------------------------------------------ 5. 跨重启恢复

def test_cross_restart_recovery():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    gc.enqueue("qwen", "survivor-1")
    gc.enqueue("deepseek", "survivor-2")

    # 模拟进程重启：全新实例从同一目录加载
    revived = ConversationGC(directory=tmp)
    diag = revived.get_diagnostics()
    assert diag["pending_count"] == 2, "重启后必须恢复待删除队列"
    ids = sorted(e["conversation_id"] for e in diag["pending"])
    assert ids == ["survivor-1", "survivor-2"]
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_cross_restart_recovery: PASSED")


def test_corrupt_queue_file_is_ignored():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    (tmp / "conversation_gc.json").write_text("{not-json", encoding="utf-8")
    gc = ConversationGC(directory=tmp)  # 不得抛异常
    assert gc.get_diagnostics()["pending_count"] == 0
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_corrupt_queue_file_is_ignored: PASSED")


# ------------------------------------------------------------------ 6. drain_once

def test_drain_once_uses_resolver():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    calls = {"n": 0}

    def resolver(provider_key):
        calls["n"] += 1
        assert provider_key in ("qwen", "deepseek")
        return FakeProvider(result=True)

    gc.set_resolver(resolver)
    gc.enqueue("qwen", "drain-1")
    gc.enqueue("deepseek", "drain-2")
    result = asyncio.run(gc.drain_once())

    assert result == {"attempted": 2, "succeeded": 2}
    assert calls["n"] == 2, "必须通过 resolver 重建 provider 实例"
    assert gc.get_diagnostics()["pending_count"] == 0
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_drain_once_uses_resolver: PASSED")


def test_drain_once_without_resolver_is_noop():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    gc.enqueue("qwen", "no-resolver")
    assert asyncio.run(gc.drain_once()) == {"attempted": 0, "succeeded": 0}
    assert gc.get_diagnostics()["pending_count"] == 1, "无 resolver 时记录必须保留"
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_drain_once_without_resolver_is_noop: PASSED")


# ------------------------------------------------------------------ 7/8. 降级与无循环

def test_disabled_gate_falls_back_to_fire_and_forget():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    os.environ["GATEWAY_DISABLE_CONVERSATION_GC"] = "1"
    try:
        gc = _fresh(tmp)
        assert gc.is_disabled() is True
        provider = FakeProvider(result=True)

        async def _run():
            gc.schedule(provider, "qwen", "legacy-1")
            await asyncio.sleep(0.05)

        asyncio.run(_run())
        assert provider.deleted == ["legacy-1"], "关闭 GC 时必须保持历史 fire-and-forget 行为"
        assert not gc.queue_path.exists(), "关闭 GC 时不得落盘"
        assert gc.get_diagnostics()["enabled"] is False
    finally:
        os.environ.pop("GATEWAY_DISABLE_CONVERSATION_GC", None)
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_disabled_gate_falls_back_to_fire_and_forget: PASSED")


def test_schedule_without_event_loop_keeps_entry():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    provider = FakeProvider(result=True)
    # 同步上下文（无运行中事件循环）调用不得抛异常
    gc.schedule(provider, "kimi", "no-loop")
    assert gc.get_diagnostics()["pending_count"] == 1, "无事件循环时必须保留队列记录"
    assert provider.deleted == [], "无事件循环时不得发起删除"
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_schedule_without_event_loop_keeps_entry: PASSED")


def test_schedule_success_clears_entry():
    tmp = Path(tempfile.mkdtemp(prefix="cgc_test_"))
    gc = _fresh(tmp)
    provider = FakeProvider(result=True)

    async def _run():
        gc.schedule(provider, "qwen", "sched-ok")
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    diag = gc.get_diagnostics()
    assert provider.deleted == ["sched-ok"]
    assert diag["pending_count"] == 0
    assert diag["stats"]["scheduled"] == 1
    assert diag["stats"]["deleted"] == 1
    shutil.rmtree(tmp, ignore_errors=True)
    print("test_schedule_success_clears_entry: PASSED")


# ------------------------------------------------------------------ 9. server 接线

def test_server_wiring():
    srv = importlib.import_module("server")

    # resolver 已注入
    assert srv.conversation_gc._resolver is not None, "server 必须注入 provider 构造器"

    # 12 → 9 处调用点全部改为入队调度，且使用 dispatched_provider 作为节点标识
    src = Path(srv.__file__).read_text(encoding="utf-8")
    assert "asyncio.create_task(active_provider.delete_conversation(" not in src, (
        "不得残留历史 fire-and-forget 调用点"
    )
    assert src.count("conversation_gc.schedule(active_provider, dispatched_provider,") == 9, (
        "全部 9 处会话删除调用点必须改为持久化队列"
    )

    from fastapi.testclient import TestClient

    client = TestClient(srv.app)
    diag = client.get("/v1/diagnostics").json()
    assert "conversation_gc" in diag
    cg_diag = diag["conversation_gc"]
    assert cg_diag["enabled"] is True
    assert cg_diag["max_attempts"] >= 1
    assert "pending_count" in cg_diag

    # 端到端：入队 + drain_once 走到 server 的 _make_provider（无真实令牌 → 构造失败，记录保留）
    srv.conversation_gc.clear()
    srv.conversation_gc.enqueue("qwen", "e2e-1")
    asyncio.run(srv.conversation_gc.drain_once())
    assert srv.conversation_gc.get_diagnostics()["pending_count"] == 1, (
        "构造失败时不得丢失待删除记录"
    )
    srv.conversation_gc.clear()
    print("test_server_wiring: PASSED")


def run_all():
    test_enqueue_persists_atomically()
    test_enqueue_is_idempotent()
    test_success_removes_from_queue()
    test_failure_keeps_entry_with_attempts()
    test_dropped_after_max_attempts()
    test_cross_restart_recovery()
    test_corrupt_queue_file_is_ignored()
    test_drain_once_uses_resolver()
    test_drain_once_without_resolver_is_noop()
    test_disabled_gate_falls_back_to_fire_and_forget()
    test_schedule_without_event_loop_keeps_entry()
    test_schedule_success_clears_entry()
    test_server_wiring()
    print("\n✅ All conversation GC tests (OPT-016) passed!")


if __name__ == "__main__":
    run_all()
