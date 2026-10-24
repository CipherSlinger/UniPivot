"""健康探针创建的远端临时会话必须被回收。

[PROBE-GC] `HealthMonitor.probe_provider` 以 `conversation_id=None` 调用
`provider.chat(...)`，五个上游因此各建一个临时会话，并在用户的网页侧栏里留下
一条真实的「hi」对话。探针**丢弃**了返回的 `conversation_id`，也从不调用
`delete_conversation`——`ConversationGC` 的三个调用点全在 `server.py` 的请求路径上，
`healing/monitor.py` 一处都没有。

规模：`HEAL_CHECK_INTERVAL` 默认 300 秒 × 5 个节点 = **每天 1440 个**探针会话，
每个都是带 `hi` 的完整对话。这有两重伤害：
1. 用户侧栏被探针垃圾填满，真实的匿名调用会话反而找不到了；
2. 每 60 秒一个「hi」的固定节奏，正是风控用来判定自动化客户端的时序特征。

探针的流本身是被正确终结的（实测 `finally` 会跑），所以这不是连接泄漏；
泄漏的是**上游侧的账面资源**，而它恰恰是风控看得见的那一面。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"


def test_probe_schedules_its_conversation_for_deletion():
    """探针从流元数据里拿到的 `conversation_id` 必须交给 ConversationGC。

    实测（修复前）：探针返回健康，但该会话 id 既没入队也没被删除——
    `conversation_gc` 的 `scheduled` 计数保持不变，上游侧栏永久残留。
    """
    from healing.monitor import HealthMonitor
    from providers.conversation_gc import conversation_gc

    scheduled_before = conversation_gc.get_diagnostics()["stats"]["scheduled"]
    probe_conv = "probe_conv_1"

    class FakeProvider:
        def chat(self, **kw):
            async def gen():
                yield "h", {}
                # 真实适配器在尾部元数据帧里回传 conversation_id
                yield "", {"conversation_id": probe_conv}

            return gen()

        async def delete_conversation(self, conversation_id):
            deleted.append(conversation_id)
            return True

    deleted: list[str] = []

    async def main():
        monitor = HealthMonitor()
        monitor.register_provider(
            "qwen", lambda: type("S", (), {"token": "t"})(), FakeProvider
        )
        ok, err = await monitor.probe_provider("qwen")
        assert ok, err
        await asyncio.sleep(0.15)

    asyncio.run(main())

    diag = conversation_gc.get_diagnostics()["stats"]
    assert diag["scheduled"] > scheduled_before or probe_conv in deleted, (
        f"探针创建的会话 {probe_conv!r} 既没有入 GC 队列也没有被删除——"
        f"默认 300s × 5 节点 = 每天 1440 个「hi」对话永久残留在用户侧栏"
    )
    print("test_probe_schedules_its_conversation_for_deletion: PASSED")


def test_probe_does_not_leave_the_conversation_when_gc_disabled():
    """GC 总开关关闭时，探针仍不能把会话丢在地上。"""
    from healing.monitor import HealthMonitor

    deleted: list[str] = []

    class FakeProvider:
        def chat(self, **kw):
            async def gen():
                yield "h", {}
                yield "", {"conversation_id": "probe_conv_2"}

            return gen()

        async def delete_conversation(self, conversation_id):
            deleted.append(conversation_id)
            return True

    async def main():
        monitor = HealthMonitor()
        monitor.register_provider(
            "deepseek", lambda: type("S", (), {"token": "t"})(), FakeProvider
        )
        await monitor.probe_provider("deepseek")
        await asyncio.sleep(0.15)

    os.environ["GATEWAY_DISABLE_CONVERSATION_GC"] = "1"
    try:
        asyncio.run(main())
    finally:
        os.environ.pop("GATEWAY_DISABLE_CONVERSATION_GC", None)

    assert "probe_conv_2" in deleted, (
        f"GC 关闭时探针会话没有被直接删除（deleted={deleted}）——"
        f"总开关的含义是「不做持久化重试」，不是「留下垃圾」"
    )
    print("test_probe_does_not_leave_the_conversation_when_gc_disabled: PASSED")


def test_a_slow_but_connected_upstream_is_not_reported_unhealthy():
    """握手成功但回答很长 => 健康；完全无响应 => 不健康。

    探针改为消费整条流之后，15s 超时覆盖的不再只是握手而是整段回答。五个上游
    都**不支持** `max_tokens`，提问「hi」可能换来一段长回答——若把它判为不健康，
    网关会去冷却一个完全正常的节点。实测（修复前语义）：同样的长回答场景判
    `ok=False`，与「上游真的挂了」无法区分。
    """
    import healing.monitor as hm

    class Endless:
        def chat(self, **kw):
            async def gen():
                yield "h", {"conversation_id": "c-long"}
                for _ in range(500):
                    await asyncio.sleep(0.2)
                    yield "x", {}

            return gen()

        async def delete_conversation(self, cid):
            return True

    class Dead:
        def chat(self, **kw):
            async def gen():
                await asyncio.sleep(30)
                yield "h", {}

            return gen()

        async def delete_conversation(self, cid):
            return True

    async def run(cls):
        monitor = hm.HealthMonitor()
        monitor.register_provider(
            "qwen", lambda: type("S", (), {"token": "t"})(), cls
        )
        orig = hm.asyncio.wait_for

        async def fast(coro, timeout=None):
            return await orig(coro, 1.5)

        hm.asyncio.wait_for = fast
        try:
            return await monitor.probe_provider("qwen")
        finally:
            hm.asyncio.wait_for = orig

    ok_long, err_long = asyncio.run(run(Endless))
    assert ok_long, (
        f"握手成功后只是回答长，却被判不健康（{err_long!r}）——"
        f"五个上游都不支持 max_tokens，这会把正常节点拖进冷却"
    )

    ok_dead, err_dead = asyncio.run(run(Dead))
    assert not ok_dead, "完全无响应的上游必须判不健康"
    assert "超时" in (err_dead or ""), err_dead
    print("test_a_slow_but_connected_upstream_is_not_reported_unhealthy: PASSED")


def test_a_protocol_without_a_completion_marker_stays_healthy():
    """无完成标记的协议缺键 = 未知，绝不能判截断。"""
    import healing.monitor as hm

    class NoMarker:
        def chat(self, **kw):
            async def gen():
                yield "h", {}
                yield "", {"conversation_id": "c3"}  # 没有 upstream_complete 键

            return gen()

        async def delete_conversation(self, cid):
            return True

    async def main():
        monitor = hm.HealthMonitor()
        monitor.register_provider(
            "deepseek", lambda: type("S", (), {"token": "t"})(), NoMarker
        )
        return await monitor.probe_provider("deepseek")

    ok, err = asyncio.run(main())
    assert ok, f"无标记协议被误判为截断（{err!r}）——三家上游会全线变红"
    print("test_a_protocol_without_a_completion_marker_stays_healthy: PASSED")


def run_all():
    test_probe_schedules_its_conversation_for_deletion()
    test_probe_does_not_leave_the_conversation_when_gc_disabled()
    test_a_slow_but_connected_upstream_is_not_reported_unhealthy()
    test_a_protocol_without_a_completion_marker_stays_healthy()
    print("\n✅ All probe conversation reclamation tests passed!")


if __name__ == "__main__":
    run_all()