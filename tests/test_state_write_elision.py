"""冷却状态的落盘语义与静默失效测试 (tests/test_state_write_elision.py)。

[H5] `_persist_state()` 位于两个**每请求**路径上：`record_success`（上游首包成功
之后）与 `record_failure`。原实现无条件写盘——临时文件 + `fsync` + `os.replace`，
并且是在 `asyncio.Lock` 内同步执行。实测在一个健康节点上，20 次成功后发生了
**20 次**磁盘写入，而内容逐字节相同。fsync 在本机中位数约 0.5ms、p95 约 1.2ms，
落在事件循环里就是每请求的固定停顿，也是多连接并发时的排队来源。

本套件锁住三件事，缺一不可：

1. **状态没变就不写**：健康节点的成功不得产生磁盘写入；
2. **状态变了必须写**：冷却产生/升级、失败计数变化、冷却被成功清除，都必须落盘
   ——「少写」绝不能退化成「漏写」，否则跨重启避让（OPT-011）就失去依据；
3. **失败不得静默**：写盘异常必须留下日志。曾经这里是一条裸 `except: pass`，一次
   `NameError`（`json` 未导入）让整条持久化链路失效而日志中毫无痕迹。

另外锁住 `save_state` 的返回值语义：只有**确实写盘成功**才返回 True，否则调用方
会把「写失败」的内容记成「已落盘」，一次瞬时磁盘故障就让状态直到下次变化前都不再
重试。

运行: .venv/bin/python tests/test_state_write_elision.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ.pop("GATEWAY_DISABLE_RISK_LEDGER", None)
os.environ.pop("GATEWAY_DISABLE_COOLDOWN_GATE", None)

from providers.failover import CooldownTracker  # noqa: E402


class _WriteCounter:
    """统计对冷却文件的真实写盘次数（在类级别捕获，避免单例被替换后失效）。"""

    def __init__(self):
        self.count = 0
        self._orig = None

    def __enter__(self):
        self._orig = type(CooldownTracker().risk_ledger)._atomic_write

        def spy(ledger_self, path, content):
            self.count += 1
            return self._orig(ledger_self, path, content)

        type(CooldownTracker().risk_ledger)._atomic_write = spy
        return self

    def __exit__(self, *exc):
        type(CooldownTracker().risk_ledger)._atomic_write = self._orig
        return False


def _run(coro):
    return asyncio.run(coro)


def test_healthy_success_does_not_write_state():
    tracker = CooldownTracker()
    with _WriteCounter() as w:
        for _ in range(10):
            _run(tracker.record_success("qwen"))
    assert w.count == 0, (
        f"健康节点的 10 次成功产生了 {w.count} 次磁盘写入：每请求一次 fsync 落在"
        f"事件循环里，而状态内容没有任何变化"
    )
    print("test_healthy_success_does_not_write_state: PASSED")


def test_state_change_is_always_persisted():
    tracker = CooldownTracker()
    with _WriteCounter() as w:
        _run(tracker.record_failure("qwen", 429, "rate limit 频控"))
        assert w.count == 1, f"冷却产生没有落盘 (writes={w.count})"
        assert tracker.get_remaining_cooldown("qwen") > 0

        # 失败计数变化也必须落盘（它决定 auth/waf 的升级次数，跨重启有意义）
        _run(tracker.record_failure("qwen", 429, "rate limit 频控"))
        assert w.count == 2, f"失败计数从 1 变 2 没有落盘 (writes={w.count})"

        # 另一个节点进入冷却：整份状态变化，必须落盘
        _run(tracker.record_failure("glm", 401, "令牌失效"))
        assert w.count == 3, f"第二个节点进入冷却没有落盘 (writes={w.count})"

        # 冷却被成功清除：状态变化，必须落盘
        _run(tracker.record_success("qwen"))
        assert w.count == 4, f"冷却清除没有落盘 (writes={w.count})"
        assert tracker.get_remaining_cooldown("qwen") == 0

        # 清除之后再次成功：内容不变，不写
        _run(tracker.record_success("qwen"))
        assert w.count == 4, (
            f"状态未变的成功仍然写盘了 (writes={w.count})——省略必须只在「无变化」时生效"
        )
    print("test_state_change_is_always_persisted: PASSED")


def test_persisted_state_is_what_disk_holds():
    """省略写入不得让磁盘落后于内存。"""
    import json

    tracker = CooldownTracker()
    _run(tracker.record_failure("kimi", 429, "频控"))
    _run(tracker.record_success("kimi"))
    on_disk = json.loads(tracker.risk_ledger.cooldown_path.read_text())
    # 比「某个字段等于某值」更强：磁盘内容必须与内存快照**整体一致**。
    # 省略写入的唯一合法性来源就是「二者本来相同」，所以这里直接测这一点。
    assert on_disk == tracker._state_snapshot(), (
        f"磁盘状态落后于内存：磁盘={on_disk} 内存={tracker._state_snapshot()}"
    )
    assert "kimi" not in on_disk.get("cooldowns", {}), (
        f"已经解除的冷却仍留在磁盘上，重启后会复活: {on_disk.get('cooldowns')}"
    )
    assert "kimi" not in on_disk.get("failure_counts", {}), (
        f"已恢复健康的节点仍以 0 计数留在磁盘上: {on_disk.get('failure_counts')}"
    )
    print("test_persisted_state_is_what_disk_holds: PASSED")


def test_persist_failure_is_logged_not_swallowed():
    """写盘异常必须留下日志：裸 `except: pass` 曾让整条链路静默失效。"""
    tracker = CooldownTracker()
    original = tracker.risk_ledger.save_state

    def boom(state):
        raise NameError("name 'json' is not defined")

    tracker.risk_ledger.save_state = boom
    logger = logging.getLogger("gateway.failover")
    records: list[logging.LogRecord] = []

    class Sink(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Sink(level=logging.WARNING)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        _run(tracker.record_failure("doubao", 500, "上游 5xx"))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        tracker.risk_ledger.save_state = original

    messages = [r.getMessage() for r in records if "风控持久化" in r.getMessage()]
    assert messages, (
        f"持久化失败被完全吞掉了，日志里没有任何痕迹（这正是 json 未导入时发生的事）: "
        f"{[r.getMessage() for r in records]}"
    )
    assert "NameError" in messages[0]
    print("test_persist_failure_is_logged_not_swallowed: PASSED")


def test_save_state_reports_actual_disk_outcome():
    tracker = CooldownTracker()
    ledger = tracker.risk_ledger
    assert ledger.save_state({"cooldowns": {}}) is True

    original = type(ledger)._atomic_write
    type(ledger)._atomic_write = lambda self, path, content: False
    try:
        assert ledger.save_state({"cooldowns": {}}) is False, (
            "写盘失败时 save_state 仍报成功：调用方会把这次内容记成「已落盘」，"
            "一次瞬时故障就让状态在内容再次变化前都不再重试"
        )
    finally:
        type(ledger)._atomic_write = original
    print("test_save_state_reports_actual_disk_outcome: PASSED")


def run_all():
    test_healthy_success_does_not_write_state()
    test_state_change_is_always_persisted()
    test_persisted_state_is_what_disk_holds()
    test_persist_failure_is_logged_not_swallowed()
    test_save_state_reports_actual_disk_outcome()
    print("\n✅ All state write-elision tests passed!")


if __name__ == "__main__":
    run_all()