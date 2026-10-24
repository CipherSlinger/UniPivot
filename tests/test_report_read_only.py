"""以 GET 方式读取报告，却把仓库写了一遍 (PROTOCOL / ISOLATION)

`GET /v1/reports/latest` 是一个**读**接口——前端打开「优化建议书」弹窗时调它，
`/v1/tasks/evaluation` 是它的兜底（`static/index.html`）。两者都不是动作。

实测到的三个事实（本机，2026-10-24）：

1. 这个 GET 每次都重写 `reports/optimization_recommendations.json`、
   `optimization_proposal.json`、`optimization_recommendations.md` 三个**被 git 跟踪**的
   文件。内容逐字节相同，但 `generated_at` 换成了当前时间，于是 `git status` 永远脏，
   `reports/` 的每次 diff 都是无意义的噪声。跑一遍 `tests/test_server.py` 或
   `tests/test_dispatcher.py` 就能复现：mtime 变化 + 三个 ` M `。
2. 生成在事件循环上同步执行：实测 **4.11 ms**（中位，n=20）的字典拼装，
   外加三次同步文件写入——对一个纯读接口来说，这是每个读请求都在付的固定停顿。
3. 报告本身**永远不包含任何运行时数据**：`evaluate_optimization_opportunities()`
   的 `tasks` 参数默认 `[]`，`server.py` 两个调用点都不传，所以内容只取决于代码里的常量。
   也就是说第 1 条里每次重写的那份字节，和上一次完全相同（除时间戳）。

还有一处结构缺陷把这三点锁在一起：`/v1/reports/latest` 被**注册了两次**。
先注册的 `get_tasks_evaluation` 抢走了这个路径，后注册的
`get_latest_optimization_report`（它本来就是「先读文件、没有才生成」的只读契约）
永远不会被派发到——死代码。所以「只读契约」存在，只是够不着。

本套件钉住的是**可观测的行为**，不是实现：
- 读取路径不得改动 git 工作区（这是本次修复的核心断言，也是回归最容易重新引入的一条）；
- 读取路径不得在事件循环上做那 4.11 ms 的同步工作；
- 已经落地的快照必须原样返回（`/v1/tasks/evaluation` 的消费者断言 `markdown_report`）；
- `TESTING=1` 下不得写到真实的 `reports/`（OPT-019 的同类缺口——那次的成因是
  `TaskEvaluator` 接收 `project_root` 却拿模块级 `REPORTS_DIR` 落盘，构造参数是装饰品）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# [OPT-019] 隔离运行：真实 session/ 与 reports/ 目录不得被测试写入。
os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ["GATEWAY_DISABLE_CREDENTIAL_WATCH"] = "1"
os.environ["GATEWAY_DISABLE_RISK_LEDGER"] = "1"
os.environ["GATEWAY_DISABLE_PROMPT_FOLDING"] = "1"
os.environ["QWEN_AUTH_TOKEN"] = "fake_qwen"
os.environ["DEEPSEEK_AUTH_TOKEN"] = "fake_ds"
os.environ["DOUBAO_AUTH_TOKEN"] = "fake_doubao"

from fastapi.testclient import TestClient  # noqa: E402

import server as srv  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent / "reports"
TRACKED = (
    REPORTS / "optimization_recommendations.json",
    REPORTS / "optimization_proposal.json",
    REPORTS / "optimization_recommendations.md",
)


def _fingerprint() -> dict:
    """mtime + 内容哈希。只看内容哈希会漏掉「重写同样字节」这一种（实测正是这种）。"""
    out = {}
    for p in TRACKED:
        out[p.name] = (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def _listing() -> dict:
    return {p.name: p.read_bytes() for p in sorted(REPORTS.glob("*")) if p.is_file()}


# ------------------------------------------------ 1. 读取不得改动被测仓库（核心）

def test_reading_the_report_does_not_rewrite_the_repository():
    """`GET /v1/reports/latest` 之后，`reports/` 的 mtime 与字节都必须原封不动。

    修复前：mtime 每次变化、三个文件全部被重写（实测 `git status` 出现三个 ` M `）。
    """
    client = TestClient(srv.app)
    before = _fingerprint()
    r = client.get("/v1/reports/latest")
    assert r.status_code == 200, f"读取接口返回 {r.status_code}: {r.text[:200]}"
    after = _fingerprint()

    changed = [n for n in before if before[n][0] != after[n][0]]
    assert not changed, (
        f"GET /v1/reports/latest 重写了仓库文件: mtime 变化 {changed} —— "
        "读接口不得产生写副作用"
    )
    differs = [n for n in before if before[n][1] != after[n][1]]
    assert not differs, f"GET /v1/reports/latest 改变了文件内容: {differs}"

    print("test_reading_the_report_does_not_rewrite_the_repository: PASSED")


def test_stored_snapshot_is_served_verbatim():
    """已经有快照时，返回的必须是**文件里的那一份**，而不是现算一份新的。

    这条把「只读」钉在数据上：一个「先读后写」的实现能通过上一条（写完字节相同），
    但这一条要求 `generated_at` 与磁盘上完全一致。
    """
    snapshot = TRACKED[0]
    on_disk = json.loads(snapshot.read_text(encoding="utf-8"))
    d = TestClient(srv.app).get("/v1/reports/latest").json()
    assert d.get("generated_at") == on_disk.get("generated_at"), (
        f"返回的报告不是磁盘快照：generated_at {d.get('generated_at')!r} "
        f"!= {on_disk.get('generated_at')!r}"
    )
    assert d.get("title") == on_disk.get("title")
    print("test_stored_snapshot_is_served_verbatim: PASSED")


def test_absent_snapshot_is_generated_once_not_persisted():
    """快照不存在时（全新检出）现算一次，且**不**落盘。

    全新 checkout 里 `reports/` 为空时接口必须仍然可用——这正是
    `get_latest_optimization_report` 原本的契约，此前因为路由被抢占而不可达。
    """
    repo_before = _fingerprint()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_reports = Path(tmp) / "reports"
        tmp_reports.mkdir()
        original = srv._REPORTS_DIR
        srv._REPORTS_DIR = tmp_reports
        try:
            d = TestClient(srv.app).get("/v1/reports/latest").json()
            assert d["title"] == "项目优化空间与迭代需求建议书"
            assert "optimization_opportunities" in d
            leftovers = list(tmp_reports.glob("*"))
            assert not leftovers, f"读取接口在空目录里落盘了: {leftovers}"
        finally:
            srv._REPORTS_DIR = original
    # 也必须**没有**偷偷写到别处：空快照目录只证明「没写到这里」，
    # 修复前正是写到真实仓库去了，那样这条用例会以「通过了」的方式说谎。
    repo_after = _fingerprint()
    leaked = [n for n in repo_before if repo_before[n] != repo_after[n]]
    assert not leaked, f"快照缺失时读取接口改写到了别处（真实仓库）: {leaked}"
    print("test_absent_snapshot_is_generated_once_not_persisted: PASSED")


# ------------------------------------------- 2. 不得在事件循环上做同步重活

def test_report_generation_runs_off_the_event_loop():
    """4.11 ms 的字典拼装不得挡住事件循环。

    这是**每个**读请求都在付的固定停顿。判据用「并发心跳的延迟」而不是源码形状：
    服务端同步阻塞事件循环时，同一循环上的心跳任务会被推迟到整段同步工作结束之后。
    """
    import asyncio

    from healing.task_dispatcher import TaskEvaluator

    async def _probe():
        evaluator = TaskEvaluator()

        async def _heartbeat():
            # 一个苛求的心跳：每 1 ms 记一次，看最长空档
            gaps, last = [], time.perf_counter()
            for _ in range(400):
                await asyncio.sleep(0.001)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now
            return max(gaps)

        beat = asyncio.ensure_future(_heartbeat())
        await asyncio.sleep(0)  # 让心跳先跑起来
        t0 = time.perf_counter()
        for _ in range(10):
            await srv._generate_optimization_report(evaluator)
        elapsed = time.perf_counter() - t0
        max_gap = await beat
        return elapsed, max_gap

    elapsed, max_gap = asyncio.run(_probe())
    # 10 次生成约 41 ms 的同步工作。若全在事件循环上，单次空档必然 >> 40 ms；
    # 放到线程里之后，单次空档只剩调度抖动（实测 < 15 ms）。
    assert max_gap < 0.040, (
        f"事件循环被报告生成阻塞了 {max_gap * 1000:.0f} ms（10 次生成共 {elapsed * 1000:.0f} ms）"
        "——同步 CPU 工作必须移出事件循环"
    )
    print(f"test_report_generation_runs_off_the_event_loop: PASSED "
          f"(max event-loop gap {max_gap * 1000:.1f} ms)")


# --------------------------------------- 3. 路由唯一性：同一路径不得注册两次

def test_no_path_is_registered_twice():
    """同一个 (路径, 方法) 注册两次时，先注册的赢——后一个是死代码，且没有任何报错。

    `get_latest_optimization_report`（先读快照、没有才生成，本身就是只读契约）
    此前就是这样被 `get_tasks_evaluation` 完全遮住的。这条断言覆盖**全部**路由，
    不只看这两个：重复注册不会报错、不会告警，只会在某天悄悄换掉一个处理函数，
    而唯一的征兆是「我明明改了它却不生效」。
    """
    from collections import defaultdict

    seen = defaultdict(list)
    for route in srv.app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        methods = tuple(sorted(getattr(route, "methods", None) or []))
        seen[(path, methods)].append(route.endpoint.__name__)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, (
        "同一 (路径, 方法) 注册了多个处理函数（后者永远不会被派发）: "
        + ", ".join(f"{k[0]} {k[1]} -> {v}" for k, v in sorted(dupes.items()))
    )
    print(f"test_no_path_is_registered_twice: PASSED ({len(seen)} unique routes)")


# ------------------------------------------- 4. 存储位置随注入的根目录走

def test_evaluator_writes_under_its_own_project_root():
    """`TaskEvaluator(project_root=X)` 的落盘位置必须跟着 X 走。

    修复前 `project_root` 是装饰品：落盘用的是模块级 `REPORTS_DIR`，
    诚实性完全依赖调用方恰好在 `patch()` 里跑。
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        snapshot = _listing()
        TaskEvaluator_ = _evaluator_cls()
        TaskEvaluator_(project_root=root).evaluate_optimization_opportunities(tasks=[])
        written = sorted(p.name for p in (root / "reports").glob("*"))
        assert written == [
            "optimization_proposal.json",
            "optimization_recommendations.json",
            "optimization_recommendations.md",
        ], f"落盘到了 {written}，不在注入的 project_root 下"
        # 且真实仓库没有被顺带写一遍
        now = _listing()
        changed = [n for n in snapshot if snapshot[n] != now.get(n)]
        assert not changed, f"注入 project_root 后仍然改动了真实仓库: {changed}"
    print("test_evaluator_writes_under_its_own_project_root: PASSED")


# ------------------------------- 5. 更一般的形式：GET 处理函数不得写盘

def test_no_get_route_writes_to_disk_at_runtime():
    """运行时地面真相：**任何** GET 路由都不得写盘——无论写在多少层之下。

    这条用例的前身是静态可达性分析（从 GET 处理函数顺着调用图往下找写操作），
    它有两个致命的近似误差，合起来正好把它变成一条**会说谎**的断言：

    1. **假阴性**：第一版只扫处理函数体，而修复前的写动作藏在两层之下
       （`get_tasks_evaluation` -> `TaskDispatcher.get_latest_evaluation_report`
       -> `TaskEvaluator.evaluate_optimization_opportunities`），于是修复前的代码
       直接通过——为「防止读接口写盘」而写的用例，对那个真实的写盘无感。
    2. **假阳性**：改成按调用图解析后，它按**名字**匹配，`dict.get()` 被当成了
       `DeviceIdentityStore.get`（一个确实会落盘的 mint-once 方法），于是报告
       `/v1/admin/dispatch-audit` 在写盘——实测该接口只读，是被冤枉的。

    静态分析在这里本质上是在猜「代码会做什么」，而这件事可以直接观测：把写
    原语（`open` 的写模式、`Path.write_text/bytes`、`json.dump`、`os.replace`、
    `mkdir`）全部挂上跳闸器，然后**真的**把每个无参 GET 路由请求一遍。写一次
    即失败，并打印是哪条路由、哪个调用点写的。

    这是本缺陷的一般形式：写动作可以藏在任意深度，而运行时观测不需要知道它在哪。
    """
    import builtins
    import io
    import json as _json

    ROOT = Path(__file__).resolve().parent.parent
    _orig_io_open = io.open
    _orig_builtins_open = builtins.open
    _orig_dump = _json.dump
    _orig_replace = os.replace
    _orig_mkdir = Path.mkdir
    _orig_write_text = Path.write_text
    _orig_write_bytes = Path.write_bytes

    writes: list = []
    active = {"on": False}

    def _note(what: str, target) -> None:
        if not active["on"]:
            return
        import traceback

        frames = [
            f"{Path(f.filename).name}:{f.lineno}:{f.name}"
            for f in traceback.extract_stack()[:-2]
            if "site-packages" not in f.filename and "/tests/" not in f.filename
        ]
        writes.append(f"{what} -> {target} @ {' <- '.join(frames[-3:][::-1])}")

    def _guard_open(file, mode="r", *a, **kw):
        if any(c in str(mode) for c in "wxa+"):
            _note(f"open(mode={mode!r})", file)
        return _orig_io_open(file, mode, *a, **kw)

    def _guard_builtins_open(file, mode="r", *a, **kw):
        if any(c in str(mode) for c in "wxa+"):
            _note(f"builtins.open(mode={mode!r})", file)
        return _orig_builtins_open(file, mode, *a, **kw)

    def _guard_dump(obj, fp, *a, **kw):
        _note("json.dump", getattr(fp, "name", fp))
        return _orig_dump(obj, fp, *a, **kw)

    def _guard_replace(src, dst, *a, **kw):
        _note("os.replace", dst)
        return _orig_replace(src, dst, *a, **kw)

    def _guard_mkdir(self, *a, **kw):
        _note("mkdir", self)
        return _orig_mkdir(self, *a, **kw)

    def _guard_write_text(self, *a, **kw):
        _note("Path.write_text", self)
        return _orig_write_text(self, *a, **kw)

    def _guard_write_bytes(self, *a, **kw):
        _note("Path.write_bytes", self)
        return _orig_write_bytes(self, *a, **kw)

    patched = {
        (io, "open"): _guard_open,
        (builtins, "open"): _guard_builtins_open,
        (_json, "dump"): _guard_dump,
        (os, "replace"): _guard_replace,
        (Path, "mkdir"): _guard_mkdir,
        (Path, "write_text"): _guard_write_text,
        (Path, "write_bytes"): _guard_write_bytes,
    }
    for (mod, attr), fn in patched.items():
        setattr(mod, attr, fn)

    client = TestClient(srv.app)
    probed, skipped_param = [], []
    try:
        active["on"] = True
        for route in srv.app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None) or set()
            if path is None or "GET" not in methods:
                continue
            if "{" in path:  # 需要路径参数的路由另算，避免把 404 当成读路径
                skipped_param.append(path)
                continue
            client.get(path, follow_redirects=False)
            probed.append(path)
    finally:
        active["on"] = False
        io.open = _orig_io_open
        builtins.open = _orig_builtins_open
        _json.dump = _orig_dump
        os.replace = _orig_replace
        Path.mkdir = _orig_mkdir
        Path.write_text = _orig_write_text
        Path.write_bytes = _orig_write_bytes

    assert not writes, (
        f"GET 路由在写盘（读接口不得产生写副作用）: "
        + "; ".join(dict.fromkeys(writes))
    )
    assert "/v1/reports/latest" in probed, f"未探测到核心读接口，用例失效: {probed}"
    print(f"test_no_get_route_writes_to_disk_at_runtime: PASSED "
          f"({len(probed)} GET routes probed, {len(skipped_param)} need path params)")


def _evaluator_cls():
    from healing.task_dispatcher import TaskEvaluator
    return TaskEvaluator


def run_all():
    tests = [
        test_reading_the_report_does_not_rewrite_the_repository,
        test_stored_snapshot_is_served_verbatim,
        test_absent_snapshot_is_generated_once_not_persisted,
        test_report_generation_runs_off_the_event_loop,
        test_no_path_is_registered_twice,
        test_no_get_route_writes_to_disk_at_runtime,
        test_evaluator_writes_under_its_own_project_root,
    ]
    for fn in tests:
        fn()
    print(f"\n✅ All report-read-only tests passed! ({len(tests)} cases)")


if __name__ == "__main__":
    run_all()