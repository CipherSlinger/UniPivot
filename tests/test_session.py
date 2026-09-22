"""session_store 的存取与解析优先级测试。

运行: .venv/bin/python tests/test_session.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import session_store as ss


def test_session_roundtrip(tmp_path: Path):
    p = tmp_path / "s.json"
    s = ss.Session(token="tok123", cookie="a=1; b=2", user_agent="ua", captured_at=time.time())
    s.save(p)
    loaded = ss.Session.load(p)
    assert loaded is not None
    assert loaded.token == "tok123"
    assert loaded.cookie == "a=1; b=2"
    assert loaded.user_agent == "ua"
    assert abs(loaded.age) < 10
    print("[PASS] Session 存取往返")


def test_load_missing_and_corrupt(tmp_path: Path):
    assert ss.Session.load(tmp_path / "none.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json{", encoding="utf-8")
    assert ss.Session.load(bad) is None
    print("[PASS] 缺失/损坏文件安全返回 None")


def test_env_priority(tmp_path: Path):
    """环境变量 > session 文件 > None。"""
    saved_qwen, saved_ds, saved_db = ss.QWEN_SESSION_FILE, ss.DEEPSEEK_SESSION_FILE, ss.DOUBAO_SESSION_FILE
    saved_env = {k: os.environ.get(k) for k in ("QWEN_AUTH_TOKEN", "QWEN_COOKIE", "DEEPSEEK_AUTH_TOKEN", "DOUBAO_AUTH_TOKEN", "DOUBAO_COOKIE")}
    try:
        ss.QWEN_SESSION_FILE = tmp_path / "qwen.json"
        ss.DEEPSEEK_SESSION_FILE = tmp_path / "deepseek.json"
        ss.DOUBAO_SESSION_FILE = tmp_path / "doubao.json"

        # 环境变量优先
        os.environ["QWEN_AUTH_TOKEN"] = "env_tok"
        os.environ["QWEN_COOKIE"] = "env_cookie=1"
        os.environ.pop("DEEPSEEK_AUTH_TOKEN", None)
        os.environ["DOUBAO_AUTH_TOKEN"] = "doubao_env_tok"
        q = ss.resolve_qwen()
        assert q.token == "env_tok" and q.cookie == "env_cookie=1"
        db = ss.resolve_doubao()
        assert db.token == "doubao_env_tok"

        # 无环境变量时回退到 session 文件
        ss.Session(token="file_tok", cookie="c=2").save(tmp_path / "deepseek.json")
        d = ss.resolve_deepseek()
        assert d.token == "file_tok" and d.cookie == "c=2"

        os.environ.pop("DOUBAO_AUTH_TOKEN", None)
        ss.Session(token="doubao_file_tok", cookie="db=1").save(tmp_path / "doubao.json")
        db2 = ss.resolve_doubao()
        assert db2.token == "doubao_file_tok" and db2.cookie == "db=1"

        # 都没有时返回 None
        os.environ.pop("QWEN_AUTH_TOKEN", None)
        os.environ.pop("QWEN_COOKIE", None)
        ss.QWEN_SESSION_FILE = tmp_path / "qwen_missing.json"
        assert ss.resolve_qwen() is None
        print("[PASS] 环境变量 > session 文件 > None 的解析优先级")
    finally:
        ss.QWEN_SESSION_FILE, ss.DEEPSEEK_SESSION_FILE, ss.DOUBAO_SESSION_FILE = saved_qwen, saved_ds, saved_db
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_server_reads_session_file(tmp_path: Path):
    """网关无环境变量时能从 session 文件读到令牌。"""
    import importlib

    # 用一个独立进程验证，避免污染当前 server 模块状态
    import subprocess

    saved_qwen = ss.QWEN_SESSION_FILE
    ss.QWEN_SESSION_FILE = tmp_path / "qwen.json"
    ss.Session(token="file_tok_abc", cookie="k=v").save(tmp_path / "qwen.json")
    try:
        code = (
            "import os, sys;"
            "os.environ['GATEWAY_DISABLE_AUTO_REFRESH'] = '1';"
            "os.environ.pop('QWEN_AUTH_TOKEN', None);"
            "os.environ.pop('QWEN_COOKIE', None);"
            "os.environ.pop('DEEPSEEK_AUTH_TOKEN', None);"
            "sys.path.insert(0, %r);"
            "import session_store as ss;"
            "ss.QWEN_SESSION_FILE = %r;"
            "import server;"
            "s = server.resolve_qwen();"
            "assert s and s.token == 'file_tok_abc', s;"
            "print('ok:', s.token)"
        ) % (str(Path(__file__).resolve().parent.parent), str(tmp_path / "qwen.json"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
        assert r.returncode == 0, r.stderr
        assert "ok: file_tok_abc" in r.stdout
        print("[PASS] 网关从 session 文件读取令牌")
    finally:
        ss.QWEN_SESSION_FILE = saved_qwen


def test_atomic_write_permissions_and_empty(tmp_path: Path):
    """验证原子写入、文件/目录权限设置（0o600 / 0o700）以及空文件警告安全处理。"""
    import threading
    import warnings

    sub_dir = tmp_path / "secure_dir"
    target_file = sub_dir / "session.json"

    # 1. 验证 Session.save 原子写入及目录和文件权限
    s = ss.Session(token="tok_atomic", cookie="c=1", user_agent="ua_atomic")
    s.save(target_file)

    assert target_file.exists()
    assert not any(sub_dir.glob(".*.tmp")), "不应残留临时文件"

    # 校验 POSIX 权限
    dir_mode = os.stat(sub_dir).st_mode & 0o777
    file_mode = os.stat(target_file).st_mode & 0o777
    assert dir_mode == 0o700, f"目录权限应为 0o700, 实际为 {oct(dir_mode)}"
    assert file_mode == 0o600, f"文件权限应为 0o600, 实际为 {oct(file_mode)}"

    # 2. 验证多线程并发写入时的线程安全（RLock）
    errors = []

    def writer(idx: int):
        try:
            sess = ss.Session(token=f"tok_{idx}", cookie=f"c={idx}")
            sess.save(target_file)
            loaded = ss.Session.load(target_file)
            assert loaded is not None and loaded.token.startswith("tok_")
        except Exception as ex:
            errors.append(ex)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"并发写入发生异常: {errors}"

    # 3. 验证空文件与损坏文件安全处理（触发警告并返回 None 而不崩溃）
    empty_file = sub_dir / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = ss.Session.load(empty_file)
        assert res is None, "空文件应返回 None"
        assert len(w) >= 1
        assert "为空" in str(w[-1].message)

    whitespace_file = sub_dir / "whitespace.json"
    whitespace_file.write_text("   \n\t  ", encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ss.Session.load(whitespace_file) is None
        assert len(w) >= 1
        assert "为空" in str(w[-1].message)

    corrupted_file = sub_dir / "corrupted.json"
    corrupted_file.write_text("{not a valid json", encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ss.Session.load(corrupted_file) is None
        assert len(w) >= 1
        assert "失败" in str(w[-1].message)

    non_dict_file = sub_dir / "non_dict.json"
    non_dict_file.write_text("[1, 2, 3]", encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ss.Session.load(non_dict_file) is None
        assert len(w) >= 1
        assert "非字典" in str(w[-1].message)

    print("[PASS] 原子写入、文件目录权限（0o700/0o600）与空文件异常安全处理")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_session_roundtrip(td)
        test_load_missing_and_corrupt(td)
        test_env_priority(td)
        test_server_reads_session_file(td)
        test_atomic_write_permissions_and_empty(td)
        print("\n全部通过")
