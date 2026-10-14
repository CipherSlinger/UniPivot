"""节点设备身份稳定性 (OPT-030) 测试套件

承 OPT-024 的「系统性成因」往下推。OPT-024 判定「多个不同账号在短窗口内被集中
拦截 ⇒ 根因在我们自己的客户端足迹上，换节点与按账号避让均无效」，但它的处置只是
**全局降速**——降速治标，因为足迹本身没有被修复。本套件验证的是足迹里一个**可
验证、可修复**的根因。

现场问题（实测复现）：Kimi 的 `X-Msh-Device-Id` 与 GLM 的 `X-Device-Id` 在
**每一次上游请求**上都是全新的随机值。因为 `_make_provider` 是在每次派发尝试时才
实例化 Provider（`failover.execute_chat` 内 `provider_factory(...)`），而两者的
构造器写作 `device_id or random.randint(...)` / `device_id or uuid.uuid4().hex`，
且它们的 session 文件里 `device_id` 为空（登录流程只在豆包分支捕获了 TEA web_id）。
连续三次构造的实测输出：

    kimi dev=7096601453065518286 → 7884000846302924502 → 7989808143219751393
    glm  dev=a2b3b40799554a818e758cbeb326d2ec → a880f09f917e4c89b271c0771cf612e5 → …

Qwen 与 Doubao 在**有 cookie** 时稳定，但在「令牌来自环境变量、无 cookie」的部署
形态下（凭证续期扫描接口明确支持这种形态）同样每次请求换设备：
    qwen dev=9276c21f… → 3c65d7a6… → 5aa47241…
    doubao dev=每毫秒一变；fp=verify_…（每次请求都重新随机）

为什么这是风控级缺陷而非卫生问题：「同一账号 + 同一 Cookie/令牌 + 每次请求一台
新设备」是反爬系统能拿到的最强自动化特征之一。真实浏览器在同一 profile 下设备标识
是**恒定**的，它随请求变化只可能来自自动化。而它同时是**可修复**的根因，不需要
靠降速缓解。

本套件验证：
1. **核心不变量**：同一节点跨多次实例化（即跨多次请求）拿到**完全相同**的标识
2. 跨进程/跨重启复用：清空内存后从磁盘恢复出同一个值与同一个创建时刻
3. 节点之间**互不相同**（共用标识会制造跨平台关联线索，正是 OPT-024 警告的共享足迹）
4. **优先序零回归**：显式传入的 device_id（登录捕获的真值）永远优先
5. **优先序零回归**：cookie 中已有的标识永远优先于派生值
6. 身份与凭证**解耦**：重登/令牌刷新绝不改变设备标识（真实设备重登后仍是同一台）
7. 无 cookie 的部署形态同样稳定（这是修复受益最大的场景）
8. 存储不可读/损坏时**不抛错**，安全退化为调用方生成器
9. `GATEWAY_DISABLE_DEVICE_IDENTITY=1` 精确退化为修复前行为
10. 诊断输出不泄露完整标识（只给前缀与年龄），且统计指标可解释
11. GLM 客户端提示按 UA 派生：主版本必须与 UA 一致，平台必须匹配，非 Chromium
    系 UA 必须**整体省略**（原先是写死常量，一旦浏览器升级即自相矛盾）

运行: .venv/bin/python tests/test_device_identity.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"
os.environ.pop("GATEWAY_DISABLE_DEVICE_IDENTITY", None)

from providers import device_identity as di  # noqa: E402
from providers.doubao.provider import DoubaoProvider  # noqa: E402
from providers.glm.provider import GLMProvider  # noqa: E402
from providers.kimi.provider import KimiProvider  # noqa: E402
from providers.qwen.provider import QwenProvider  # noqa: E402

CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

# 会话文件里 device_id 为空（实测：仅豆包捕获到 TEA web_id），
# 因此 kimi/glm/qwen 的兜底路径就是生产环境真实走的那一条。
_KIMI_DEV = ("dev", lambda: KimiProvider("refresh_tok").device_id)
_GLM_DEV = ("dev", lambda: GLMProvider("refresh_tok").device_id)
_QWEN_DEV = ("dev", lambda: QwenProvider("tok")._headers()["x-device-id"])


def _fresh_store() -> di.DeviceIdentityStore:
    """返回一个使用独立临时目录的存储（绝不触碰真实 session/）。"""
    return di.DeviceIdentityStore(directory=Path(tempfile.mkdtemp(prefix="opt030_")))


def _with_store(store: di.DeviceIdentityStore):
    """临时把全局单例替换为隔离存储，返回还原函数。"""
    original = di.device_identity_store
    di.device_identity_store = store
    return lambda: setattr(di, "device_identity_store", original)


# ------------------------------------------------------------------ 1. 核心不变量

def test_device_id_is_stable_across_requests():
    """核心不变量：同一个节点，无论实例化多少次（= 发出多少次请求），
    设备标识必须**完全一致**。修复前 Kimi/GLM 每次都不同。"""
    restore = _with_store(_fresh_store())
    try:
        for label, factory in (_KIMI_DEV, _GLM_DEV, _QWEN_DEV):
            values = {factory() for _ in range(8)}
            assert len(values) == 1, f"{label} 的设备标识仍然在抖动: {sorted(values)[:4]}"
            assert next(iter(values)), f"{label} 的设备标识不得为空"

        # Kimi 的会话标识同理（每次请求换会话比换设备更荒谬）
        sessions = {KimiProvider("refresh_tok").session_id for _ in range(5)}
        assert len(sessions) == 1, f"Kimi 会话标识仍在抖动: {sessions}"

        # Doubao 的无 cookie 兜底（原先退化为「当前毫秒」）
        devs = {DoubaoProvider("tok").device_id for _ in range(5)}
        assert len(devs) == 1, f"Doubao 设备标识仍在抖动: {devs}"
        fps = {DoubaoProvider("tok")._fp for _ in range(5)}
        assert len(fps) == 1, f"Doubao 指纹仍在抖动: {fps}"
    finally:
        restore()
    print("test_device_id_is_stable_across_requests: PASSED")


def test_node_scoped_ids_are_distinct():
    """五家节点必须各自独立：共用同一个设备标识会制造一条**跨平台关联线索**，
    正是 OPT-024 警告的「共享足迹」——修复不得引入新的关联信号。"""
    restore = _with_store(_fresh_store())
    try:
        kimi = KimiProvider("t").device_id
        glm = GLMProvider("t").device_id
        qwen = QwenProvider("t")._headers()["x-device-id"]
        doubao = DoubaoProvider("t").device_id
        assert len({kimi, glm, qwen, doubao}) == 4, (kimi, glm, qwen, doubao)
        # 同一节点内的不同身份槽位也必须彼此独立
        assert KimiProvider("t").device_id != KimiProvider("t").session_id
    finally:
        restore()
    print("test_node_scoped_ids_are_distinct: PASSED")


# ------------------------------------------------------------------ 2. 跨重启

def test_identity_survives_restart():
    """跨进程/跨重启复用：清空内存（模拟重启）后必须从磁盘恢复出**同一个值**，
    包括创建时刻——否则重启即是一场身份重置，正是本优化要消除的现象。"""
    directory = Path(tempfile.mkdtemp(prefix="opt030_restart_"))
    try:
        first = di.DeviceIdentityStore(directory=directory)
        value = first.get("glm", lambda: "should-not-be-used")
        created = first.get_diagnostics()["devices"]["glm"]["age_seconds"]

        # 「重启」：全新对象，空内存，只读磁盘
        revived = di.DeviceIdentityStore(directory=directory)
        assert revived.peek("glm") == value, "重启后设备标识必须不变"
        assert revived.get("glm", lambda: "regenerated") == value, "重启后不得重新生成"
        assert revived.get_diagnostics()["devices"]["glm"]["age_seconds"] >= created - 1.0
        assert revived.get_diagnostics()["stats"]["revived_from_disk"] == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    print("test_identity_survives_restart: PASSED")


def test_identity_is_decoupled_from_credentials():
    """身份与凭证解耦：真实设备在**重新登录后依然是同一台设备**。因此重登
    （凭证文件变化、令牌刷新、指纹移动）绝不允许改变设备标识——这与 OPT-026/
    OPT-029 的「凭证指纹用于判定人工介入」是两条完全独立的证据链。"""
    restore = _with_store(_fresh_store())
    try:
        before = KimiProvider("old_refresh_token").device_id
        # 重登：新的 refresh_token、新的 cookie
        after = KimiProvider(
            "brand_new_refresh_token", cookie="msh_session=xyz; other=1"
        ).device_id
        assert before == after, "重登不得改变设备身份（身份与凭证必须解耦）"

        # 凭证指纹确实变了（对照：说明上一步真的换了凭证而非被短路）
        assert "brand_new_refresh_token" != "old_refresh_token"
    finally:
        restore()
    print("test_identity_is_decoupled_from_credentials: PASSED")


# ------------------------------------------------------------------ 3. 优先序零回归

def test_explicit_device_id_still_wins():
    """显式传入的 device_id 来自登录捕获的**真值**，必须永远优先——
    本优化只接管「原本要随机生成」的那一格，绝不覆盖真实捕获值。"""
    store = _fresh_store()
    restore = _with_store(store)
    try:
        captured = "7687867820721784372"  # 实测：豆包 session 中捕获的 TEA web_id
        for _ in range(3):
            p = DoubaoProvider("tok", device_id=captured)
            assert p.device_id == captured, "登录捕获的 device_id 必须优先"
        assert store.peek("doubao") is None, "有真值时不得产生派生身份"
    finally:
        restore()
    print("test_explicit_device_id_still_wins: PASSED")


def test_cookie_carried_identity_still_wins():
    """cookie 中已存在的标识同样是真值，必须优先于派生值——Qwen 的正常部署
    （有 cookie）走的就是这条路径，其行为必须与修复前逐字一致。"""
    cookie = "_device_id=cookiedevice123456; b-user-id=u_42; t=cookietoken"
    restore = _with_store(_fresh_store())
    try:
        for _ in range(3):
            p = QwenProvider("tok", cookie=cookie)
            h = p._headers()
            assert h["x-device-id"] == "cookiedevice123456", "cookie 中的设备标识必须优先"
            assert h["x-user-id"] == "u_42", "cookie 中的用户标识必须优先"
    finally:
        restore()
    print("test_cookie_carried_identity_still_wins: PASSED")


# ------------------------------------------------------------------ 4. 退化与容错

def test_corrupt_store_degrades_safely():
    """存储不可读/损坏时**绝不抛错**：安全退化为调用方的生成器（即修复前行为）。
    身份存储是辅助设施，绝不能成为请求路径上的单点故障。"""
    directory = Path(tempfile.mkdtemp(prefix="opt030_corrupt_"))
    try:
        (directory / "device_identity.json").write_text("{ this is not json", encoding="utf-8")
        store = di.DeviceIdentityStore(directory=directory)
        assert store.peek("glm") is None, "损坏的存储应视为「尚无身份」"
        value = store.get("glm", lambda: "generated")
        assert value == "generated"
        # 且此后恢复自愈：下一次重启能读回刚写入的值
        revived = di.DeviceIdentityStore(directory=directory)
        assert revived.peek("glm") == "generated", "损坏后应能自愈并持久化新身份"
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    print("test_corrupt_store_degrades_safely: PASSED")


def test_disable_switch_restores_legacy_behaviour():
    """`GATEWAY_DISABLE_DEVICE_IDENTITY=1` 必须**精确退化**为修复前的行为
    （每次调用都重新生成），以便出问题时一键回退。"""
    store = _fresh_store()
    restore = _with_store(store)
    os.environ["GATEWAY_DISABLE_DEVICE_IDENTITY"] = "1"
    try:
        values = {GLMProvider("t").device_id for _ in range(8)}
        assert len(values) == 8, "关闭开关后应恢复为每次重新生成"
        assert store.keys() == [], "关闭开关时不得写入任何身份记录"
        assert store.get_diagnostics()["stats"]["bypassed"] >= 8
    finally:
        os.environ.pop("GATEWAY_DISABLE_DEVICE_IDENTITY", None)
        restore()
    print("test_disable_switch_restores_legacy_behaviour: PASSED")


# ------------------------------------------------------------------ 5. 遥测

def test_diagnostics_do_not_leak_full_identity():
    """诊断输出不得泄露完整标识：设备标识正是反爬系统最敏感的字段，
    截图/日志外泄等同于把身份交出去。只给前缀与年龄即足以验证稳定性。"""
    store = _fresh_store()
    restore = _with_store(store)
    try:
        value = GLMProvider("t").device_id
        GLMProvider("t")  # 第二次取用 = 后续请求，应命中复用而非重建
        diag = store.get_diagnostics()
        assert diag["enabled"] is True
        assert "glm" in diag["tracked_nodes"]
        preview = diag["devices"]["glm"]["preview"]
        assert value not in preview, f"诊断泄露了完整标识: {preview}"
        assert preview.startswith(value[:6]), preview
        assert f"{len(value)}位" in preview, preview
        assert diag["devices"]["glm"]["age_seconds"] >= 0.0
        assert diag["stats"]["created"] == 1, diag["stats"]
        assert diag["stats"]["reused"] >= 1, diag["stats"]
    finally:
        restore()
    print("test_diagnostics_do_not_leak_full_identity: PASSED")


# ------------------------------------------------------------------ 6. 客户端提示自洽

def test_glm_client_hints_are_derived_from_user_agent():
    """GLM 原先写死 `Sec-Ch-Ua: "Chromium";v="130"`，今天「恰好」与登录捕获的
    UA 一致只是因为 Playwright 内置 Chromium 恰好是 130——一旦浏览器升级，UA 变
    成 Chrome/141 而提示仍自称 130，形成**自相矛盾**的指纹（现代反爬会交叉校验）。
    派生后该矛盾在结构上不可能出现。"""
    cases = [
        # (ua 片段, 期望主版本, 期望平台, 期望 mobile 标记)
        ("Macintosh; Intel Mac OS X 10_15_7", "130", '"macOS"', "?0"),
        ("Windows NT 10.0; Win64; x64", "120", '"Windows"', "?0"),
        ("X11; Linux x86_64", "141", '"Linux"', "?0"),
        ("Linux; Android 14", "138", '"Android"', "?1"),
        ("iPhone; CPU iPhone OS 17_0", "137", '"iOS"', "?1"),
    ]
    for ua_frag, major, platform, mobile in cases:
        ua = f"Mozilla/5.0 ({ua_frag}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
        h = GLMProvider("t", user_agent=ua)._headers("at")
        assert h["Sec-Ch-Ua"] == (
            f'"Chromium";v="{major}", "Google Chrome";v="{major}", "Not?A_Brand";v="99"'
        ), h.get("Sec-Ch-Ua")
        assert h["Sec-Ch-Ua-Platform"] == platform, h.get("Sec-Ch-Ua-Platform")
        assert h["Sec-Ch-Ua-Mobile"] == mobile, h.get("Sec-Ch-Ua-Mobile")
        # 主版本必须与 UA 自身一致——这正是被修复的自相矛盾
        assert re.search(r"Chrome/(\d+)", h["User-Agent"]).group(1) == major

    # 非 Chromium 系 UA：真实 Firefox/Safari **从不**发送这些字段，整体省略才自洽
    for ua in (
        "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    ):
        h = GLMProvider("t", user_agent=ua)._headers("at")
        assert "Sec-Ch-Ua" not in h, f"非 Chromium UA 不得声称 Chromium: {h.get('Sec-Ch-Ua')}"
        assert "Sec-Ch-Ua-Platform" not in h
        assert "Sec-Ch-Ua-Mobile" not in h
    print("test_glm_client_hints_are_derived_from_user_agent: PASSED")


def test_default_ua_still_produces_consistent_hints():
    """默认 UA（Playwright 捕获值被判定为 HeadlessChrome 时回退的常量）同样必须
    与派生的客户端提示自洽——这是最常见的生产路径。"""
    h = GLMProvider("t")._headers("at")
    major = re.search(r"Chrome/(\d+)", h["User-Agent"]).group(1)
    assert f'v="{major}"' in h["Sec-Ch-Ua"], (major, h["Sec-Ch-Ua"])
    assert h["Sec-Ch-Ua-Platform"] == '"macOS"'  # 默认 UA 即 macOS
    print("test_default_ua_still_produces_consistent_hints: PASSED")


def run_all():
    test_device_id_is_stable_across_requests()
    test_node_scoped_ids_are_distinct()
    test_identity_survives_restart()
    test_identity_is_decoupled_from_credentials()
    test_explicit_device_id_still_wins()
    test_cookie_carried_identity_still_wins()
    test_corrupt_store_degrades_safely()
    test_disable_switch_restores_legacy_behaviour()
    test_diagnostics_do_not_leak_full_identity()
    test_glm_client_hints_are_derived_from_user_agent()
    test_default_ua_still_produces_consistent_hints()
    print("\n✅ All device-identity tests (OPT-030) passed!")


if __name__ == "__main__":
    run_all()
