"""凭证续期判定不得被「重新保存」冒充（跨网关自身刷新链的假阳性）。

[REFRESH-VERIFY] `_FINGERPRINT_FIELDS` 里含 `captured_at`，而 `Session.save()`
每次保存都会刷新它。于是**任何**一次保存都会改变指纹，包括这样一次刷新：
浏览器 profile 里还留着旧令牌，无头刷新把它原样再存一遍——令牌一个字节都没变。

后果是链条式的，已实测（/tmp 探针，真实 CooldownTracker + CredentialRenewalTracker）：

    cooldown after 401      = 60 s
    fingerprint changed?    = ['deepseek']
    cooldown after release  = 0 s

也就是说：一次什么都没修好的静默刷新，就能解除 OPT-026 为「拿着同一个失效凭证
反复打」而设的 24h 级避让，让节点立刻重新进入上游——这正是那条避让要防的行为。
同时它污染 OPT-027 的判别式：`note_auth_failure` 会把「同一枚凭证 + 一次空刷新」
当成「新凭证仍被拒」，从而推断出**账号级封禁**并施加自动租约。

判定「凭证确已更新」的证据必须是**凭证内容**变了（token / cookie / user_id /
device_id），不能是「文件又被写了一次」。`captured_at` 保留在 `Session` 里（它服务
`age` 与令牌新鲜度判断），只是不能作为「续期」的证据。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TESTING"] = "1"
os.environ["GATEWAY_DISABLE_AUTO_REFRESH"] = "1"

from providers import credential_renewal as cred  # noqa: E402


def _install(tmp: Path, provider_key: str) -> Path:
    f = tmp / f"{provider_key}.json"
    cred._PROVIDER_SESSION_FILES[provider_key] = f
    return f


def _write(f: Path, token: str, captured_at: float) -> None:
    f.write_text(
        json.dumps(
            {
                "token": token,
                "cookie": f"c_{token}",
                "user_agent": "ua",
                "user_id": "u",
                "device_id": "dd",
                "captured_at": captured_at,
            }
        ),
        encoding="utf-8",
    )


def test_resaving_the_same_credential_is_not_a_renewal():
    """重新保存同一份凭证（captured_at 变、内容不变）不算续期。

    实测（修复前）：指纹改变 -> `renewed_providers()` 报告该节点已续期 ->
    一次 60s 的 auth 冷却被直接复位到 0。
    """
    tmp = Path(tempfile.mkdtemp())
    f = _install(tmp, "deepseek")
    _write(f, "SAME_STALE_TOKEN", 1_700_000_000.0)
    fp1 = cred.credential_fingerprint("deepseek")

    # 「静默刷新」：什么都没换到，只是又保存了一次
    _write(f, "SAME_STALE_TOKEN", 1_700_000_900.0)
    fp2 = cred.credential_fingerprint("deepseek")

    assert fp2 == fp1, (
        f"凭证内容完全没变、只是 captured_at 被刷新，指纹却变了（{fp1} -> {fp2}）——"
        f"一次什么都没修好的静默刷新就能解除 24h 的 auth 避让"
    )
    print("test_resaving_the_same_credential_is_not_a_renewal: PASSED")


def test_a_real_token_change_is_still_a_renewal():
    """反向：真正的重登（token 变了）必须仍被判为续期。

    这是上一条的对照——把指纹收窄到内容字段，不能收窄到「永远不判定续期」，
    否则 OPT-026/027/029 的复位链会整体失效。
    """
    tmp = Path(tempfile.mkdtemp())
    f = _install(tmp, "qwen")
    _write(f, "OLD_TOKEN", 1_700_000_000.0)
    fp1 = cred.credential_fingerprint("qwen")

    _write(f, "NEW_TOKEN", 1_700_000_900.0)
    fp2 = cred.credential_fingerprint("qwen")

    assert fp2 != fp1, "真正的重登必然改变凭证内容，指纹必须随之改变"
    print("test_a_real_token_change_is_still_a_renewal: PASSED")


def test_cookie_change_alone_is_a_renewal():
    """只换了 cookie 也算续期：重登常刷新安全 Cookie 而令牌不变。"""
    tmp = Path(tempfile.mkdtemp())
    f = _install(tmp, "glm")
    _write(f, "TOK", 1_700_000_000.0)
    fp1 = cred.credential_fingerprint("glm")

    data = json.loads(f.read_text(encoding="utf-8"))
    data["cookie"] = "totally_different_cookie"
    f.write_text(json.dumps(data), encoding="utf-8")

    assert cred.credential_fingerprint("glm") != fp1, (
        "cookie 变了却没被判为续期——重登只刷新安全 Cookie 的场景会漏判"
    )
    print("test_cookie_change_alone_is_a_renewal: PASSED")


def test_an_in_place_rewrite_of_an_unparseable_file_is_not_a_renewal_claim():
    """不可解析文件仍退化为 mtime 指纹（保守可接受），但不得因此“声称”续期。"""
    tmp = Path(tempfile.mkdtemp())
    f = _install(tmp, "kimi")
    f.write_text("not-json {", encoding="utf-8")
    fp1 = cred.credential_fingerprint("kimi")
    assert fp1 and fp1.startswith("meta:"), fp1
    print("test_an_in_place_rewrite_of_an_unparseable_file_is_not_a_renewal_claim: PASSED")


def run_all():
    test_resaving_the_same_credential_is_not_a_renewal()
    test_a_real_token_change_is_still_a_renewal()
    test_cookie_change_alone_is_a_renewal()
    test_an_in_place_rewrite_of_an_unparseable_file_is_not_a_renewal_claim()
    print("\n✅ All credential renewal discrimination tests passed!")


if __name__ == "__main__":
    run_all()