import asyncio
import httpx
from session_store import resolve_doubao
from providers.doubao import DoubaoProvider

async def test_endpoint(name, url, method, json_data):
    s = resolve_doubao()
    p = DoubaoProvider(s.token, cookie=s.cookie)
    headers = p._headers()
    headers["Accept"] = "application/json, text/plain, */*"
    params = p._security_params()

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            if method == "POST":
                resp = await client.post(url, params=params, headers=headers, json=json_data)
            else:
                resp = await client.get(url, params=params, headers=headers)
            print(f"[{name}] status: {resp.status_code}, resp: {resp.text[:300]}")
            return resp
        except Exception as e:
            print(f"[{name}] error: {e}")

async def main():
    # Test conversation id created earlier: 38443458940820994
    cid = "38443458940820994"

    # Test 1: /alice/conversation/clear
    await test_endpoint(
        "alice_clear",
        "https://www.doubao.com/alice/conversation/clear",
        "POST",
        {"conversation_id": cid}
    )

    # Test 2: /samantha/im/conversation/batch_delete
    await test_endpoint(
        "samantha_batch_delete",
        "https://www.doubao.com/samantha/im/conversation/batch_delete",
        "POST",
        {"conversation_ids": [cid], "delete_all": False}
    )

    # Test 3: /im/conversation/batch_del_user_conv
    await test_endpoint(
        "im_batch_del",
        "https://www.doubao.com/im/conversation/batch_del_user_conv",
        "POST",
        {"conversation_id": [cid], "delete_all": False, "conversation_type": 1}
    )

    # Test 4: /samantha/im/conversation/list (to see format of response)
    await test_endpoint(
        "samantha_conv_list",
        "https://www.doubao.com/samantha/im/conversation/list",
        "POST",
        {"cursor": "0", "batch_size": 10, "conversation_types": [1]}
    )

    # Test 5: /alice/conversation/list
    await test_endpoint(
        "alice_conv_list",
        "https://www.doubao.com/alice/conversation/list",
        "POST",
        {"index": 0, "batch_size": 10, "conversation_types": [1]}
    )

asyncio.run(main())
