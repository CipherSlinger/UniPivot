import asyncio
import json
from pathlib import Path
from providers.doubao import DoubaoProvider
from session_store import resolve_doubao

async def main():
    s = resolve_doubao()
    assert s, "No doubao session"
    p = DoubaoProvider(s.token, cookie=s.cookie)
    messages = [{"role": "user", "content": "测试会话"}]
    print("Creating a conversation...")
    cid = None
    async for delta, meta in p.chat(messages, "doubao"):
        if "conversation_id" in meta:
            cid = meta["conversation_id"]
            print("Captured conversation_id:", cid)
    print("Done chat, cid:", cid)

asyncio.run(main())
