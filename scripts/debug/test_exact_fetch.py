import asyncio, os, glob, json, uuid
from playwright.async_api import async_playwright
from login import QWEN_PROFILE

for p in glob.glob(str(QWEN_PROFILE / "Singleton*")):
    try: os.remove(p)
    except: pass

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            str(QWEN_PROFILE),
            headless=True,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto("https://www.qianwen.com/", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        req_id = uuid.uuid4().hex
        session_id = uuid.uuid4().hex

        js = """async ({ req_id, session_id }) => {
            let req;
            window.webpackChunk_ali_qianwen_web.push([['test_' + Date.now()], {}, (r) => { req = r; }]);
            const m = req(33669);

            const userInfo = window._USER_ || {};
            const userId = userInfo.userId || '';

            const prompt = "1+1等于几？一句话回答";
            const body = {
                req_id: req_id,
                parent_req_id: "0",
                messages: [{
                    mime_type: "text/plain",
                    content: prompt,
                    meta_data: { ori_query: prompt },
                    status: "complete"
                }],
                scene: "chat",
                sub_scene: "",
                scene_param: "first_turn",
                session_id: session_id,
                biz_id: "ai_qwen",
                topic_id: "",
                model: "Qwen3.7-Max",
                from: "default",
                protocol_version: "v2",
                messages_merge: false,
                chat_client: "h5",
                deep_search: null,
                temporary: false,
                params_extra: { supports_cowork: "false" },
                chat_mode: "quick",
                cms_test_data_ids: "@17998_C@17997_A@16860_B@17307_D@17219_B@17936_A@17260_B@16901_B@17131_A@17844_A@",
                bucket: {}
            };

            const authRes = await m.doQwenAuth({
                url: 'https://chat2.qianwen.com/api/v2/chat',
                method: 'POST',
                body: body,
                appendCommonParams: true,
            });

            const extraHeaders = {
                'x-platform': 'pc_tongyi',
                'prod_id': 'tongyi',
                'x-user-id': userId,
                'x-device-id': '0189d527-f5e4-4856-bc33-49c6c2801d47',
                'x-wpk-reqid': req_id,
                'x-chat-id': req_id,
                'x-chat-biz': JSON.stringify({
                    chatId: req_id,
                    agentId: '',
                    enableWebp: '',
                    runtimeEnabled: true,
                    debugEnabled: true
                }),
                'accept': 'application/json, text/event-stream, text/plain, */*',
                'content-type': 'application/json',
                ...authRes.signedHeaders,
            };

            const resp = await fetch(authRes.url, {
                method: 'POST',
                headers: extraHeaders,
                body: JSON.stringify(body),
                credentials: 'include',
            });

            const text = await resp.text();
            return { status: resp.status, text: text.slice(0, 1500) };
        }"""

        res = await page.evaluate(js, {"req_id": req_id, "session_id": session_id})
        print("STATUS:", res["status"])
        print("TEXT:\n", res["text"])
        await ctx.close()

asyncio.run(main())
