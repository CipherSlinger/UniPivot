import asyncio, os, glob
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

        # Monitor requests
        page.on("request", lambda req: print("REQ:", req.method, req.url) if "api" in req.url else None)

        await page.goto("https://www.qianwen.com/", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        # Let's inspect the DOM input or search for textarea
        textarea = await page.query_selector("textarea")
        print("Textarea found:", textarea is not None)
        if textarea:
            await textarea.fill("你好，请介绍一下你自己")
            # Look for send button
            btn = await page.query_selector("div[class*='sendBtn'], button[class*='send']")
            print("Send button found:", btn is not None)
            await page.keyboard.press("Enter")
            print("Pressed enter, waiting 5 seconds for network traffic...")
            await page.wait_for_timeout(5000)

        await ctx.close()

asyncio.run(main())
