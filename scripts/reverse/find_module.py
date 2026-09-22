import re
from pathlib import Path
from playwright.sync_api import sync_playwright

profile = Path('session/profiles/doubao').resolve()

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        str(profile),
        headless=True,
        channel='chrome',
        viewport={'width': 1280, 'height': 800}
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto('https://www.doubao.com/chat/', wait_until='domcontentloaded')
    page.wait_for_timeout(3000)

    u = "https://lf-flow-web-cdn.doubao.com/obj/flow-doubao/doubao/chat/static/js/async/s1-conversation-list-v2.76742f14.js"
    res = context.request.get(u)
    text = res.text()

    # Let's find module 569306
    idx = text.find("569306:")
    if idx != -1:
        print(text[idx:idx+1500])
    else:
        print("569306 not found in this file, searching in other files...")
        # search across all JS files for 569306:
        # We can also check window.webpackChunk_ali_qianwen_web or doubao equivalent webpack chunk!

    context.close()
