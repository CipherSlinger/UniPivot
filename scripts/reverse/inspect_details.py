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

    # Let's inspect the target files
    urls = [
        "https://lf-flow-web-cdn.doubao.com/obj/flow-doubao/doubao/chat/static/js/async/s1-conversation-list-v2.76742f14.js",
        "https://lf-flow-web-cdn.doubao.com/obj/flow-doubao/doubao/chat/static/js/async/s2-layout-shortcut-actions.c4604387.js",
        "https://lf-flow-web-cdn.doubao.com/obj/flow-doubao/doubao/chat/static/js/async/14852.f13f2489.js"
    ]

    for u in urls:
        print(f"================ {u.split('/')[-1]} ================")
        res = context.request.get(u)
        text = res.text()
        for kw in ["batch_del_user_conv", "operate_type", "AGWBatchDeleteConversation", "/alice/conversation/clear"]:
            pos = 0
            while True:
                pos = text.find(kw, pos)
                if pos == -1:
                    break
                print(f"--- keyword: {kw} at {pos} ---")
                start = max(0, pos - 300)
                end = min(len(text), pos + 500)
                print(text[start:end])
                pos += len(kw)

    context.close()
