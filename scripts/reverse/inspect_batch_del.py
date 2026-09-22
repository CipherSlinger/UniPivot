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

    # Search around 716701 and BATCH_DELETE_USER_CONVERSATION
    idx = text.find("batch_del_user_conv")
    print(text[max(0, idx - 500):min(len(text), idx + 1000)])

    context.close()
