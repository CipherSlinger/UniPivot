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

    # Find the function that creates `(0,i.S)({url:"/im/conversation/batch_del_user_conv"`
    # Let's inspect module 569306 or search for uplinkKey
    print("Searching for uplinkKey logic...")
    # Find definition of i.S or function with uplinkKey
    matches = [m.start() for m in re.finditer(r'uplinkKey', text)]
    for pos in matches:
        print(text[max(0, pos - 200):min(len(text), pos + 300)])
        print("=" * 40)

    context.close()
