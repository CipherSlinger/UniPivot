import json
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

    js_urls = []
    def on_request(req):
        if 'doubao.com' in req.url and req.url.endswith('.js'):
            js_urls.append(req.url)

    page.on('request', on_request)
    page.goto('https://www.doubao.com/chat/', wait_until='domcontentloaded')
    page.wait_for_timeout(3000)

    # Let's inspect the bundle containing BATCH_DELETE_USER_CONVERSATION or `batch_del_user_conv`
    # We saw it was in `s1-conversation-list-v2.76742f14.js`
    target_js = None
    for u in js_urls:
        if 'conversation-list' in u:
            target_js = u
            break

    if target_js:
        res = context.request.get(target_js)
        text = res.text()
        print("Found target JS, length:", len(text))
        # Search for where `batch_del_user_conv` is called or where the delete button is rendered
        # Let's search for "delete" in Chinese or English in this file
        matches = [m.start() for m in re.finditer(r'batch_del_user_conv', text)]
        for idx in matches:
            print("--- Around batch_del_user_conv ---")
            print(text[max(0, idx - 400):min(len(text), idx + 600)])

    context.close()
