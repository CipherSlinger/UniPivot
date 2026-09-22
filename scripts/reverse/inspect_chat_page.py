import json
from pathlib import Path
from playwright.sync_api import sync_playwright

profile = Path('session/profiles/doubao').resolve()

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        str(profile),
        headless=False,
        channel='chrome',
        viewport={'width': 1280, 'height': 800}
    )
    page = context.pages[0] if context.pages else context.new_page()

    requests_log = []

    def on_request(req):
        if 'doubao.com' in req.url and not any(ext in req.url for ext in ['.js', '.css', '.png', '.jpg', '.svg', '.woff', '.ico', '.webp']):
            requests_log.append({
                'method': req.method,
                'url': req.url,
                'headers': req.headers,
                'post_data': req.post_data
            })

    page.on('request', on_request)

    page.goto('https://www.doubao.com/chat/38443458940820994', wait_until='domcontentloaded')
    print("Page loaded.")
    page.wait_for_timeout(6000)

    # Let's inspect the page content / buttons
    btns = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('button, div[role=\"button\"], svg')).map(el => {
            return {
                tag: el.tagName,
                aria: el.getAttribute('aria-label'),
                testid: el.getAttribute('data-testid'),
                className: el.className ? String(el.className).slice(0, 50) : '',
                title: el.getAttribute('title'),
                text: el.innerText ? el.innerText.slice(0, 30) : ''
            };
        }).filter(b => b.aria || b.title || (b.text && b.text.trim().length > 0));
    }""")
    print("Found buttons/clickable count:", len(btns))
    for b in btns:
        if any(w in str(b) for w in ['删', 'del', '更多', 'more', '操作', '设置', '清理']):
            print("  Interesting button:", b)

    context.close()
