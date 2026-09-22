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

    page.goto('https://www.doubao.com/chat/', wait_until='domcontentloaded')
    print("Page loaded. Waiting for sidebar to appear...")
    page.wait_for_timeout(5000)

    # Let's inspect conversation items on the sidebar
    # Look for items with conversation id or links
    # Let's dump the HTML structure of sidebar or conversation list
    sidebar_items = page.evaluate("""() => {
        const items = document.querySelectorAll('[data-testid*="conversation"], [class*="conversation-item"], [class*="history-item"], [class*="chat-item"]');
        return Array.from(items).map(el => ({
            tag: el.tagName,
            className: el.className,
            text: el.innerText ? el.innerText.slice(0, 50) : '',
            innerHTML: el.innerHTML.slice(0, 100)
        }));
    }""")
    print("Found sidebar items count:", len(sidebar_items))
    if sidebar_items:
        print("First 3 items:", json.dumps(sidebar_items[:3], ensure_ascii=False, indent=2))

    # Let's also check all links
    links = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('a')).map(a => ({
            href: a.href,
            text: a.innerText ? a.innerText.slice(0, 30) : ''
        })).filter(a => a.href.includes('/chat/'));
    }""")
    print("Found chat links count:", len(links))
    for l in links[:5]:
        print("  Link:", l)

    context.close()
