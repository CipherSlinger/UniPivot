import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

profile = Path('session/profiles/doubao').resolve()

# Clean locks
for p in profile.glob("*lock*"):
    try: p.unlink()
    except Exception: pass
for p in profile.glob("Singleton*"):
    try: p.unlink()
    except Exception: pass

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        str(profile),
        headless=True,
        channel='chrome',
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
    page = context.pages[0] if context.pages else context.new_page()

    # Search for all JS files
    page.goto('https://www.doubao.com/chat/38443458940820994', wait_until='domcontentloaded')
    page.wait_for_timeout(4000)

    # Inspect the webpack modules
    # In webpack 5, window.webpackChunk* contains modules
    res = page.evaluate("""() => {
        const chunks = Object.keys(window).filter(k => k.includes('webpackChunk'));
        return chunks;
    }""")
    print("Webpack chunks found on window:", res)

    # Let's inspect the page title and url
    print("Current URL:", page.url)
    print("Title:", page.title())

    context.close()
