import urllib.request
import json
from session_store import resolve_doubao

s = resolve_doubao()
req = urllib.request.Request("https://www.doubao.com/chat/", headers={"User-Agent": "curl/7.88.1"})
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        print("Status:", resp.status)
except Exception as e:
    print("Error:", e)
