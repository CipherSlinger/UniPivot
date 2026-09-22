#!/usr/bin/env python3
"""扫码/账号登录并自动捕获网页端令牌（Playwright 驱动系统 Chrome）。

登录一次后，令牌自动保存到 session/qwen.json、session/deepseek.json、session/doubao.json，
网关启动时自动读取；令牌过期后可用 --refresh 静默刷新（复用已登录配置）。

Qwen 走国内版通义千问（www.qianwen.com），捕获 tongyi_sso_ticket
（或阿里云 login_aliyunid_ticket）及 WAF 安全 Cookie。

用法:
    .venv/bin/python login.py qwen            # 弹窗登录通义千问或完成人机校验
    .venv/bin/python login.py deepseek        # 弹窗登录 DeepSeek（需过 WAF/Cloudflare 人机校验）
    .venv/bin/python login.py doubao          # 弹窗登录豆包
    .venv/bin/python login.py all             # 依次登录全部站点
    .venv/bin/python login.py qwen --refresh  # 静默刷新，不弹窗
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import BrowserContext, Page, sync_playwright

from session_store import (
    DEEPSEEK_SESSION_FILE,
    DOUBAO_SESSION_FILE,
    GLM_SESSION_FILE,
    KIMI_SESSION_FILE,
    QWEN_SESSION_FILE,
    SESSION_DIR,
    Session,
)

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
]
QWEN_PROFILE = SESSION_DIR / "profiles" / "qwen"
DS_PROFILE = SESSION_DIR / "profiles" / "deepseek"
DOUBAO_PROFILE = SESSION_DIR / "profiles" / "doubao"
KIMI_PROFILE = SESSION_DIR / "profiles" / "kimi"
GLM_PROFILE = SESSION_DIR / "profiles" / "glm"

# 网页端入口
QWEN_HOME = "https://www.qianwen.com/"
DS_HOME = "https://chat.deepseek.com/"
DOUBAO_HOME = "https://www.doubao.com/chat/"
KIMI_HOME = "https://kimi.moonshot.cn/"
GLM_HOME = "https://chatglm.cn/"

# 国内版通义与豆包鉴权票据所在的 Cookie 名
QWEN_TICKET_COOKIES = ["tongyi_sso_ticket", "login_aliyunid_ticket"]
DOUBAO_TICKET_COOKIES = ["sessionid", "sessionid_ss"]
KIMI_TICKET_COOKIES = ["refresh_token", "access_token", "k_token"]
GLM_TICKET_COOKIES = ["chatglm_token", "chatglm_refresh_token", "token"]

_READ_DS_TOKEN = """
() => {
  try {
    const raw = localStorage.getItem('userToken');
    if (!raw) return null;
    const o = JSON.parse(raw);
    return (o && o.value) ? o.value : null;
  } catch (e) { return null; }
}
"""

_VERIFY_QWEN_ENDPOINT = """
async () => {
  try {
    let req;
    window.webpackChunk_ali_qianwen_web.push([['check_' + Date.now()], {}, (r) => { req = r; }]);
    let m = null;
    try { m = req(33669); } catch (e) {}
    if (!m || !m.doQwenAuth) {
      if (req && req.c) {
        for (const k in req.c) {
          if (req.c[k] && req.c[k].exports && req.c[k].exports.doQwenAuth) {
            m = req.c[k].exports;
            break;
          }
        }
      }
    }
    if (!m || !m.doQwenAuth) return { waf: false };
    const body = {
      req_id: "check_" + Date.now(),
      parent_req_id: "0",
      messages: [{mime_type: "text/plain", content: "hi", status: "complete"}],
      scene: "chat",
      scene_param: "first_turn",
      session_id: "00000000000000000000000000000000",
      biz_id: "ai_qwen",
      model: "Qwen",
      from: "default",
      protocol_version: "v2",
      chat_client: "h5",
      temporary: true,
      chat_mode: "quick",
    };
    const authRes = await m.doQwenAuth({
      url: 'https://chat2.qianwen.com/api/v2/chat',
      method: 'POST',
      body: body,
      appendCommonParams: true,
    });
    const resp = await fetch(authRes.url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream, */*',
        'x-platform': 'pc_tongyi',
        ...authRes.signedHeaders,
      },
      body: JSON.stringify(body),
      credentials: 'include',
    });
    const reader = resp.body.getReader();
    const { value } = await reader.read();
    reader.cancel();
    const text = new TextDecoder().decode(value);
    const hasWaf = text.includes('RGV587') || text.includes('FAIL_SYS_USER_VALIDATE');
    let punishUrl = null;
    if (hasWaf) {
      try {
        const parsed = JSON.parse(text);
        punishUrl = (parsed.data && parsed.data.url) || null;
      } catch (e) {}
    }
    return { waf: hasWaf, punishUrl: punishUrl };
  } catch (e) {
    return { waf: false };
  }
}
"""


def _clean_locks(profile: Path) -> None:
    """清理 Chromium 残留的单例锁文件，防止浏览器启动假死。"""
    if not profile.exists():
        return
    for pattern in ("Singleton*", "*lock*"):
        for p in profile.glob(pattern):
            try:
                if p.is_symlink() or p.is_file():
                    p.unlink()
            except Exception:
                pass


def _launch(p, profile: Path, headless: bool):
    profile.mkdir(parents=True, exist_ok=True)
    _clean_locks(profile)
    args = list(LAUNCH_ARGS)
    try:
        context = p.chromium.launch_persistent_context(
            str(profile),
            headless=headless,
            channel="chrome",
            args=args,
            user_agent=DEFAULT_UA,
        )
    except Exception:
        context = p.chromium.launch_persistent_context(
            str(profile),
            headless=headless,
            args=args,
            user_agent=DEFAULT_UA,
        )
    page = context.pages[0] if context.pages else context.new_page()
    return context, page


def _safe_goto(page: Page, url: str) -> None:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        print(f"  [提示] 页面导航被中断（{type(e).__name__}），继续等待…")
    try:
        page.wait_for_timeout(2500)
    except Exception:
        pass


def _safe_evaluate(page: Page, js: str, *args):
    """页面求值，吞掉导航中途的瞬时错误。"""
    try:
        return page.evaluate(js, *args)
    except Exception:
        return None


def _get_cookie_ticket(context: BrowserContext, cookie_names: list[str]) -> Optional[str]:
    for c in context.cookies():
        if c["name"] in cookie_names and c["value"]:
            return c["value"]
    return None


def _get_qwen_ticket(context: BrowserContext) -> Optional[str]:
    return _get_cookie_ticket(context, QWEN_TICKET_COOKIES)


def _get_doubao_ticket(context: BrowserContext) -> Optional[str]:
    return _get_cookie_ticket(context, DOUBAO_TICKET_COOKIES)


def _get_kimi_ticket(context: BrowserContext, page: Page) -> Optional[str]:
    # 优先从 localStorage 提取 refresh_token
    token = _safe_evaluate(
        page,
        """() => {
        try {
            return localStorage.getItem('refresh_token') ||
                   localStorage.getItem('access_token') ||
                   localStorage.getItem('token') ||
                   sessionStorage.getItem('refresh_token') ||
                   '';
        } catch(e) { return ''; }
    }""",
    )
    if token and str(token).strip():
        return str(token).strip()
    return _get_cookie_ticket(context, KIMI_TICKET_COOKIES)


def _get_glm_ticket(context: BrowserContext, page: Page) -> Optional[str]:
    # 优先从 localStorage 提取 chatglm_token / token / refresh_token
    token = _safe_evaluate(
        page,
        """() => {
        try {
            return localStorage.getItem('chatglm_token') ||
                   localStorage.getItem('chatglm_refresh_token') ||
                   localStorage.getItem('token') ||
                   localStorage.getItem('access_token') ||
                   '';
        } catch(e) { return ''; }
    }""",
    )
    if token and str(token).strip():
        return str(token).strip()
    return _get_cookie_ticket(context, GLM_TICKET_COOKIES)



def _save_session_from_context(context: BrowserContext, page: Page, token: str, session_file: Path) -> Session:
    cookies = {c["name"]: c["value"] for c in context.cookies()}
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
    ua = _safe_evaluate(page, "() => navigator.userAgent") or ""
    if not ua or "HeadlessChrome" in ua:
        ua = DEFAULT_UA
    uid = str(_safe_evaluate(page, "() => (window._USER_ && window._USER_.userId) || ''") or "")
    dev_id = str(_safe_evaluate(page, """() => {
        try {
            const tea = JSON.parse(localStorage.getItem('__tea_cache_tokens_497858') || '{}');
            return tea.web_id || tea.user_unique_id || '';
        } catch(e) { return ''; }
    }""") or "")
    session = Session(token=token, cookie=cookie_str, user_agent=ua, user_id=uid, device_id=dev_id)
    session.save(session_file)
    return session


def login_qwen(headless: bool = False, timeout: int = 300) -> Optional[Session]:
    """登录通义千问或完成人机验证。

    - 静默刷新 (headless=True): 后台加载页面获取最新票据与安全 Cookie。
    - 弹窗登录 (headless=False): 打开浏览器窗口进入主站。若未登录则等待登录；
      若触发人机滑块则等待用户拖动滑块完成校验，确认无阻断后保存最新票据。
    """
    with sync_playwright() as p:
        context, page = _launch(p, QWEN_PROFILE, headless)
        try:
            _safe_goto(page, QWEN_HOME)

            if headless:
                # 静默刷新模式
                page.wait_for_timeout(2000)
                token = _get_qwen_ticket(context)
                if not token:
                    print("  [失败] 未检测到通义千问票据，请使用弹窗模式登录：login.py qwen")
                    return None
                check = _safe_evaluate(page, _VERIFY_QWEN_ENDPOINT)
                if check and check.get("waf"):
                    print("  [提示] 通义千问触发人机验证（RGV587），静默刷新无法自动处理滑块，请运行：login.py qwen")
                    return None
                session = _save_session_from_context(context, page, token, QWEN_SESSION_FILE)
                print(f"  [成功] Qwen 令牌与安全 Cookie 已静默刷新（{token[:10]}…，Cookie {len(context.cookies())} 项）")
                return session

            # 弹窗交互模式
            print(f"  [等待] 浏览器已打开，正在检测通义千问登录状态与人机验证（最多等 {timeout}s）…")
            deadline = time.time() + timeout
            notified_login = False
            notified_captcha = False
            user_logged_in = False

            while time.time() < deadline:
                captcha_selectors = (
                    "div[id*='nocaptcha'], div[class*='nc_wrapper'], iframe[src*='punish'], "
                    ".baxia-dialog, div[class*='captcha'], #baxia-punish, #nc_1_wrapper"
                )
                has_captcha = False
                try:
                    has_captcha = page.locator(captcha_selectors).count() > 0 or "punish" in page.url
                except Exception as e:
                    if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                        print("  [提示] 浏览器窗口已由用户关闭。")
                        return None

                if has_captcha:
                    if not notified_captcha:
                        print("  [验证] ⚠️ 检测到阿里云安全人机验证（滑块），请在浏览器窗口中拖动滑块完成校验…")
                        notified_captcha = True
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        return None
                    continue

                # 1. 检测登录状态
                user_info = _safe_evaluate(page, "() => window._USER_ || {}") or {}
                uid = user_info.get("userId") or user_info.get("showName")
                token = _get_qwen_ticket(context)

                if not uid and not token:
                    if not notified_login:
                        print("  [提示] 请在弹出的浏览器窗口中扫码或输入账号完成登录…")
                        notified_login = True
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        return None
                    continue

                if not user_logged_in:
                    user_logged_in = True
                    display_id = uid or (token[:10] + "…" if token else "未知")
                    print(f"  [就绪] 已确认通义千问登录态（用户: {display_id}）")

                # 2. 已登录，测试对话接口是否被 WAF 拦截
                check = _safe_evaluate(page, _VERIFY_QWEN_ENDPOINT)
                if check and check.get("waf"):
                    punish_url = check.get("punishUrl")
                    if punish_url and "punish" not in page.url:
                        print("  [验证] ⚠️ 通义千问对话接口触发了人机验证（RGV587）！")
                        print("  [验证] 正在自动加载安全滑块页面，请在窗口中完成滑块验证…")
                        _safe_goto(page, punish_url)
                        notified_captcha = True
                    elif not notified_captcha:
                        print("  [验证] ⚠️ 通义千问对话接口触发了人机验证（RGV587）！请在浏览器窗口中完成滑块验证…")
                        notified_captcha = True
                    try:
                        page.wait_for_timeout(2000)
                    except Exception:
                        return None
                    continue

                # 3. 登录且接口校验通过（无滑块拦截），保存最新凭据
                if token:
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        pass
                    session = _save_session_from_context(context, page, token, QWEN_SESSION_FILE)
                    print(f"  [成功] 通义千问登录及接口验证通过！最新凭据已保存至 {QWEN_SESSION_FILE.name}（Cookie {len(context.cookies())} 项）")
                    print("  [完成] 浏览器窗口将在 3 秒后关闭…")
                    try:
                        page.wait_for_timeout(3000)
                    except Exception:
                        pass
                    return session

                try:
                    page.wait_for_timeout(1200)
                except Exception:
                    return None

            print(f"  [失败] {timeout}s 内未完成通义千问验证或登录，请重试。")
            return None
        except Exception as e:
            if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                print("  [提示] 浏览器窗口已关闭。")
                return None
            raise
        finally:
            try:
                context.close()
            except Exception:
                pass


def login_deepseek(headless: bool = False, timeout: int = 300) -> Optional[Session]:
    """登录 DeepSeek 或刷新令牌。"""
    with sync_playwright() as p:
        context, page = _launch(p, DS_PROFILE, headless)
        captured: dict = {"token": None}

        def on_request(req):
            if captured["token"]:
                return
            if "/api/" not in req.url:
                return
            auth = req.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                captured["token"] = auth[7:].strip()

        page.on("request", on_request)
        try:
            _safe_goto(page, DS_HOME)
            if not headless:
                print(f"  [等待] 请在弹出的浏览器中完成 DeepSeek 登录或 Cloudflare 人机校验（最多等 {timeout}s）…")

            deadline = time.time() + timeout
            while time.time() < deadline:
                token = captured.get("token") or _safe_evaluate(page, _READ_DS_TOKEN)
                if token:
                    session = _save_session_from_context(context, page, token, DEEPSEEK_SESSION_FILE)
                    label = "静默刷新" if headless else "登录/校验"
                    print(f"  [成功] DeepSeek {label}成功！令牌已保存到 {DEEPSEEK_SESSION_FILE.name}（{token[:10]}…）")
                    if not headless:
                        page.wait_for_timeout(3000)
                    return session
                page.wait_for_timeout(1200)

            print(f"  [失败] {timeout}s 内未捕获到 DeepSeek 令牌，请重试。")
            return None
        except Exception as e:
            if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                print("  [提示] 浏览器窗口已关闭。")
                return None
            raise
        finally:
            try:
                context.close()
            except Exception:
                pass


def login_doubao(headless: bool = False, timeout: int = 300) -> Optional[Session]:
    """登录豆包（Doubao）或静默刷新令牌。"""
    with sync_playwright() as p:
        context, page = _launch(p, DOUBAO_PROFILE, headless)
        try:
            _safe_goto(page, DOUBAO_HOME)

            if headless:
                # 静默刷新模式
                page.wait_for_timeout(2500)
                token = _get_doubao_ticket(context)
                if not token:
                    print("  [失败] 未检测到豆包有效票据，请使用弹窗模式登录：login.py doubao")
                    return None
                session = _save_session_from_context(context, page, token, DOUBAO_SESSION_FILE)
                print(f"  [成功] 豆包令牌与 Cookie 已静默刷新（{token[:10]}…，Cookie {len(context.cookies())} 项）")
                return session

            # 弹窗交互模式
            print(f"  [等待] 浏览器已打开，正在检测豆包登录状态与人机验证（最多等 {timeout}s）…")
            deadline = time.time() + timeout
            notified_login = False
            notified_captcha = False

            while time.time() < deadline:
                captcha_selectors = (
                    "iframe[src*='verify'], iframe[src*='captcha'], "
                    ".captcha-verify-container, div[class*='captcha']"
                )
                has_captcha = False
                try:
                    has_captcha = page.locator(captcha_selectors).count() > 0
                except Exception as e:
                    if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                        print("  [提示] 浏览器窗口已由用户关闭。")
                        return None

                if has_captcha:
                    if not notified_captcha:
                        print("  [验证] ⚠️ 检测到豆包安全人机验证，请在浏览器窗口中完成校验…")
                        notified_captcha = True
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        return None
                    continue

                token = _get_doubao_ticket(context)
                if not token:
                    if not notified_login:
                        print("  [提示] 请在弹出的浏览器窗口中登录豆包账号（手机号/验证码/抖音扫码）…")
                        notified_login = True
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        return None
                    continue

                # 捕获到票据且无人机验证拦截
                try:
                    page.wait_for_timeout(2000)
                except Exception:
                    pass
                session = _save_session_from_context(context, page, token, DOUBAO_SESSION_FILE)
                print(f"  [成功] 豆包登录成功！凭据已保存至 {DOUBAO_SESSION_FILE.name}（Cookie {len(context.cookies())} 项）")
                print("  [完成] 浏览器窗口将在 3 秒后关闭…")
                try:
                    page.wait_for_timeout(3000)
                except Exception:
                    pass
                return session

            print(f"  [失败] {timeout}s 内未完成豆包验证或登录，请重试。")
            return None
        except Exception as e:
            if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                print("  [提示] 浏览器窗口已关闭。")
                return None
            raise
        finally:
            try:
                context.close()
            except Exception:
                pass


def login_kimi(headless: bool = False, timeout: int = 300) -> Optional[Session]:
    """登录 Kimi (kimi.moonshot.cn) 并捕获 refresh_token 凭据。"""
    with sync_playwright() as p:
        context, page = _launch(p, KIMI_PROFILE, headless)
        try:
            _safe_goto(page, KIMI_HOME)

            if headless:
                page.wait_for_timeout(2500)
                token = _get_kimi_ticket(context, page)
                if not token:
                    print("  [失败] 未检测到 Kimi 票据，请使用弹窗模式登录：login.py kimi")
                    return None
                session = _save_session_from_context(context, page, token, KIMI_SESSION_FILE)
                print(f"  [成功] Kimi 令牌已静默刷新（{token[:10]}…，Cookie {len(context.cookies())} 项）")
                return session

            print(f"  [等待] 浏览器已打开，正在检测 Kimi 登录状态（最多等 {timeout}s）…")
            deadline = time.time() + timeout
            notified_login = False

            while time.time() < deadline:
                token = _get_kimi_ticket(context, page)
                if not token:
                    if not notified_login:
                        print("  [提示] 请在弹出的浏览器窗口中登录 Kimi 账号（微信扫码/手机号验证码）…")
                        notified_login = True
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        return None
                    continue

                try:
                    page.wait_for_timeout(2000)
                except Exception:
                    pass
                session = _save_session_from_context(context, page, token, KIMI_SESSION_FILE)
                print(f"  [成功] Kimi 登录成功！凭据已保存至 {KIMI_SESSION_FILE.name}（Cookie {len(context.cookies())} 项）")
                print("  [完成] 浏览器窗口将在 3 秒后关闭…")
                try:
                    page.wait_for_timeout(3000)
                except Exception:
                    pass
                return session

            print(f"  [失败] {timeout}s 内未完成 Kimi 登录，请重试。")
            return None
        except Exception as e:
            if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                print("  [提示] 浏览器窗口已关闭。")
                return None
            raise
        finally:
            try:
                context.close()
            except Exception:
                pass


def login_glm(headless: bool = False, timeout: int = 300) -> Optional[Session]:
    """登录 智谱清言 (chatglm.cn) 并捕获 token 凭据。"""
    with sync_playwright() as p:
        context, page = _launch(p, GLM_PROFILE, headless)
        try:
            _safe_goto(page, GLM_HOME)

            if headless:
                page.wait_for_timeout(2500)
                token = _get_glm_ticket(context, page)
                if not token:
                    print("  [失败] 未检测到智谱 GLM 票据，请使用弹窗模式登录：login.py glm")
                    return None
                session = _save_session_from_context(context, page, token, GLM_SESSION_FILE)
                print(f"  [成功] 智谱 GLM 令牌已静默刷新（{token[:10]}…，Cookie {len(context.cookies())} 项）")
                return session

            print(f"  [等待] 浏览器已打开，正在检测智谱清言登录状态（最多等 {timeout}s）…")
            deadline = time.time() + timeout
            notified_login = False

            while time.time() < deadline:
                token = _get_glm_ticket(context, page)
                if not token:
                    if not notified_login:
                        print("  [提示] 请在弹出的浏览器窗口中登录智谱清言账号（微信扫码/手机号验证码）…")
                        notified_login = True
                    try:
                        page.wait_for_timeout(1500)
                    except Exception:
                        return None
                    continue

                try:
                    page.wait_for_timeout(2000)
                except Exception:
                    pass
                session = _save_session_from_context(context, page, token, GLM_SESSION_FILE)
                print(f"  [成功] 智谱清言登录成功！凭据已保存至 {GLM_SESSION_FILE.name}（Cookie {len(context.cookies())} 项）")
                print("  [完成] 浏览器窗口将在 3 秒后关闭…")
                try:
                    page.wait_for_timeout(3000)
                except Exception:
                    pass
                return session

            print(f"  [失败] {timeout}s 内未完成智谱清言登录，请重试。")
            return None
        except Exception as e:
            if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                print("  [提示] 浏览器窗口已关闭。")
                return None
            raise
        finally:
            try:
                context.close()
            except Exception:
                pass


def refresh_qwen(timeout: int = 60) -> Optional[Session]:
    return login_qwen(headless=True, timeout=timeout)


def refresh_deepseek(timeout: int = 60) -> Optional[Session]:
    return login_deepseek(headless=True, timeout=timeout)


def refresh_doubao(timeout: int = 60) -> Optional[Session]:
    return login_doubao(headless=True, timeout=timeout)


def refresh_kimi(timeout: int = 60) -> Optional[Session]:
    return login_kimi(headless=True, timeout=timeout)


def refresh_glm(timeout: int = 60) -> Optional[Session]:
    return login_glm(headless=True, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description="扫码登录并自动捕获网页端令牌（Playwright 驱动系统 Chrome）")
    parser.add_argument("provider", choices=["qwen", "deepseek", "doubao", "kimi", "glm", "all"], help="要登录的站点")
    parser.add_argument("--refresh", action="store_true", help="静默刷新令牌（复用已登录的浏览器配置，不弹窗）")
    args = parser.parse_args()

    LOGIN_HANDLERS = {
        "qwen": login_qwen,
        "deepseek": login_deepseek,
        "doubao": login_doubao,
        "kimi": login_kimi,
        "glm": login_glm,
    }

    providers = ["qwen", "deepseek", "doubao", "kimi", "glm"] if args.provider == "all" else [args.provider]
    timeout = 60 if args.refresh else 300
    failed = False
    for name in providers:
        mode_str = "静默刷新" if args.refresh else "弹窗登录/人机校验"
        print(f"== {name.capitalize()}（{mode_str}）==")
        try:
            fn = LOGIN_HANDLERS[name]
            res = fn(headless=args.refresh, timeout=timeout)
            if not res:
                failed = True
        except Exception as e:
            if "closed" in str(e).lower() or "targetclosed" in type(e).__name__.lower():
                print("  [提示] 浏览器窗口已关闭。")
            else:
                print(f"  [错误] {type(e).__name__}: {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
