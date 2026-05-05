import asyncio
import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP
from PIL import Image


mcp = FastMCP("browser-control")

DEFAULT_TIMEOUT_MS = 15_000
DEFAULT_SETTLE_MS = 500

_playwright: Any = None
_browser: Any = None
_browser_headless: Optional[bool] = None
_sessions: dict[str, "BrowserSession"] = {}
_global_lock = asyncio.Lock()
_MAX_LOG_EVENTS = 1_000
DEFAULT_SESSION_ID = "default"

BROWSER_CONTROL_CAPABILITIES = {
    "recommended_flow": [
        "browser_open_url",
        "browser_compact_state",
        "browser_click_semantic",
        "browser_type_semantic",
        "browser_fill_form",
        "browser_select_option_semantic",
        "browser_visual_checkpoint",
        "browser_get_issue_summary",
    ],
    "core_sessions": [
        "browser_create_session",
        "browser_list_sessions",
        "browser_status",
        "browser_close_session",
        "browser_close_all",
    ],
    "core_navigation": [
        "browser_open_url",
        "browser_compact_state",
        "browser_snapshot",
        "browser_wait_for_idle",
        "browser_wait_for_text",
        "browser_wait_for_selector",
        "browser_wait_for_url",
    ],
    "core_actions": [
        "browser_click_semantic",
        "browser_type_semantic",
        "browser_fill_form",
        "browser_select_option_semantic",
        "browser_key",
    ],
    "visual": [
        "browser_visual_checkpoint",
        "browser_screenshot",
    ],
    "diagnostics": [
        "browser_get_issue_summary",
        "browser_get_console_logs",
        "browser_get_page_errors",
        "browser_get_network_errors",
        "browser_get_http_errors",
        "browser_clear_logs",
    ],
    "human_in_the_loop": [
        "browser_request_human_input",
        "browser_get_human_checkpoints",
        "browser_resolve_human_checkpoint",
    ],
    "fallback_escape_hatches": [
        "browser_click",
        "browser_type",
        "browser_evaluate",
        "browser_close",
    ],
    "principles": [
        "Prefer semantic actions over element indexes.",
        "Use browser_fill_form for long forms; do not parallelize field entry on one page.",
        "Use visual checkpoints plus view_image for human-visible layout claims.",
        "Check issue summary when behavior is broken or suspicious.",
        "Use one session_id per role/user when testing multi-role flows.",
    ],
}


@dataclass
class BrowserSession:
    session_id: str
    role: str
    context: Any
    page: Any
    created_at: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    console_logs: list[dict[str, Any]] = field(default_factory=list)
    page_errors: list[dict[str, Any]] = field(default_factory=list)
    network_errors: list[dict[str, Any]] = field(default_factory=list)
    http_errors: list[dict[str, Any]] = field(default_factory=list)
    human_checkpoints: list[dict[str, Any]] = field(default_factory=list)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(buffer: list[dict[str, Any]], event: dict[str, Any]) -> None:
    buffer.append({"timestamp": _now_iso(), **event})
    if len(buffer) > _MAX_LOG_EVENTS:
        del buffer[: len(buffer) - _MAX_LOG_EVENTS]


async def _console_message_payload(message: Any) -> list[str]:
    values: list[str] = []
    for arg in message.args:
        try:
            values.append(str(await arg.json_value()))
        except Exception:
            try:
                values.append(await arg.evaluate("arg => String(arg)"))
            except Exception:
                values.append(str(arg))
    return values


def _install_page_observers(page: Any, session: BrowserSession) -> None:
    def on_console(message: Any) -> None:
        async def collect() -> None:
            _append_event(
                session.console_logs,
                {
                    "session_id": session.session_id,
                    "role": session.role,
                    "type": message.type,
                    "text": message.text,
                    "location": message.location,
                    "args": await _console_message_payload(message),
                    "url": page.url,
                },
            )

        asyncio.create_task(collect())

    def on_page_error(error: Exception) -> None:
        _append_event(
            session.page_errors,
            {
                "session_id": session.session_id,
                "role": session.role,
                "message": str(error),
                "url": page.url,
            },
        )

    def on_request_failed(request: Any) -> None:
        failure = request.failure or {}
        _append_event(
            session.network_errors,
            {
                "session_id": session.session_id,
                "role": session.role,
                "url": request.url,
                "method": request.method,
                "resource_type": request.resource_type,
                "failure": failure,
                "page_url": page.url,
            },
        )

    def on_response(response: Any) -> None:
        if response.status < 400:
            return
        request = response.request
        _append_event(
            session.http_errors,
            {
                "session_id": session.session_id,
                "role": session.role,
                "url": response.url,
                "status": response.status,
                "status_text": response.status_text,
                "method": request.method,
                "resource_type": request.resource_type,
                "page_url": page.url,
            },
        )

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("requestfailed", on_request_failed)
    page.on("response", on_response)


def _filter_events(
    events: list[dict[str, Any]],
    limit: int,
    level: Optional[str] = None,
    contains: Optional[str] = None,
) -> list[dict[str, Any]]:
    filtered = events
    if level:
        filtered = [event for event in filtered if event.get("type") == level]
    if contains:
        needle = contains.lower()
        filtered = [
            event
            for event in filtered
            if needle in json.dumps(event, ensure_ascii=False, default=str).lower()
        ]
    return filtered[-limit:]


async def _ensure_browser(headless: bool = True) -> Any:
    global _playwright, _browser, _browser_headless

    if _browser:
        return _browser

    from playwright.async_api import async_playwright

    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(
        headless=headless,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    _browser_headless = headless
    return _browser


async def _create_session(session_id: str, role: Optional[str] = None, headless: bool = True) -> BrowserSession:
    if session_id in _sessions:
        return _sessions[session_id]

    browser = await _ensure_browser(headless=headless)
    context = await browser.new_context(
        viewport={"width": 1440, "height": 1000},
        locale="en-US",
        timezone_id=os.getenv("TZ", "America/Monterrey"),
    )
    context.set_default_timeout(DEFAULT_TIMEOUT_MS)
    page = await context.new_page()
    session = BrowserSession(
        session_id=session_id,
        role=role or session_id,
        context=context,
        page=page,
        created_at=_now_iso(),
    )
    _install_page_observers(page, session)
    _sessions[session_id] = session
    return session


async def _ensure_session(
    session_id: str = DEFAULT_SESSION_ID,
    role: Optional[str] = None,
    headless: bool = True,
) -> BrowserSession:
    async with _global_lock:
        session = _sessions.get(session_id)
        if session and session.page and not session.page.is_closed():
            return session
        if session:
            _sessions.pop(session_id, None)
        return await _create_session(session_id=session_id, role=role, headless=headless)


async def _current_session(session_id: str = DEFAULT_SESSION_ID) -> BrowserSession:
    session = _sessions.get(session_id)
    if not session or not session.page or session.page.is_closed():
        raise RuntimeError(
            f"No browser page is open for session '{session_id}'. Call browser_open_url or browser_create_session first."
        )
    return session


async def _wait_for_page_ready(
    page: Any,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    wait_for_networkidle: bool = False,
    require_body_text: bool = False,
    settle_ms: int = DEFAULT_SETTLE_MS,
) -> dict[str, Any]:
    """
    Wait until the page is stable enough for DOM inspection or screenshots.
    The body-text check prevents black/blank early captures on slow SPAs.
    """
    deadline = asyncio.get_running_loop().time() + (timeout_ms / 1000)
    notes: list[str] = []

    async def time_left_ms() -> int:
        remaining = int((deadline - asyncio.get_running_loop().time()) * 1000)
        return max(250, remaining)

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=await time_left_ms())
    except Exception as exc:
        notes.append(f"domcontentloaded timeout: {exc}")

    try:
        await page.wait_for_function(
            "() => ['interactive', 'complete'].includes(document.readyState)",
            timeout=await time_left_ms(),
        )
    except Exception as exc:
        notes.append(f"document.readyState timeout: {exc}")

    if wait_for_networkidle:
        try:
            await page.wait_for_load_state("networkidle", timeout=await time_left_ms())
        except Exception as exc:
            notes.append(f"networkidle timeout: {exc}")

    if require_body_text:
        try:
            await page.wait_for_function(
                """() => {
                    const body = document.body;
                    if (!body) return false;
                    const text = (body.innerText || '').trim();
                    const rect = body.getBoundingClientRect();
                    return text.length > 0 && rect.width > 0 && rect.height > 0;
                }""",
                timeout=await time_left_ms(),
            )
        except Exception as exc:
            notes.append(f"body text timeout: {exc}")

    if settle_ms > 0:
        await page.wait_for_timeout(settle_ms)

    return {
        "ready": True,
        "url": page.url,
        "timestamp": _now_iso(),
        "notes": notes,
    }


async def _observe_ui_signals(page: Any) -> dict[str, Any]:
    try:
        signals = await page.evaluate(
            """() => {
                const selectors = [
                    '[role="alert"]',
                    '[aria-live]',
                    '.toast',
                    '.Toastify__toast',
                    '.notification',
                    '.alert',
                    '.error',
                    '.success',
                    '[data-sonner-toast]'
                ];
                const items = [];
                for (const selector of selectors) {
                    for (const el of document.querySelectorAll(selector)) {
                        const rect = el.getBoundingClientRect();
                        const text = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
                        if (!text) continue;
                        items.push({
                            selector,
                            text: text.slice(0, 300),
                            visible: !!(rect.width || rect.height),
                            box: {
                                x: Math.round(rect.x),
                                y: Math.round(rect.y),
                                width: Math.round(rect.width),
                                height: Math.round(rect.height)
                            }
                        });
                    }
                }
                return items.slice(0, 20);
            }"""
        )
        return {"messages": signals}
    except Exception as exc:
        return {"messages": [], "error": str(exc)}


async def _action_observation(
    session: BrowserSession,
    before_url: str,
    before_title: str,
) -> dict[str, Any]:
    page = session.page
    after_title = await page.title()
    ui_signals = await _observe_ui_signals(page)
    return {
        "url_changed": page.url != before_url,
        "title_changed": after_title != before_title,
        "before_url": before_url,
        "after_url": page.url,
        "before_title": before_title,
        "after_title": after_title,
        "ui_signals": ui_signals,
        "issue_counts": _issue_counts(session),
    }


async def _resolve_locator(
    page: Any,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    placeholder: Optional[str] = None,
    name: Optional[str] = None,
    text: Optional[str] = None,
    role: Optional[str] = None,
    exact: bool = False,
) -> Any:
    if selector:
        return page.locator(selector).first
    if label:
        return page.get_by_label(label, exact=exact).first
    if placeholder:
        return page.get_by_placeholder(placeholder, exact=exact).first
    if role and name:
        return page.get_by_role(role, name=name, exact=exact).first
    if name:
        return page.locator(f'[name="{name}"], #{name}').first
    if text:
        return page.get_by_text(text, exact=exact).first
    raise ValueError("Provide selector, label, placeholder, name, role+name, or text.")


def _issue_counts(session: BrowserSession) -> dict[str, int]:
    return {
        "console_warnings_or_errors": len(
            [event for event in session.console_logs if event.get("type") in {"warning", "error"}]
        ),
        "page_errors": len(session.page_errors),
        "network_errors": len(session.network_errors),
        "http_errors": len(session.http_errors),
    }


async def _compact_state(session: BrowserSession, text_chars: int = 1_000, max_elements: int = 20) -> dict[str, Any]:
    page = session.page
    try:
        text = await page.locator("body").inner_text(timeout=3_000)
    except Exception:
        text = ""
    text = text.strip()
    if len(text) > text_chars:
        text = text[:text_chars] + "\n...[truncated]"

    elements = await page.locator(
        "a,button,input,textarea,select,[role=button],[role=link],[contenteditable=true]"
    ).evaluate_all(
        """(els, maxElements) => els.slice(0, maxElements).map((el, index) => {
            const rect = el.getBoundingClientRect();
            const label = (
                el.getAttribute('aria-label') ||
                el.getAttribute('placeholder') ||
                el.innerText ||
                el.value ||
                el.textContent ||
                ''
            ).trim().replace(/\\s+/g, ' ').slice(0, 100);
            return {
                index,
                tag: el.tagName.toLowerCase(),
                role: el.getAttribute('role'),
                type: el.getAttribute('type'),
                label,
                visible: !!(rect.width || rect.height),
                box: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                }
            };
        })""",
        max_elements,
    )

    return {
        "session_id": session.session_id,
        "role": session.role,
        "url": page.url,
        "title": await page.title(),
        "text_preview": text,
        "top_interactive_elements": elements,
        "issue_counts": _issue_counts(session),
    }


async def _action_result(
    session: BrowserSession,
    ready: Optional[dict[str, Any]] = None,
    include_snapshot: bool = False,
    include_compact_state: bool = True,
    observation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    page = session.page
    result: dict[str, Any] = {
        "ok": True,
        "session_id": session.session_id,
        "role": session.role,
        "url": page.url,
        "title": await page.title(),
        "ready": ready,
        "issue_counts": _issue_counts(session),
    }
    if observation:
        result["observation"] = observation
    if include_compact_state:
        result["state"] = await _compact_state(session)
    if include_snapshot:
        result["snapshot"] = await _snapshot(session)
    return result


async def _snapshot(session: BrowserSession, max_text_chars: int = 8_000, max_elements: int = 80) -> dict[str, Any]:
    page = session.page
    title = await page.title()
    url = page.url
    try:
        text = await page.locator("body").inner_text(timeout=5_000)
    except Exception:
        text = ""
    if len(text) > max_text_chars:
        text = text[:max_text_chars] + "\n...[truncated]"

    elements = await page.locator(
        "a,button,input,textarea,select,[role=button],[role=link],[contenteditable=true]"
    ).evaluate_all(
        """(els, maxElements) => els.slice(0, maxElements).map((el, index) => {
            const rect = el.getBoundingClientRect();
            const label = (
                el.getAttribute('aria-label') ||
                el.getAttribute('placeholder') ||
                el.innerText ||
                el.value ||
                el.textContent ||
                ''
            ).trim().replace(/\\s+/g, ' ').slice(0, 160);
            return {
                index,
                tag: el.tagName.toLowerCase(),
                role: el.getAttribute('role'),
                type: el.getAttribute('type'),
                name: el.getAttribute('name'),
                id: el.id,
                label,
                visible: !!(rect.width || rect.height),
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                width: Math.round(rect.width),
                height: Math.round(rect.height),
            };
        })""",
        max_elements,
    )

    return {
        "session_id": session.session_id,
        "role": session.role,
        "url": url,
        "title": title,
        "text": text,
        "interactive_elements": elements,
        "issue_counts": _issue_counts(session),
    }


async def _screenshot_metadata(page: Any, path: Path, full_page: bool) -> dict[str, Any]:
    viewport = page.viewport_size
    image_width = None
    image_height = None
    try:
        with Image.open(path) as image:
            image_width, image_height = image.size
    except Exception:
        pass

    return {
        "path": str(path),
        "url": page.url,
        "title": await page.title(),
        "timestamp": _now_iso(),
        "full_page": full_page,
        "viewport": viewport,
        "image": {
            "width": image_width,
            "height": image_height,
        },
        "review_required": "If this screenshot is used for visual QA, open this path with view_image before making visual claims.",
    }


@mcp.tool()
async def browser_capabilities() -> str:
    """
    Return the recommended browser-control tool surface and usage principles.
    """
    return _json({"ok": True, "capabilities": BROWSER_CONTROL_CAPABILITIES})


@mcp.tool()
async def browser_create_session(
    session_id: str,
    role: Optional[str] = None,
    headless: bool = True,
) -> str:
    """
    Create an isolated browser session with its own cookies, localStorage, page, and logs.
    """
    try:
        session = await _ensure_session(session_id=session_id, role=role, headless=headless)
        return _json(
            {
                "ok": True,
                "session_id": session.session_id,
                "role": session.role,
                "created_at": session.created_at,
                "url": session.page.url,
                "open_human_checkpoints": len(
                    [item for item in session.human_checkpoints if not item.get("resolved")]
                ),
                "note": "Use this session_id in browser tools to keep role cookies/storage isolated.",
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_list_sessions() -> str:
    """
    List active isolated browser sessions.
    """
    return _json(
        {
            "ok": True,
            "browser_open": bool(_browser),
            "browser_headless": _browser_headless,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "role": session.role,
                    "created_at": session.created_at,
                    "page_open": bool(session.page and not session.page.is_closed()),
                    "url": session.page.url if session.page and not session.page.is_closed() else None,
                    "issue_counts": _issue_counts(session),
                    "open_human_checkpoints": len(
                        [item for item in session.human_checkpoints if not item.get("resolved")]
                    ),
                }
                for session in _sessions.values()
            ],
        }
    )


@mcp.tool()
async def browser_status() -> str:
    """
    Return browser-control MCP status. This MCP does not call any LLM or external API.
    """
    try:
        import playwright

        return _json(
            {
                "ok": True,
                "playwright": str(playwright),
                "browser_open": bool(_browser),
                "browser_headless": _browser_headless,
                "recommended_flow": BROWSER_CONTROL_CAPABILITIES["recommended_flow"],
                "fallback_escape_hatches": BROWSER_CONTROL_CAPABILITIES["fallback_escape_hatches"],
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "role": session.role,
                        "created_at": session.created_at,
                        "page_open": bool(session.page and not session.page.is_closed()),
                        "current_url": session.page.url if session.page and not session.page.is_closed() else None,
                        "issue_counts": _issue_counts(session),
                        "open_human_checkpoints": len(
                            [item for item in session.human_checkpoints if not item.get("resolved")]
                        ),
                    }
                    for session in _sessions.values()
                ],
                "llm_api_keys_required": False,
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "llm_api_keys_required": False})


@mcp.tool()
async def browser_open_url(
    url: str,
    session_id: str = DEFAULT_SESSION_ID,
    role: Optional[str] = None,
    headless: bool = True,
    wait_until: Literal["load", "domcontentloaded", "networkidle"] = "domcontentloaded",
    wait_for_networkidle: bool = False,
    require_body_text: bool = True,
    settle_ms: int = DEFAULT_SETTLE_MS,
    include_snapshot: bool = True,
) -> str:
    """
    Open a URL in Chromium and return a page snapshot.
    """
    try:
        session = await _ensure_session(session_id=session_id, role=role, headless=headless)
        async with session.lock:
            page = session.page
            await page.goto(url, wait_until=wait_until)
            ready = await _wait_for_page_ready(
                page,
                wait_for_networkidle=wait_for_networkidle,
                require_body_text=require_body_text,
                settle_ms=settle_ms,
            )
            return _json(await _action_result(session, ready=ready, include_snapshot=include_snapshot))
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_snapshot(
    session_id: str = DEFAULT_SESSION_ID,
    max_text_chars: int = 8_000,
    max_elements: int = 80,
    wait_before: bool = True,
    require_body_text: bool = False,
    settle_ms: int = DEFAULT_SETTLE_MS,
) -> str:
    """
    Return current URL, title, visible page text, and interactive elements.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            ready = None
            if wait_before:
                ready = await _wait_for_page_ready(
                    page,
                    require_body_text=require_body_text,
                    settle_ms=settle_ms,
                )
            return _json({"ok": True, "session_id": session.session_id, "role": session.role, "ready": ready, "snapshot": await _snapshot(session, max_text_chars, max_elements)})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_compact_state(
    session_id: str = DEFAULT_SESSION_ID,
    text_chars: int = 1_000,
    max_elements: int = 20,
    wait_before: bool = True,
    settle_ms: int = DEFAULT_SETTLE_MS,
) -> str:
    """
    Return a lightweight state summary for follow-up actions without a full DOM snapshot.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            ready = None
            if wait_before:
                ready = await _wait_for_page_ready(page, settle_ms=settle_ms)
            return _json({"ok": True, "session_id": session.session_id, "role": session.role, "ready": ready, "state": await _compact_state(session, text_chars, max_elements)})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_click(
    session_id: str = DEFAULT_SESSION_ID,
    selector: Optional[str] = None,
    element_index: Optional[int] = None,
    text: Optional[str] = None,
    wait_after: bool = True,
    wait_for_networkidle: bool = False,
    settle_ms: int = DEFAULT_SETTLE_MS,
    include_snapshot: bool = False,
    include_compact_state: bool = True,
) -> str:
    """
    Fallback click by selector, element index, or visible text.
    Prefer browser_click_semantic for normal QA flows.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            before_url = page.url
            before_title = await page.title()
            if selector:
                await page.locator(selector).first.click()
            elif element_index is not None:
                locator = page.locator(
                    "a,button,input,textarea,select,[role=button],[role=link],[contenteditable=true]"
                ).nth(element_index)
                await locator.click()
            elif text:
                await page.get_by_text(text, exact=False).first.click()
            else:
                raise ValueError("Provide selector, element_index, or text.")
            ready = None
            if wait_after:
                ready = await _wait_for_page_ready(
                    page,
                    wait_for_networkidle=wait_for_networkidle,
                    settle_ms=settle_ms,
                )
            observation = await _action_observation(session, before_url, before_title)
            return _json(
                await _action_result(
                    session,
                    ready=ready,
                    include_snapshot=include_snapshot,
                    include_compact_state=include_compact_state,
                    observation=observation,
                )
            )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_type(
    session_id: str = DEFAULT_SESSION_ID,
    selector: Optional[str] = None,
    element_index: Optional[int] = None,
    text: str = "",
    clear: bool = True,
    include_snapshot: bool = False,
    include_compact_state: bool = True,
) -> str:
    """
    Fallback type/fill by selector or element index.
    Prefer browser_type_semantic or browser_fill_form for normal QA flows.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            before_url = page.url
            before_title = await page.title()
            locator = page.locator(selector).first if selector else page.locator(
                "input,textarea,[contenteditable=true]"
            ).nth(element_index or 0)
            if clear:
                await locator.fill(text)
            else:
                await locator.type(text)
            observation = await _action_observation(session, before_url, before_title)
            return _json(
                await _action_result(
                    session,
                    include_snapshot=include_snapshot,
                    include_compact_state=include_compact_state,
                    observation=observation,
                )
            )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_key(
    key: str,
    session_id: str = DEFAULT_SESSION_ID,
    wait_after: bool = True,
    wait_for_networkidle: bool = False,
    settle_ms: int = DEFAULT_SETTLE_MS,
    include_snapshot: bool = False,
    include_compact_state: bool = True,
) -> str:
    """
    Press a keyboard key, for example Enter, Escape, Tab, Control+A.
    Use for keyboard-only actions, submits, and shortcuts.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            before_url = page.url
            before_title = await page.title()
            await page.keyboard.press(key)
            ready = None
            if wait_after:
                ready = await _wait_for_page_ready(
                    page,
                    wait_for_networkidle=wait_for_networkidle,
                    settle_ms=settle_ms,
                )
            observation = await _action_observation(session, before_url, before_title)
            return _json(
                await _action_result(
                    session,
                    ready=ready,
                    include_snapshot=include_snapshot,
                    include_compact_state=include_compact_state,
                    observation=observation,
                )
            )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_click_semantic(
    session_id: str = DEFAULT_SESSION_ID,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    placeholder: Optional[str] = None,
    name: Optional[str] = None,
    text: Optional[str] = None,
    role: Optional[str] = None,
    exact: bool = False,
    wait_after: bool = True,
    wait_for_networkidle: bool = False,
    settle_ms: int = DEFAULT_SETTLE_MS,
    include_snapshot: bool = False,
    include_compact_state: bool = True,
) -> str:
    """
    Click an element using semantic locators before falling back to CSS.
    Prefer this over element indexes for dynamic UIs.
    When a Radix/shadcn/MUI overlay intercepts the click, automatically retries
    via JavaScript el.click() to bypass pointer-events interception.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            before_url = page.url
            before_title = await page.title()
            locator = await _resolve_locator(
                page,
                selector=selector,
                label=label,
                placeholder=placeholder,
                name=name,
                text=text,
                role=role,
                exact=exact,
            )
            try:
                await locator.click()
            except Exception as click_exc:
                # Fallback: JS el.click() bypasses pointer-events overlays (Radix, shadcn, MUI, Headless UI).
                # This is safe — el.click() dispatches a real click event on the element directly.
                try:
                    await locator.evaluate("el => el.click()")
                except Exception:
                    raise click_exc
            ready = None
            if wait_after:
                ready = await _wait_for_page_ready(
                    page,
                    wait_for_networkidle=wait_for_networkidle,
                    settle_ms=settle_ms,
                )
            return _json(
                await _action_result(
                    session,
                    ready=ready,
                    include_snapshot=include_snapshot,
                    include_compact_state=include_compact_state,
                    observation=await _action_observation(session, before_url, before_title),
                )
            )
    except Exception as exc:
        overlay_hint = (
            " If a modal overlay is intercepting clicks, use browser_evaluate with an IIFE "
            "to find the element inside [role='dialog'] or [role='alertdialog'] and call .click() directly."
        )
        return _json({"ok": False, "error": str(exc) + overlay_hint, "session_id": session_id})


@mcp.tool()
async def browser_type_semantic(
    value: str,
    session_id: str = DEFAULT_SESSION_ID,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    placeholder: Optional[str] = None,
    name: Optional[str] = None,
    role: Optional[str] = None,
    exact: bool = False,
    clear: bool = True,
    press_after: Optional[str] = None,
    include_snapshot: bool = False,
    include_compact_state: bool = True,
) -> str:
    """
    Fill/type into a field using label, placeholder, name, role+name, or CSS selector.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            before_url = page.url
            before_title = await page.title()
            locator = await _resolve_locator(
                page,
                selector=selector,
                label=label,
                placeholder=placeholder,
                name=name,
                role=role,
                exact=exact,
            )
            if clear:
                await locator.fill(value)
            else:
                await locator.type(value)
            if press_after:
                await locator.press(press_after)
            return _json(
                await _action_result(
                    session,
                    include_snapshot=include_snapshot,
                    include_compact_state=include_compact_state,
                    observation=await _action_observation(session, before_url, before_title),
                )
            )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_fill_form(
    fields: list[dict[str, Any]],
    session_id: str = DEFAULT_SESSION_ID,
    submit: bool = False,
    submit_selector: Optional[str] = None,
    submit_text: Optional[str] = None,
    settle_ms: int = DEFAULT_SETTLE_MS,
    include_snapshot: bool = False,
) -> str:
    """
    Fill multiple form fields sequentially. Do not parallelize field entry on one page.

    Each field supports: value, selector, label, placeholder, name, role, exact,
    clear, press_after. Example:
    {"label": "Email", "value": "qa@example.com"}
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            before_url = page.url
            before_title = await page.title()
            results = []
            for index, field in enumerate(fields):
                value = str(field.get("value", ""))
                locator = await _resolve_locator(
                    page,
                    selector=field.get("selector"),
                    label=field.get("label"),
                    placeholder=field.get("placeholder"),
                    name=field.get("name"),
                    role=field.get("role"),
                    text=field.get("text"),
                    exact=bool(field.get("exact", False)),
                )
                if bool(field.get("clear", True)):
                    await locator.fill(value)
                else:
                    await locator.type(value)
                if field.get("press_after"):
                    await locator.press(str(field["press_after"]))
                results.append(
                    {
                        "index": index,
                        "target": {
                            key: field.get(key)
                            for key in ("selector", "label", "placeholder", "name", "role", "text")
                            if field.get(key)
                        },
                        "filled": True,
                    }
                )
            submitted = False
            if submit:
                if submit_selector or submit_text:
                    submit_locator = await _resolve_locator(page, selector=submit_selector, text=submit_text)
                    await submit_locator.click()
                else:
                    await page.keyboard.press("Enter")
                submitted = True
            ready = await _wait_for_page_ready(page, settle_ms=settle_ms)
            return _json(
                {
                    **await _action_result(
                        session,
                        ready=ready,
                        include_snapshot=include_snapshot,
                        observation=await _action_observation(session, before_url, before_title),
                    ),
                    "fields": results,
                    "submitted": submitted,
                }
            )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_select_option_semantic(
    value: str,
    session_id: str = DEFAULT_SESSION_ID,
    selector: Optional[str] = None,
    label: Optional[str] = None,
    placeholder: Optional[str] = None,
    name: Optional[str] = None,
    option_text: Optional[str] = None,
    exact: bool = False,
    settle_ms: int = DEFAULT_SETTLE_MS,
    include_snapshot: bool = False,
) -> str:
    """
    Select a native select option or autocomplete/combobox option as one operation.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            before_url = page.url
            before_title = await page.title()
            locator = await _resolve_locator(
                page,
                selector=selector,
                label=label,
                placeholder=placeholder,
                name=name,
                exact=exact,
            )
            tag_name = await locator.evaluate("el => el.tagName.toLowerCase()")
            selected_by = "native-select"
            if tag_name == "select":
                try:
                    await locator.select_option(label=option_text or value)
                except Exception:
                    await locator.select_option(value=value)
            else:
                await locator.fill(value)
                await page.wait_for_timeout(settle_ms)
                option = option_text or value
                try:
                    await page.get_by_role("option", name=option, exact=exact).first.click(timeout=3_000)
                    selected_by = "role-option"
                except Exception:
                    await page.get_by_text(option, exact=exact).first.click(timeout=3_000)
                    selected_by = "text-option"
            ready = await _wait_for_page_ready(page, settle_ms=settle_ms)
            return _json(
                {
                    **await _action_result(
                        session,
                        ready=ready,
                        include_snapshot=include_snapshot,
                        observation=await _action_observation(session, before_url, before_title),
                    ),
                    "selected_by": selected_by,
                    "value": value,
                    "option_text": option_text,
                }
            )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_screenshot(
    session_id: str = DEFAULT_SESSION_ID,
    full_page: bool = True,
    return_base64: bool = False,
    wait_before: bool = True,
    wait_for_networkidle: bool = False,
    require_body_text: bool = True,
    settle_ms: int = 750,
) -> str:
    """
    Capture a screenshot. Returns a local file path by default.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            ready = None
            if wait_before:
                ready = await _wait_for_page_ready(
                    page,
                    wait_for_networkidle=wait_for_networkidle,
                    require_body_text=require_body_text,
                    settle_ms=settle_ms,
                )
            fd, raw_path = tempfile.mkstemp(prefix="browser_control_", suffix=".png")
            os.close(fd)
            path = Path(raw_path)
            await page.screenshot(path=str(path), full_page=full_page)
            metadata = await _screenshot_metadata(page, path, full_page)
            result: dict[str, Any] = {"ok": True, "ready": ready, **metadata}
            if return_base64:
                result["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            return _json(result)
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_visual_checkpoint(
    session_id: str = DEFAULT_SESSION_ID,
    label: str = "visual-checkpoint",
    full_page: bool = True,
    wait_for_networkidle: bool = False,
    require_body_text: bool = True,
    settle_ms: int = 1_000,
) -> str:
    """
    Capture a stabilized screenshot specifically for human-style visual QA.

    Use this before making claims about layout, overflow, clipped text, colors,
    spacing, or whether the UI looks correct. The returned path should be opened
    with view_image by the calling agent before visual conclusions.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            ready = await _wait_for_page_ready(
                page,
                wait_for_networkidle=wait_for_networkidle,
                require_body_text=require_body_text,
                settle_ms=settle_ms,
            )
            safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)[:60]
            fd, raw_path = tempfile.mkstemp(prefix=f"browser_control_{safe_label}_", suffix=".png")
            os.close(fd)
            path = Path(raw_path)
            await page.screenshot(path=str(path), full_page=full_page)
            metadata = await _screenshot_metadata(page, path, full_page)
            return _json(
                {
                    "ok": True,
                    "label": label,
                    "ready": ready,
                    **metadata,
                    "visual_qa_instruction": "Open path with view_image now; do not rely on DOM snapshot for layout/overflow/visual-quality claims.",
                    "state": await _compact_state(session, text_chars=600, max_elements=12),
                }
            )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_wait_for_idle(
    session_id: str = DEFAULT_SESSION_ID,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    wait_for_networkidle: bool = True,
    require_body_text: bool = False,
    settle_ms: int = DEFAULT_SETTLE_MS,
) -> str:
    """
    Wait until the current page is stable enough for QA inspection.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            ready = await _wait_for_page_ready(
                page,
                timeout_ms=timeout_ms,
                wait_for_networkidle=wait_for_networkidle,
                require_body_text=require_body_text,
                settle_ms=settle_ms,
            )
            return _json({"ok": True, "session_id": session.session_id, "role": session.role, "ready": ready})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_wait_for_text(text: str, session_id: str = DEFAULT_SESSION_ID, exact: bool = False, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    """
    Wait until text appears on the current page.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            await page.get_by_text(text, exact=exact).first.wait_for(timeout=timeout_ms)
            return _json({"ok": True, "session_id": session.session_id, "role": session.role, "url": page.url, "text": text, "timestamp": _now_iso()})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "text": text, "session_id": session_id})


@mcp.tool()
async def browser_wait_for_selector(selector: str, session_id: str = DEFAULT_SESSION_ID, state: Literal["attached", "detached", "visible", "hidden"] = "visible", timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    """
    Wait until a CSS selector reaches the requested state.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            await page.locator(selector).first.wait_for(state=state, timeout=timeout_ms)
            return _json({"ok": True, "session_id": session.session_id, "role": session.role, "url": page.url, "selector": selector, "state": state, "timestamp": _now_iso()})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "selector": selector, "state": state, "session_id": session_id})


@mcp.tool()
async def browser_wait_for_url(url_pattern: str, session_id: str = DEFAULT_SESSION_ID, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    """
    Wait until the current page URL matches a Playwright URL pattern.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            await page.wait_for_url(url_pattern, timeout=timeout_ms)
            return _json({"ok": True, "session_id": session.session_id, "role": session.role, "url": page.url, "pattern": url_pattern, "timestamp": _now_iso()})
    except Exception as exc:
        current_url = None
        session = _sessions.get(session_id)
        if session and session.page:
            current_url = session.page.url
        return _json({"ok": False, "error": str(exc), "pattern": url_pattern, "current_url": current_url, "session_id": session_id})


@mcp.tool()
async def browser_evaluate(script: str, session_id: str = DEFAULT_SESSION_ID) -> str:
    """
    Escape hatch: evaluate JavaScript in the current page.
    Prefer semantic tools first; use this for app-specific inspection or hard cases.
    Scripts with top-level return statements are automatically wrapped in an IIFE.
    Always write scripts as IIFE: (function() { ... })() to avoid SyntaxErrors.
    """
    try:
        session = await _current_session(session_id)
        async with session.lock:
            page = session.page
            try:
                result = await page.evaluate(script)
            except Exception as eval_exc:
                # Auto-wrap bare return statements in an IIFE and retry once
                if "Illegal return statement" in str(eval_exc) or (
                    "SyntaxError" in str(eval_exc) and "return" in script
                ):
                    result = await page.evaluate(f"(function() {{\n{script}\n}})()")
                else:
                    raise
            return _json({"ok": True, "session_id": session.session_id, "role": session.role, "result": result})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_get_console_logs(
    session_id: str = DEFAULT_SESSION_ID,
    level: Optional[Literal["log", "debug", "info", "warning", "error"]] = None,
    contains: Optional[str] = None,
    limit: int = 100,
) -> str:
    """
    Return captured browser console messages.
    """
    try:
        session = await _current_session(session_id)
        return _json(
            {
                "ok": True,
                "session_id": session.session_id,
                "role": session.role,
                "count": len(session.console_logs),
                "returned": _filter_events(session.console_logs, limit=limit, level=level, contains=contains),
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_get_page_errors(session_id: str = DEFAULT_SESSION_ID, limit: int = 50, contains: Optional[str] = None) -> str:
    """
    Return uncaught JavaScript errors from the page.
    """
    try:
        session = await _current_session(session_id)
        return _json(
            {
                "ok": True,
                "session_id": session.session_id,
                "role": session.role,
                "count": len(session.page_errors),
                "returned": _filter_events(session.page_errors, limit=limit, contains=contains),
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_get_network_errors(session_id: str = DEFAULT_SESSION_ID, limit: int = 100, contains: Optional[str] = None) -> str:
    """
    Return failed network requests, such as DNS, blocked, timeout, or connection errors.
    """
    try:
        session = await _current_session(session_id)
        return _json(
            {
                "ok": True,
                "session_id": session.session_id,
                "role": session.role,
                "count": len(session.network_errors),
                "returned": _filter_events(session.network_errors, limit=limit, contains=contains),
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_get_http_errors(session_id: str = DEFAULT_SESSION_ID, status_min: int = 400, limit: int = 100, contains: Optional[str] = None) -> str:
    """
    Return HTTP responses with status >= status_min.
    """
    try:
        session = await _current_session(session_id)
        events = [event for event in session.http_errors if int(event.get("status", 0)) >= status_min]
        return _json(
            {
                "ok": True,
                "session_id": session.session_id,
                "role": session.role,
                "count": len(events),
                "returned": _filter_events(events, limit=limit, contains=contains),
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_get_issue_summary(session_id: str = DEFAULT_SESSION_ID, limit: int = 50) -> str:
    """
    Return a compact QA-oriented summary of captured console/page/network/HTTP issues.
    """
    try:
        session = await _current_session(session_id)
        console_errors = [
            event for event in session.console_logs if event.get("type") in {"error", "warning"}
        ]
        return _json(
            {
                "ok": True,
                "session_id": session.session_id,
                "role": session.role,
                "counts": {
                    "console_warnings_or_errors": len(console_errors),
                    "page_errors": len(session.page_errors),
                    "network_errors": len(session.network_errors),
                    "http_errors": len(session.http_errors),
                },
                "recent_console_warnings_or_errors": console_errors[-limit:],
                "recent_page_errors": session.page_errors[-limit:],
                "recent_network_errors": session.network_errors[-limit:],
                "recent_http_errors": session.http_errors[-limit:],
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_clear_logs(session_id: str = DEFAULT_SESSION_ID) -> str:
    """
    Clear captured browser console/page/network/HTTP events.
    """
    try:
        session = await _current_session(session_id)
        session.console_logs.clear()
        session.page_errors.clear()
        session.network_errors.clear()
        session.http_errors.clear()
        return _json({"ok": True, "session_id": session.session_id, "role": session.role})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_request_human_input(
    question: str,
    session_id: str = DEFAULT_SESSION_ID,
    reason: str = "human input required",
    category: Literal["credentials", "mfa", "captcha", "business-data", "workflow-intent", "destructive-action", "ambiguous-field", "other"] = "other",
    blocking: bool = True,
    context: Optional[str] = None,
) -> str:
    """
    Record a human-in-the-loop checkpoint for the current browser QA flow.

    This tool does not ask the human by itself. It returns needs_human=true so
    the calling agent can pause and ask the user the returned question.
    """
    try:
        session = await _current_session(session_id)
        checkpoint_id = f"{session.session_id}-{len(session.human_checkpoints) + 1}"
        checkpoint = {
            "id": checkpoint_id,
            "session_id": session.session_id,
            "role": session.role,
            "category": category,
            "reason": reason,
            "question": question,
            "context": context,
            "blocking": blocking,
            "resolved": False,
            "created_at": _now_iso(),
            "url": session.page.url if session.page and not session.page.is_closed() else None,
        }
        session.human_checkpoints.append(checkpoint)
        return _json(
            {
                "ok": True,
                "needs_human": True,
                "checkpoint": checkpoint,
                "instruction": "Pause the browser flow and ask the human this question before continuing if blocking=true.",
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id, "needs_human": True, "question": question})


@mcp.tool()
async def browser_get_human_checkpoints(
    session_id: Optional[str] = None,
    include_resolved: bool = False,
) -> str:
    """
    List human-in-the-loop checkpoints, optionally across all sessions.
    """
    try:
        sessions = [_sessions[session_id]] if session_id else list(_sessions.values())
        checkpoints = []
        for session in sessions:
            for checkpoint in session.human_checkpoints:
                if include_resolved or not checkpoint.get("resolved"):
                    checkpoints.append(checkpoint)
        return _json({"ok": True, "count": len(checkpoints), "checkpoints": checkpoints})
    except KeyError:
        return _json({"ok": False, "error": f"No session named '{session_id}'", "session_id": session_id})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_resolve_human_checkpoint(
    checkpoint_id: str,
    answer_summary: str,
    session_id: Optional[str] = None,
) -> str:
    """
    Mark a human-in-the-loop checkpoint as resolved after the human answers.
    """
    try:
        sessions = [_sessions[session_id]] if session_id else list(_sessions.values())
        for session in sessions:
            for checkpoint in session.human_checkpoints:
                if checkpoint.get("id") == checkpoint_id:
                    checkpoint["resolved"] = True
                    checkpoint["resolved_at"] = _now_iso()
                    checkpoint["answer_summary"] = answer_summary
                    return _json({"ok": True, "checkpoint": checkpoint})
        return _json({"ok": False, "error": f"No checkpoint named '{checkpoint_id}'", "checkpoint_id": checkpoint_id})
    except KeyError:
        return _json({"ok": False, "error": f"No session named '{session_id}'", "session_id": session_id})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "checkpoint_id": checkpoint_id})


@mcp.tool()
async def browser_close_session(session_id: str = DEFAULT_SESSION_ID) -> str:
    """
    Close one isolated browser session.
    """
    try:
        session = _sessions.pop(session_id, None)
        if not session:
            return _json({"ok": True, "session_id": session_id, "already_closed": True})
        async with session.lock:
            if session.context:
                await session.context.close()
        return _json({"ok": True, "session_id": session_id, "role": session.role})
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "session_id": session_id})


@mcp.tool()
async def browser_close_all() -> str:
    """
    Close all browser sessions and the shared Chromium process.
    """
    global _playwright, _browser, _browser_headless
    async with _global_lock:
        try:
            closed = []
            for session_id, session in list(_sessions.items()):
                async with session.lock:
                    if session.context:
                        await session.context.close()
                    closed.append({"session_id": session_id, "role": session.role})
            _sessions.clear()
            if _browser:
                await _browser.close()
            if _playwright:
                await _playwright.stop()
            _playwright = None
            _browser = None
            _browser_headless = None
            return _json({"ok": True, "closed": closed})
        except Exception as exc:
            return _json({"ok": False, "error": str(exc)})


@mcp.tool()
async def browser_close(session_id: str = DEFAULT_SESSION_ID) -> str:
    """
    Backward-compatible alias for browser_close_session.
    Prefer browser_close_session in new flows.
    """
    return await browser_close_session(session_id=session_id)


if __name__ == "__main__":
    mcp.run()
