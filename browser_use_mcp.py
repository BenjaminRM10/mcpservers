import asyncio
import json
import os
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


load_dotenv()

mcp = FastMCP("browser-use")


ModelProvider = Literal["browser-use", "openai", "google", "anthropic"]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _env_status() -> dict[str, bool]:
    return {
        "BROWSER_USE_API_KEY": bool(os.getenv("BROWSER_USE_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


def _build_llm(provider: ModelProvider, model: Optional[str]) -> Any:
    if provider == "browser-use":
        from browser_use import ChatBrowserUse

        return ChatBrowserUse(model=model) if model else ChatBrowserUse()

    if provider == "openai":
        from browser_use import ChatOpenAI

        return ChatOpenAI(model=model or "gpt-4.1-mini")

    if provider == "google":
        from browser_use import ChatGoogle

        return ChatGoogle(model=model or "gemini-flash-latest")

    if provider == "anthropic":
        from browser_use import ChatAnthropic

        return ChatAnthropic(model=model or "claude-sonnet-4-0", temperature=0.0)

    raise ValueError(f"Proveedor no soportado: {provider}")


def _history_summary(history: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"final_result": None}

    for name in (
        "final_result",
        "urls",
        "screenshot_paths",
        "action_names",
        "extracted_content",
        "errors",
        "model_actions",
    ):
        attr = getattr(history, name, None)
        if callable(attr):
            try:
                summary[name] = attr()
            except Exception as exc:
                summary[name] = f"Failed to read {name}: {exc}"

    return summary


async def _run_agent(
    task: str,
    provider: ModelProvider,
    model: Optional[str],
    max_steps: int,
    use_vision: bool | Literal["auto"],
    generate_gif: bool,
) -> dict[str, Any]:
    from browser_use import Agent

    llm = _build_llm(provider, model)
    agent = Agent(
        task=task,
        llm=llm,
        use_vision=use_vision,
        generate_gif=generate_gif,
    )
    history = await agent.run(max_steps=max_steps)
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "max_steps": max_steps,
        "history": _history_summary(history),
    }


@mcp.tool()
async def browser_use_status() -> str:
    """
    Check whether browser-use is importable and which supported API keys are present.
    """
    try:
        import browser_use

        return _json(
            {
                "ok": True,
                "module": str(browser_use),
                "api_keys_present": _env_status(),
                "note": "Use browser-use provider with BROWSER_USE_API_KEY, or openai/google/anthropic with the matching API key.",
            }
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "api_keys_present": _env_status()})


@mcp.tool()
async def browser_use_run(
    task: str,
    provider: ModelProvider = "browser-use",
    model: Optional[str] = None,
    max_steps: int = 30,
    use_vision: bool | Literal["auto"] = "auto",
    generate_gif: bool = False,
) -> str:
    """
    Run an autonomous browser QA/research task with browser-use.

    Args:
        task: Plain-language browser task, including the target URL and assertions.
        provider: LLM provider. Prefer "browser-use" when BROWSER_USE_API_KEY is set.
        model: Optional model name. Leave empty to use browser-use defaults.
        max_steps: Safety cap for browser actions.
        use_vision: "auto", true, or false.
        generate_gif: Whether browser-use should generate a GIF of the run.
    """
    if max_steps < 1 or max_steps > 200:
        return _json({"ok": False, "error": "max_steps debe estar entre 1 y 200."})

    try:
        result = await _run_agent(
            task=task,
            provider=provider,
            model=model,
            max_steps=max_steps,
            use_vision=use_vision,
            generate_gif=generate_gif,
        )
        return _json(result)
    except Exception as exc:
        return _json(
            {
                "ok": False,
                "error": str(exc),
                "provider": provider,
                "api_keys_present": _env_status(),
            }
        )


@mcp.tool()
async def browser_use_smoke_test(
    url: str,
    expectation: str,
    provider: ModelProvider = "browser-use",
    model: Optional[str] = None,
    max_steps: int = 20,
) -> str:
    """
    Run a focused QA smoke test against a URL and expectation.

    Args:
        url: Target URL to open.
        expectation: Behavior or content that should be verified.
        provider: LLM provider. Prefer "browser-use" when BROWSER_USE_API_KEY is set.
        model: Optional model name.
        max_steps: Safety cap for browser actions.
    """
    task = (
        f"Open {url}. Perform a QA smoke test. Verify this expectation: "
        f"{expectation}. Report pass/fail with concise evidence."
    )
    return await browser_use_run(
        task=task,
        provider=provider,
        model=model,
        max_steps=max_steps,
        use_vision="auto",
        generate_gif=False,
    )


if __name__ == "__main__":
    mcp.run()
