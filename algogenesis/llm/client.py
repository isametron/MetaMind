"""Thin LLM client for the LLaMEA experiments.

    from algogenesis.llm.client import ask, extract

    text, usage = ask("cerebras", "gpt-oss-120b", [{"role": "user", "content": "..."}])
    name, code = extract(text)          # raises ExtractError (error_type "syntax") if no python block

Providers speak the OpenAI chat-completions API, so one SDK covers all of them.
Keys come from the environment (CEREBRAS_API_KEY, GROQ_API_KEY). The "fake"
provider needs no key: it ignores the prompt and returns the PSO baseline, so
the whole pipeline can be tested offline.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

TEMPERATURE = 0.8
REASONING_EFFORT = "low"

PROVIDERS = {
    "cerebras": {"base_url": "https://api.cerebras.ai/v1", "key_env": "CEREBRAS_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
}

# Retry policy. A 429 sleeps for the server's retry-after; if that is longer than
# MAX_WAIT_S we are almost certainly out of daily quota, so give up and let the
# caller stop cleanly (the next scheduled run resumes from the log).
MAX_RETRIES = 8
MAX_WAIT_S = 900.0
DEFAULT_WAIT_S = 10.0

_PSO_PATH = Path(__file__).resolve().parents[1] / "baselines" / "pso.py"
_clients: dict = {}
_no_reasoning_effort: set = set()  # (provider, model) pairs that rejected reasoning_effort


class LLMError(Exception):
    """The API call failed in a way the caller should stop on (not a bad program)."""


class RateLimitExhausted(LLMError):
    """Out of retries, or the server asked us to wait longer than MAX_WAIT_S."""


class ExtractError(ValueError):
    """The reply contained no usable python block."""

    error_type = "syntax"


# --------------------------------------------------------------------------- ask

def ask(provider: str, model: str, messages: list[dict]) -> tuple[str, dict]:
    """Send ``messages`` and return ``(text, usage)``.

    usage = {prompt_tokens, completion_tokens, reasoning_tokens, latency_s,
             system_fingerprint, reasoning_effort_dropped}
    reasoning_tokens is None when the provider doesn't report it.
    """
    if provider == "fake":
        return _fake(messages)
    client = _get_client(provider)
    from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError

    kwargs = dict(model=model, messages=messages, temperature=TEMPERATURE)
    dropped = False
    for attempt in range(MAX_RETRIES + 1):
        use_effort = (provider, model) not in _no_reasoning_effort
        extra = {"extra_body": {"reasoning_effort": REASONING_EFFORT}} if use_effort else {}
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs, **extra)
            break
        except RateLimitError as e:
            wait = _retry_after(e)
            if wait > MAX_WAIT_S or attempt == MAX_RETRIES:
                raise RateLimitExhausted(f"429 from {provider}/{model}; retry-after {wait:.0f}s "
                                         f"(attempt {attempt + 1}). Daily quota probably spent.") from e
            print(f"[client] 429 from {provider}/{model}: sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
        except BadRequestError as e:
            if use_effort and "reasoning" in str(e).lower():
                print(f"[client] {provider}/{model} rejected reasoning_effort; retrying without it", flush=True)
                _no_reasoning_effort.add((provider, model))
                dropped = True
                continue
            raise LLMError(f"400 from {provider}/{model}: {e}") from e
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt == MAX_RETRIES:
                raise LLMError(f"{type(e).__name__} from {provider}/{model} after {attempt + 1} tries: {e}") from e
            wait = min(5.0 * 2 ** attempt, 120.0)
            print(f"[client] {type(e).__name__}: sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
    latency = time.perf_counter() - t0

    text = resp.choices[0].message.content or ""
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
        "latency_s": latency,
        "system_fingerprint": getattr(resp, "system_fingerprint", None),
        "reasoning_effort_dropped": dropped or (provider, model) in _no_reasoning_effort,
    }
    return text, usage


def _get_client(provider: str):
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; choose from {list(PROVIDERS) + ['fake']}")
    if provider not in _clients:
        from openai import OpenAI

        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["key_env"])
        if not key:
            raise LLMError(f"environment variable {cfg['key_env']} is not set")
        # max_retries=0: we do our own retrying so every 429 is visible and honoured.
        _clients[provider] = OpenAI(api_key=key, base_url=cfg["base_url"], max_retries=0, timeout=300.0)
    return _clients[provider]


def _retry_after(exc) -> float:
    """Seconds the server asked us to wait (retry-after header), with a small pad."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-requests-day", "x-ratelimit-reset-tokens-minute"):
        raw = headers.get(key)
        if raw:
            try:
                return float(raw) + 1.0
            except ValueError:
                pass
    return DEFAULT_WAIT_S


def _fake(messages: list[dict]) -> tuple[str, dict]:
    """Offline stand-in: ignores the prompt, returns the PSO baseline."""
    code = _PSO_PATH.read_text(encoding="utf-8")
    text = f"# Name: PSOBaseline\n```python\n{code}\n```\n"
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    usage = {
        "prompt_tokens": prompt_chars // 4,     # rough estimate so the token ledger has something to count
        "completion_tokens": len(text) // 4,
        "reasoning_tokens": 0,
        "latency_s": 0.0,
        "system_fingerprint": "fake",
        "reasoning_effort_dropped": False,
    }
    return text, usage


# ----------------------------------------------------------------------- extract

_NAME_RE = re.compile(r"^[ \t]*#[ \t]*Name:[ \t]*(.+?)[ \t]*$", re.MULTILINE | re.IGNORECASE)
_BLOCK_RE = re.compile(r"```[ \t]*(?:python|py)[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract(text: str) -> tuple[str, str]:
    """Return ``(name, code)``: the ``# Name:`` line and the first python block.

    No complete python block (including a block cut off by the token limit)
    raises ExtractError, whose error_type is "syntax". A missing name line is
    not an error; the name becomes "unnamed".
    """
    block = _BLOCK_RE.search(text or "")
    if not block or not block.group(1).strip():
        raise ExtractError("no ```python block found in the reply")
    code = block.group(1).rstrip() + "\n"
    m = _NAME_RE.search(text)
    name = m.group(1).strip() if m else "unnamed"
    return name, code
