"""Tests for the LLM client and the experiment runner (no network, no keys)."""
import importlib.util
import json
import sys
from pathlib import Path

import types

import openai
import pytest

from algogenesis.llm import client
from algogenesis.llm.client import ExtractError, extract

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_llm", ROOT / "scripts" / "run_llm.py")
run_llm = importlib.util.module_from_spec(_spec)
sys.modules["run_llm"] = run_llm
_spec.loader.exec_module(run_llm)


# ---------------------------------------------------------------- extract()

def test_extract_name_and_first_block():
    text = "Here you go\n# Name: Foo\n```python\nx = 1\n```\nand also\n```python\ny = 2\n```"
    assert extract(text) == ("Foo", "x = 1\n")


def test_extract_accepts_py_fence_and_name_inside_block():
    assert extract("```py\n# Name: Bar\nx = 1\n```") == ("Bar", "# Name: Bar\nx = 1\n")


def test_extract_without_name_is_not_an_error():
    assert extract("```python\nx = 1\n```")[0] == "unnamed"


@pytest.mark.parametrize("text", ["no code here", "```python\nx = 1\n", "```text\nx\n```", "", None])
def test_extract_without_block_is_a_syntax_error(text):
    with pytest.raises(ExtractError) as e:
        extract(text)
    assert e.value.error_type == "syntax"


# ------------------------------------------------------------ classify_error()

@pytest.mark.parametrize("text,expected", [
    ("evaluation timed out after 300 s", "timeout"),
    ("Traceback...\nValueError: policy violation:\nline 1: import of 'os' is not allowed", "policy"),
    ("policy violation: the evaluation record was tampered with", "policy"),
    ("Traceback...\n  File \"<candidate>\", line 3\nSyntaxError: invalid syntax", "syntax"),
    ("Traceback...\nIndentationError: unexpected indent", "syntax"),
    ("ValueError: the code must define a function optimize(func, dim, bounds, budget)", "interface"),
    ("ValueError: optimize must accept (func, dim, bounds, budget); its signature is (a)", "interface"),
    ("TypeError: optimize must return (best_x, best_f, history), got NoneType", "interface"),
    ("Traceback...\nZeroDivisionError: division by zero", "runtime"),
    ("evaluation process crashed (exit code -9)", "runtime"),
])
def test_classify_error(text, expected):
    assert run_llm.classify_error(text) == expected


# -------------------------------------- bad programs end to end (real scorer)

BAD = {
    "syntax": "def optimize(func, dim, bounds, budget:\n    pass",
    "policy": "import os\ndef optimize(func, dim, bounds, budget):\n    return None, 0, []",
    "interface": "def solve(func, dim, bounds, budget):\n    return None, 0, []",
    "runtime": "def optimize(func, dim, bounds, budget):\n    return 1 / 0",
}


@pytest.mark.parametrize("kind", list(BAD))
def test_bad_programs_are_classified(kind):
    out = run_llm.score_code(BAD[kind])
    assert out["error_type"] == kind
    assert out["fitness"] == 0.0 and out["error"]


def test_run_logs_bad_replies_and_missing_block(tmp_path, monkeypatch):
    replies = iter(["no code at all", f"# Name: X\n```python\n{BAD['interface']}\n```"])
    monkeypatch.setattr(run_llm, "ask", lambda p, m, msgs: (next(replies), client._fake(msgs)[1]))
    run_llm.run("fake", "m", "B", 1, calls=2, data_dir=tmp_path)
    rows = run_llm.read_log(run_llm.log_path(tmp_path, "fake", "m", "B", 1))
    assert [r["error_type"] for r in rows] == ["syntax", "interface"]
    assert rows[0]["code"] is None and rows[0]["name"] is None


# --------------------------------------------------- arm logic (scorer faked)

def _fake_scorer(fitnesses):
    it = iter(fitnesses)

    def score(code):
        f = next(it)
        return {"fitness": f, "fitness_std": 0.1, "error": None, "error_type": None,
                "summary": [], "flags": [], "eval_s": 0.0}
    return score


def test_arm_a_child_replaces_parent_when_fitness_is_greater_or_equal(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.3, 0.5, 0.4, 0.5]))
    run_llm.run("fake", "m", "A", 1, calls=4, data_dir=tmp_path)
    rows = run_llm.read_log(run_llm.log_path(tmp_path, "fake", "m", "A", 1))
    assert [r["accepted"] for r in rows] == [True, True, False, True]   # 0.5 ties and replaces
    assert "0.5000" in rows[3]["prompt"]                                # call 4 prompt shows the 0.5 parent


def test_arm_b_keeps_best_and_every_prompt_is_the_task(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.3, 0.5, 0.4, 0.5]))
    run_llm.run("fake", "m", "B", 1, calls=4, data_dir=tmp_path)
    rows = run_llm.read_log(run_llm.log_path(tmp_path, "fake", "m", "B", 1))
    assert [r["accepted"] for r in rows] == [True, True, False, False]  # strictly better only
    assert {r["prompt"] for r in rows} == {run_llm.TASK_PROMPT}
    assert "training fitness 0.5000" in run_llm.found_path(tmp_path, "fake", "m", "B", 1).read_text()


def test_resume_continues_and_does_not_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.1, 0.2, 0.3, 0.4, 0.5]))
    run_llm.run("fake", "m", "A", 1, calls=2, data_dir=tmp_path)
    run_llm.run("fake", "m", "A", 1, calls=5, data_dir=tmp_path)   # restart with more calls
    run_llm.run("fake", "m", "A", 1, calls=5, data_dir=tmp_path)   # restart when finished: no-op
    rows = run_llm.read_log(run_llm.log_path(tmp_path, "fake", "m", "A", 1))
    assert [r["call"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["fitness"] for r in rows] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_resume_drops_a_half_written_line(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.1, 0.2, 0.3]))
    run_llm.run("fake", "m", "A", 1, calls=2, data_dir=tmp_path)
    path = run_llm.log_path(tmp_path, "fake", "m", "A", 1)
    with open(path, "ab") as f:
        f.write(b'{"call": 3, "ar')
    run_llm.run("fake", "m", "A", 1, calls=3, data_dir=tmp_path)
    assert [r["call"] for r in run_llm.read_log(path)] == [1, 2, 3]


def test_refuses_to_resume_from_a_different_model(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.1]))
    run_llm.run("fake", "m", "A", 1, calls=1, data_dir=tmp_path)
    path = run_llm.log_path(tmp_path, "fake", "m", "A", 1)
    rows = run_llm.read_log(path)
    rows[0]["model"] = "other"
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(RuntimeError, match="refusing to mix"):
        run_llm.run("fake", "m", "A", 1, calls=2, data_dir=tmp_path)


# ------------------------------------------------------------- token ledger

def test_daily_cap_stops_with_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.1] * 50))
    # arm-B calls all cost the same (it depends on the prompt length); cap 2000 -> stop once
    # used + per_call > 1900
    status = run_llm.run("fake", "m", "B", 1, calls=50, data_dir=tmp_path, daily_cap=2000)
    rows = run_llm.read_log(run_llm.log_path(tmp_path, "fake", "m", "B", 1))
    per_call = rows[0]["tokens"]["prompt"] + rows[0]["tokens"]["completion"]
    used = sum(r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rows)
    assert status == "cap" and used <= 0.95 * 2000 < used + per_call
    # a second invocation the same UTC day sees the ledger and stops immediately
    assert run_llm.run("fake", "m", "B", 1, calls=50, data_dir=tmp_path, daily_cap=2000) == "cap"
    assert len(run_llm.read_log(run_llm.log_path(tmp_path, "fake", "m", "B", 1))) == len(rows)


def test_ledger_ignores_yesterday(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.1] * 5))
    run_llm.run("fake", "m", "B", 1, calls=2, data_dir=tmp_path)
    path = run_llm.log_path(tmp_path, "fake", "m", "B", 1)
    rows = run_llm.read_log(path)
    for r in rows:
        r["time"] = "2020-01-01T00:00:00+00:00"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert run_llm.tokens_today(tmp_path, "fake", "m")[0] == 0


def test_time_limit_stops_before_the_next_call(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "score_code", _fake_scorer([0.1] * 5))
    assert run_llm.run("fake", "m", "B", 1, calls=5, deadline=0.0, data_dir=tmp_path) == "time"
    assert not run_llm.log_path(tmp_path, "fake", "m", "B", 1).exists()


# ---------------------------------------------------------- ask(): API handling

def _response(text="hi"):
    class Obj:
        pass
    r = Obj()
    r.choices = [Obj()]
    r.choices[0].message = Obj()
    r.choices[0].message.content = text
    r.usage = Obj()
    r.usage.prompt_tokens, r.usage.completion_tokens = 7, 11
    r.usage.completion_tokens_details = Obj()
    r.usage.completion_tokens_details.reasoning_tokens = 5
    r.system_fingerprint = "fp_1"
    return r


class _FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _http_error(cls, status, headers=None, body="x"):
    # A duck-typed response: avoids importing httpx/httpx2, whose name differs between openai versions.
    resp = types.SimpleNamespace(status_code=status, headers=headers or {}, request=types.SimpleNamespace())
    return cls(body, response=resp, body=None)


def test_429_sleeps_for_retry_after_then_retries(monkeypatch):
    fake = _FakeClient([_http_error(openai.RateLimitError, 429, {"retry-after": "7"}), _response("ok")])
    sleeps = []
    monkeypatch.setattr(client, "_get_client", lambda p: fake)
    monkeypatch.setattr(client.time, "sleep", sleeps.append)
    text, usage = client.ask("groq", "m429", [{"role": "user", "content": "q"}])
    assert text == "ok" and sleeps == [8.0] and len(fake.calls) == 2
    assert usage["prompt_tokens"] == 7 and usage["completion_tokens"] == 11
    assert usage["reasoning_tokens"] == 5 and usage["system_fingerprint"] == "fp_1" and usage["latency_s"] >= 0
    assert fake.calls[0]["temperature"] == 0.8
    assert fake.calls[0]["extra_body"] == {"reasoning_effort": "low"}


def test_429_with_a_huge_retry_after_gives_up_cleanly(monkeypatch):
    fake = _FakeClient([_http_error(openai.RateLimitError, 429, {"retry-after": "40000"})])
    monkeypatch.setattr(client, "_get_client", lambda p: fake)
    monkeypatch.setattr(client.time, "sleep", lambda s: pytest.fail("must not sleep for hours"))
    with pytest.raises(client.RateLimitExhausted):
        client.ask("groq", "mdaily", [{"role": "user", "content": "q"}])


def test_reasoning_effort_rejected_is_retried_without_it(monkeypatch):
    err = _http_error(openai.BadRequestError, 400, body="Unsupported parameter: reasoning_effort")
    fake = _FakeClient([err, _response("ok")])
    monkeypatch.setattr(client, "_get_client", lambda p: fake)
    text, usage = client.ask("cerebras", "mnoreason", [{"role": "user", "content": "q"}])
    assert text == "ok" and "extra_body" not in fake.calls[1] and usage["reasoning_effort_dropped"]


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client._clients.pop("groq", None)
    with pytest.raises(client.LLMError, match="GROQ_API_KEY"):
        client.ask("groq", "m", [{"role": "user", "content": "q"}])


def test_fake_provider_returns_pso_in_a_python_block():
    text, usage = client.ask("fake", "anything", [{"role": "user", "content": "ignored"}])
    name, code = extract(text)
    assert name == "PSOBaseline" and "def optimize(" in code and usage["system_fingerprint"] == "fake"


def test_log_lines_are_strict_json_even_with_nan_and_inf(tmp_path):
    path = tmp_path / "x.jsonl"
    run_llm.append_record(path, {"a": float("nan"), "b": [float("inf"), 1.5], "c": {"d": -float("inf")}})
    assert json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(c)) == {"a": None, "b": [None, 1.5], "c": {"d": None}}