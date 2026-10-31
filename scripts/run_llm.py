"""Run one LLM experiment: LLaMEA (arm A) or independent sampling (arm B).

    python scripts/run_llm.py --provider cerebras --model gpt-oss-120b --arm A --run 1
    python scripts/run_llm.py --provider fake --model test --arm B --run 1 --calls 5

Arm A (LLaMEA 1+1): call 1 = TASK_PROMPT, then build_feedback() on the current
    parent. The child replaces the parent if its fitness is >= the parent's.
Arm B (control):    every call is TASK_PROMPT alone; keep the best program.

Every call is scored with training_fitness(code) and nothing else, then appended
as one JSON line to results/llm_runs/<model>/<arm>/run<N>.jsonl. The run is
resumable: on start the log is read, the parent and the call count are rebuilt
from it, and the loop carries on from there. The best program is written to
found/<model>_<arm>_run<N>.py when the run ends (for any reason).

Stops: --calls reached, --stop-after-minutes elapsed, today's (UTC) token total
for the model within 5% of the daily cap, or the API says the quota is spent.
All of these exit 0 so a scheduled job can commit its logs; a real API failure
(bad key, unknown model) exits 1.

Fake-provider logs go to a separate directory ("fake_<model>") so they can never
be mistaken for real results when a real run resumes.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from algogenesis.fitness import training_fitness  # noqa: E402
from algogenesis.llm.client import ExtractError, LLMError, RateLimitExhausted, ask, extract  # noqa: E402
from algogenesis.llm.prompts import TASK_PROMPT, build_feedback  # noqa: E402

# Daily token caps per model (prompt + completion), from the provider free tiers.
DAILY_CAPS = {"cerebras": 1_000_000, "groq": 200_000}
CAP_MARGIN = 0.95  # stop at 95% of the cap
ERROR_TAIL_CHARS = 1500  # how much of a traceback goes back into the next prompt


# ------------------------------------------------------------------------ paths

def safe_name(model: str, provider: str = "") -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    return f"fake_{name}" if provider == "fake" else name


def log_path(data_dir: Path, provider: str, model: str, arm: str, run: int) -> Path:
    return data_dir / "results" / "llm_runs" / safe_name(model, provider) / arm / f"run{run}.jsonl"


def found_path(data_dir: Path, provider: str, model: str, arm: str, run: int) -> Path:
    return data_dir / "found" / f"{safe_name(model, provider)}_{arm}_run{run}.py"


# ------------------------------------------------------------------------- logs

def read_log(path: Path) -> list[dict]:
    """Read a JSONL log. A half-written last line (killed mid-write) is cut off."""
    if not path.exists():
        return []
    data = path.read_bytes()
    records, good_len, pos = [], 0, 0
    lines = data.splitlines(keepends=True)
    for i, raw in enumerate(lines):
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise RuntimeError(f"{path}: line {i + 1} is corrupt and is not the last line; fix by hand")
            break
        pos += len(raw)
        good_len = pos
    if good_len != len(data):
        print(f"[run_llm] dropping a partial last line in {path}", flush=True)
        with open(path, "r+b") as f:
            f.truncate(good_len)
    if records and not data[:good_len].endswith(b"\n"):
        with open(path, "ab") as f:
            f.write(b"\n")
    return records


def _json_safe(obj):
    """NaN / inf (e.g. the error of a run that never evaluated) become null, so the log is strict JSON."""
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n"
    with open(path, "ab") as f:
        f.write(line.encode("utf-8"))  # one write call per line
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------- token ledger

def tokens_today(data_dir: Path, provider: str, model: str) -> tuple[int, int]:
    """(tokens used today UTC by this provider/model across all arms and runs, largest single call)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    root = data_dir / "results" / "llm_runs" / safe_name(model, provider)
    used = biggest = 0
    for path in root.glob("*/run*.jsonl"):
        for rec in read_log(path):
            if rec.get("provider") == provider and rec.get("model") == model and str(rec.get("time", "")).startswith(today):
                t = rec["tokens"]
                n = (t.get("prompt") or 0) + (t.get("completion") or 0)
                used += n
                biggest = max(biggest, n)
    return used, biggest


# ------------------------------------------------------------ scoring & errors

def classify_error(error: str) -> str:
    """Map harness error text to syntax / interface / runtime / timeout / policy."""
    if "evaluation timed out" in error:
        return "timeout"
    if "policy violation" in error:
        return "policy"
    if re.search(r"^(Syntax|Indentation|Tab)Error", error, re.MULTILINE):
        return "syntax"
    if any(s in error for s in ("must define a function optimize", "optimize must accept", "optimize must return")):
        return "interface"
    return "runtime"


def score_code(code: str) -> dict:
    """Run training_fitness(code) and flatten the result for the log."""
    t0 = time.perf_counter()
    result = training_fitness(code)
    eval_s = time.perf_counter() - t0

    error = result.error
    if not error:
        failed = [r for r in result.runs if not r.ok]
        if failed:  # some or all runs crashed; the suite itself has no error set
            error = failed[0].error
    return {
        "fitness": float(result.fitness),
        "fitness_std": float(np.std([r.aocc for r in result.runs])) if result.runs else 0.0,
        "error": error,
        "error_type": classify_error(error) if error else None,
        "summary": result.summary(),
        "flags": list(result.flags),
        "eval_s": eval_s,
    }


# ------------------------------------------------------------------------ state

class State:
    """Parent / best-so-far, rebuilt by replaying the log through the same apply() the live loop uses."""

    def __init__(self, arm: str):
        self.arm = arm
        self.best = None      # arm A: the parent; arm B: the best program so far
        self.history = []     # one dict per call, for build_feedback
        self.last_error = None
        self.calls = 0

    def apply(self, rec: dict) -> bool:
        """Fold one call into the state. Returns True if it became the parent / new best."""
        accepted = False
        if rec["code"]:
            if self.best is None:
                accepted = True
            elif self.arm == "A":
                accepted = rec["fitness"] >= self.best["fitness"]   # 1+1: ties replace
            else:
                accepted = rec["fitness"] > self.best["fitness"]    # control: first best stays
        if accepted:
            self.best = rec
        self.calls += 1
        self.last_error = rec["error"]
        self.history.append({"call": rec["call"], "name": rec["name"], "fitness": rec["fitness"],
                             "error_type": rec["error_type"], "accepted": accepted,
                             "summary": rec["summary"]})
        return accepted


def build_prompt(state: State) -> str:
    if state.arm == "B" or state.best is None:
        return TASK_PROMPT
    p = state.best
    err = state.last_error[-ERROR_TAIL_CHARS:] if state.last_error else None
    return build_feedback(TASK_PROMPT, state.history, p["code"], p["fitness"], p["fitness_std"], err)


# -------------------------------------------------------------------------- run

def run(provider, model, arm, run_no, calls=50, deadline=None, data_dir=REPO, daily_cap=None) -> str:
    """Run (or resume) one arm/run. Returns "done", "time", "cap" or "quota"."""
    path = log_path(data_dir, provider, model, arm, run_no)
    records = read_log(path)
    for rec in records:
        if (rec["provider"], rec["model"], rec["arm"], rec["run"]) != (provider, model, arm, run_no):
            raise RuntimeError(f"{path} contains a record from {rec['provider']}/{rec['model']}/"
                               f"{rec['arm']}/run{rec['run']}; refusing to mix logs")
    state = State(arm)
    for rec in records:
        state.apply(rec)

    cap = daily_cap if daily_cap is not None else DAILY_CAPS.get(provider)
    used, biggest = tokens_today(data_dir, provider, model) if cap else (0, 0)
    tag = f"[{model} {arm}{run_no}]"
    if records:
        print(f"{tag} resuming after {state.calls} calls", flush=True)

    status = "done"
    while state.calls < calls:
        if deadline is not None and time.monotonic() >= deadline:
            status = "time"
            break
        if cap and used + biggest > CAP_MARGIN * cap:
            print(f"{tag} daily token cap: used {used:,} of {cap:,} (+{biggest:,} per call); stopping", flush=True)
            status = "cap"
            break

        prompt = build_prompt(state)
        try:
            text, usage = ask(provider, model, [{"role": "user", "content": prompt}])
        except RateLimitExhausted as e:
            print(f"{tag} {e}", flush=True)
            status = "quota"
            break

        try:
            name, code = extract(text)
            scored = score_code(code)
        except ExtractError as e:
            name, code = None, None
            scored = {"fitness": 0.0, "fitness_std": 0.0, "error": str(e), "error_type": e.error_type,
                      "summary": [], "flags": [], "eval_s": 0.0}

        n_tokens = (usage["prompt_tokens"] or 0) + (usage["completion_tokens"] or 0)
        used += n_tokens
        biggest = max(biggest, n_tokens)
        rec = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "call": state.calls + 1,
            "arm": arm,
            "run": run_no,
            "model": model,
            "provider": provider,
            "fingerprint": usage["system_fingerprint"],
            "tokens": {"prompt": usage["prompt_tokens"], "completion": usage["completion_tokens"],
                       "reasoning": usage["reasoning_tokens"]},
            "latency_s": usage["latency_s"],
            "reasoning_effort_dropped": usage["reasoning_effort_dropped"],
            "prompt": prompt,
            "raw": text,
            "name": name,
            "code": code,
            "error_type": scored["error_type"],
            "error": scored["error"],
            "fitness": scored["fitness"],
            "fitness_std": scored["fitness_std"],
            "eval_s": scored["eval_s"],
            "summary": scored["summary"],
            "flags": scored["flags"],
        }
        rec["accepted"] = state.apply(rec)
        append_record(path, rec)
        best = state.best["fitness"] if state.best else float("nan")
        print(f"{tag} call {rec['call']}/{calls} fitness {rec['fitness']:.4f} best {best:.4f} "
              f"err={rec['error_type']} tokens={n_tokens:,} (today {used:,})", flush=True)

    save_best(state, provider, model, arm, run_no, data_dir)
    return status


def save_best(state: State, provider, model, arm, run_no, data_dir) -> None:
    if state.best is None:
        return
    b = state.best
    out = found_path(data_dir, provider, model, arm, run_no)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Found by arm {arm}, run {run_no}, {provider}/{model}; training fitness {b['fitness']:.4f} "
              f"(call {b['call']} of {state.calls}); name: {b['name']}\n")
    tmp = out.with_suffix(".tmp")
    tmp.write_text(header + b["code"], encoding="utf-8")
    os.replace(tmp, out)
    print(f"[{model} {arm}{run_no}] best program (fitness {b['fitness']:.4f}) saved to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, help="cerebras | groq | fake")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True, choices=["A", "B"])
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--stop-after-minutes", type=float, default=None)
    parser.add_argument("--data-dir", type=Path, default=REPO, help="where results/ and found/ live")
    parser.add_argument("--daily-cap", type=int, default=None, help="override the per-model daily token cap")
    args = parser.parse_args()

    deadline = time.monotonic() + args.stop_after_minutes * 60 if args.stop_after_minutes else None
    try:
        status = run(args.provider, args.model, args.arm, args.run, args.calls, deadline,
                     args.data_dir, args.daily_cap)
    except LLMError as e:
        sys.exit(f"API error: {e}")
    print(f"finished: {status}")


if __name__ == "__main__":  # required: the scorer uses multiprocessing "spawn"
    main()
