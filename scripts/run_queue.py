"""Run a queue of (arm, run) jobs for one model, one after another.

    python scripts/run_queue.py --provider cerebras --model gpt-oss-120b \
        --queue A:5,B:5 --stop-after-minutes 330

"A:5,B:5" means arm A runs 1..5, then arm B runs 1..5. Everything is resumable:
finished runs are skipped instantly, a half-finished run carries on, so the
same command can be fired again every few hours until the queue is complete.
The time limit and the daily token cap are shared by the whole queue.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_llm import REPO, run  # noqa: E402

from algogenesis.llm.client import LLMError  # noqa: E402


def parse_queue(spec: str) -> list[tuple[str, int]]:
    jobs = []
    for part in spec.split(","):
        arm, n = part.strip().split(":")
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be A or B, got {arm!r}")
        jobs += [(arm, i) for i in range(1, int(n) + 1)]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--queue", required=True, help='e.g. "A:5,B:5"')
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--stop-after-minutes", type=float, default=None)
    parser.add_argument("--data-dir", type=Path, default=REPO)
    parser.add_argument("--daily-cap", type=int, default=None)
    args = parser.parse_args()

    deadline = time.monotonic() + args.stop_after_minutes * 60 if args.stop_after_minutes else None
    for arm, run_no in parse_queue(args.queue):
        try:
            status = run(args.provider, args.model, arm, run_no, args.calls, deadline,
                         args.data_dir, args.daily_cap)
        except LLMError as e:
            sys.exit(f"API error: {e}")
        if status != "done":
            print(f"queue stopped at {arm}{run_no}: {status}")
            return
    print("queue complete")


if __name__ == "__main__":
    main()
