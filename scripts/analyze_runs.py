"""Analysis of the LLM run logs: how the search progressed, what failed, what it cost.

    python scripts/analyze_runs.py

Reads data/results/llm_runs/<model>/<arm>/run*.jsonl and writes data/run_analysis/:
summary.md, run_analysis.json, fig_search_curves.png and fig_validity.png.
Uses only the logs (training fitness); held-out results are in make_paper_assets.py.
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/run_analysis"
CALLS = 50
GROUPS = {"gpt-oss-120b/A": ("gpt-oss loop (A)", "#D55E00"),
          "gpt-oss-120b/B": ("gpt-oss control (B)", "#0072B2"),
          "qwen-3.8-27b/A": ("Qwen loop (A)", "#009E73")}
# Training fitness of the baselines (scripts/demo.py / report Table 7.1).
BASELINE_TRAIN = {"PSO": 0.332, "CMA-ES": 0.500, "BIPOP-CMA-ES": 0.582}
FAMILIES = {
    "CMA-ES": r"covariance|cma|rank[-_ ]?mu",
    "Differential evolution": r"differential|mutation.*\bF\b|\bCR\b|crossover_rate|de/rand|de/best",
    "Particle swarm": r"velocity|inertia|pbest|gbest|swarm",
    "Nelder-Mead / simplex": r"nelder|simplex|reflection|contraction",
    "Restarts": r"restart|ipop|bipop",
}


def load():
    runs = {}
    for path in sorted(glob.glob(str(ROOT / "data/results/llm_runs/*/*/run*.jsonl"))):
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        p = Path(path)
        runs.setdefault(f"{p.parts[-3]}/{p.parts[-2]}", []).append((p.stem, rows))
    return runs


def code_of(r):
    m = re.search(r"```(?:python)?\n(.*?)```", r.get("raw") or "", re.S)
    return m.group(1) if m else (r.get("code") or "")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load()
    res = {}
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for key, (label, colour) in GROUPS.items():
        curves, best_call, n_impr, first_half_err, second_half_err = [], [], [], [], []
        valid_fit, lengths, errors, tokens, lat, fam = [], [], Counter(), [], [], Counter()
        accepted = []
        for _, rows in runs[key]:
            best, curve, last_impr, impr = -1.0, [], None, 0
            for r in rows:
                ok = r.get("error_type") is None and r.get("fitness") is not None
                if ok and r["fitness"] > best:
                    best, last_impr, impr = r["fitness"], int(r["call"]), impr + 1
                curve.append(best if best >= 0 else 0.0)  # no valid program yet scores 0, as in AOCC
                errors[r.get("error_type") or "none"] += 1
                (first_half_err if int(r["call"]) <= 25 else second_half_err).append(not ok)
                t = r.get("tokens") or {}
                tokens.append((t.get("prompt") or 0) + (t.get("completion") or 0))
                lat.append(r.get("latency_s") or 0)
                if key.endswith("/A") and int(r["call"]) > 1:
                    accepted.append(bool(r.get("accepted")))
                if ok:
                    valid_fit.append(r["fitness"])
                    code = code_of(r)
                    lengths.append(len([ln for ln in code.splitlines() if ln.strip()]))
                    low = code.lower()
                    for f, pat in FAMILIES.items():
                        if re.search(pat, low if f != "CMA-ES" else code, re.I):
                            fam[f] += 1
            curves.append(curve + [np.nan] * (CALLS - len(curve)))
            best_call.append(last_impr)
            n_impr.append(impr)
        C = np.array(curves, dtype=float)
        # a partial run stays NaN after its last call
        mean_curve = np.nanmean(C, axis=0)
        x = np.arange(1, CALLS + 1)
        for c in C:
            ax.plot(x, c, color=colour, lw=0.6, alpha=0.35)
        ax.plot(x, mean_curve, color=colour, lw=2, label=label)
        vf = np.array(valid_fit)
        rho, p = spearmanr(lengths, valid_fit)
        n_valid = len(vf)
        res[key] = {
            "label": label, "runs": len(curves),
            "mean_best_at_call": {c: round(float(np.nanmean(C[:, c - 1])), 4) for c in (5, 10, 25, 50)},
            "call_of_final_best": best_call, "median_call_of_final_best": float(np.median(best_call)),
            "improvements_per_run": n_impr,
            "error_rate_calls_1_25": round(float(np.mean(first_half_err)), 3),
            "error_rate_calls_26_50": round(float(np.mean(second_half_err)), 3) if second_half_err else None,
            "errors": dict(errors),
            "valid_programs": n_valid,
            "valid_mean_fitness": round(float(vf.mean()), 4),
            "share_valid_above": {b: round(float((vf > v).mean()), 3) for b, v in BASELINE_TRAIN.items()},
            "median_tokens_per_call": int(np.median(tokens)),
            "median_latency_s": round(float(np.median(lat)), 1),
            "tokens_per_improvement": int(sum(tokens) / max(1, sum(n_impr))),
            "median_code_lines": int(np.median(lengths)),
            "spearman_lines_vs_fitness": {"rho": round(float(rho), 3), "p": round(float(p), 4)},
            "acceptance_rate": round(float(np.mean(accepted)), 3) if accepted else None,
            "family_share": {f: round(fam[f] / n_valid, 3) for f in FAMILIES},
        }
    for b, v in BASELINE_TRAIN.items():
        ax.axhline(v, color="#8a8a8a", lw=0.7, ls="--")
        ax.text(50.5, v, b, fontsize=7, va="center", color="#555555")
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("LLM call")
    ax.set_ylabel("Best training fitness so far (AOCC)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_search_curves.png", dpi=200)
    plt.close(fig)

    # validity by call block
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    blocks = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    w = 0.26
    for k, (key, (label, colour)) in enumerate(GROUPS.items()):
        rates = []
        for a, b in blocks:
            v = [r.get("error_type") is None and r.get("fitness") is not None
                 for _, rows in runs[key] for r in rows if a <= int(r["call"]) <= b]
            rates.append(np.mean(v) if v else np.nan)
        ax.bar(np.arange(len(blocks)) + (k - 1) * w, rates, w, color=colour, label=label)
    ax.set_xticks(range(len(blocks)), [f"{a}–{b}" for a, b in blocks])
    ax.set_xlabel("LLM calls")
    ax.set_ylabel("Share of programs that ran")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_validity.png", dpi=200)
    plt.close(fig)

    (OUT / "run_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_summary(res)
    print(json.dumps(res, indent=1))


def write_summary(res):
    g = res
    L = ["# LLM run analysis", "",
         "Generated by `scripts/analyze_runs.py` from the logs in `data/results/llm_runs/`. "
         "All fitness values here are training fitness (BBOB instances 1–3, 5-D); held-out results are in the paper.",
         "", "| | " + " | ".join(v["label"] for v in g.values()) + " |",
         "|---|" + "---|" * len(g)]

    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(v) for v in g.values()) + " |")

    row("Runs", lambda v: str(v["runs"]))
    for c in (5, 10, 25, 50):
        row(f"Mean best fitness after {c} calls", lambda v, c=c: f"{v['mean_best_at_call'][c]:.3f}")
    row("Call at which the final best was found (median)", lambda v: f"{v['median_call_of_final_best']:.0f}")
    row("Improvements per run", lambda v: ", ".join(map(str, v["improvements_per_run"])))
    row("Error rate, calls 1–25 / 26–50", lambda v: f"{v['error_rate_calls_1_25']:.0%} / "
        + (f"{v['error_rate_calls_26_50']:.0%}" if v["error_rate_calls_26_50"] is not None else "–"))
    row("Valid programs", lambda v: str(v["valid_programs"]))
    row("Mean fitness of valid programs", lambda v: f"{v['valid_mean_fitness']:.3f}")
    for b in BASELINE_TRAIN:
        row(f"Valid programs above {b} ({BASELINE_TRAIN[b]})", lambda v, b=b: f"{v['share_valid_above'][b]:.0%}")
    row("Child accepted (loop only)", lambda v: f"{v['acceptance_rate']:.0%}" if v["acceptance_rate"] is not None else "–")
    row("Median tokens per call", lambda v: f"{v['median_tokens_per_call']:,}")
    row("Tokens per improvement", lambda v: f"{v['tokens_per_improvement']:,}")
    row("Median latency per call (s)", lambda v: f"{v['median_latency_s']}")
    row("Median lines of code", lambda v: str(v["median_code_lines"]))
    row("Spearman: code length vs fitness", lambda v: f"ρ = {v['spearman_lines_vs_fitness']['rho']:.2f} "
        f"(p = {v['spearman_lines_vs_fitness']['p']:.3f})")
    for f in FAMILIES:
        row(f"Valid programs using {f}", lambda v, f=f: f"{v['family_share'][f]:.0%}")
    L += ["", "Family shares come from a keyword scan of each program's code, so they show what a program "
          "mentions, not what it does.", "",
          "![Search curves](fig_search_curves.png)", "", "![Validity by call](fig_validity.png)", ""]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
