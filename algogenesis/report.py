"""Comparison report: results tables, significance tests and convergence curves.

Works on SuiteResults from the harness, so baselines and LLM-generated
algorithms are compared exactly the same way.

Significance: for every function x dim, each algorithm's final errors are
compared with a reference algorithm's using a two-sided Mann-Whitney U
(Wilcoxon rank-sum) test at alpha = 0.05. Errors at or below the success
threshold (1e-8) count as equal, so two algorithms that both solved a problem
are not called different because one reached 1e-14 and the other 1e-12.
The result is marked + (significantly better than the reference), − (worse)
or ≈ (no significant difference), the usual win/tie/loss format in
metaheuristics papers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import mannwhitneyu

from .harness import SUCCESS_ERROR, SuiteResult

ALPHA = 0.05
BETTER, WORSE, TIE = "+", "−", "≈"

# Reference categorical palette, light mode, in fixed order: an algorithm keeps
# its colour in every plot because colours are assigned by its position in the
# algorithm list, never by rank.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_SECONDARY, GRID, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def problems(suites: Sequence[SuiteResult]) -> list[tuple[str, int]]:
    """Every (function, dim) pair that appears in any suite, in first-seen order."""
    seen: dict[tuple[str, int], None] = {}
    for s in suites:
        for r in s.runs:
            seen.setdefault((r.function, r.dim), None)
    return list(seen)


def final_errors(suite: SuiteResult, function: str, dim: int) -> np.ndarray:
    return np.array([r.best_error for r in suite.runs if r.function == function and r.dim == dim])


def significance(errors: np.ndarray, reference: np.ndarray, alpha: float = ALPHA) -> tuple[str, float]:
    """Return (+/−/≈, p-value) for ``errors`` against ``reference``; lower error is better."""
    a = np.maximum(errors, SUCCESS_ERROR)
    b = np.maximum(reference, SUCCESS_ERROR)
    if len(a) == 0 or len(b) == 0:
        return TIE, float("nan")
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return TIE, 1.0
    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    if p >= alpha:
        return TIE, p
    # Decide direction by median, falling back to mean when medians coincide.
    da, db = np.median(a), np.median(b)
    if da == db:
        da, db = np.mean(a), np.mean(b)
    return (BETTER if da < db else WORSE), p


def comparison_rows(suites: Sequence[SuiteResult], reference: str) -> list[dict]:
    """One row per (algorithm, function, dim): summary stats plus the test against ``reference``."""
    by_name = {s.name: s for s in suites}
    if reference not in by_name:
        raise ValueError(f"reference {reference!r} is not one of {list(by_name)}")
    rows = []
    for s in suites:
        summary = {(r["function"], r["dim"]): r for r in s.summary()}
        for fn, dim in problems(suites):
            if (fn, dim) not in summary:
                continue
            row = dict(summary[(fn, dim)])
            if s.name == reference:
                row["vs_reference"], row["p_value"] = "", float("nan")
            else:
                row["vs_reference"], row["p_value"] = significance(
                    final_errors(s, fn, dim), final_errors(by_name[reference], fn, dim))
            rows.append(row)
    return rows


def win_tie_loss(rows: list[dict], reference: str) -> dict[str, tuple[int, int, int]]:
    """Per algorithm: how many problems it is better / equal / worse than the reference on."""
    totals: dict[str, list[int]] = {}
    for r in rows:
        if r["algorithm"] == reference:
            continue
        t = totals.setdefault(r["algorithm"], [0, 0, 0])
        t[{BETTER: 0, TIE: 1, WORSE: 2}[r["vs_reference"]]] += 1
    return {k: tuple(v) for k, v in totals.items()}


# --- Tables -----------------------------------------------------------------------

def _grid(rows: list[dict]) -> tuple[list[str], list[tuple[str, int]], dict]:
    algorithms = list(dict.fromkeys(r["algorithm"] for r in rows))
    probs = list(dict.fromkeys((r["function"], r["dim"]) for r in rows))
    cell = {(r["algorithm"], r["function"], r["dim"]): r for r in rows}
    return algorithms, probs, cell


def _best_in_row(cell: dict, algorithms: list[str], fn: str, dim: int, key: str, lower_is_better: bool) -> set[str]:
    vals = {a: cell[(a, fn, dim)][key] for a in algorithms if (a, fn, dim) in cell}
    if not vals:
        return set()
    target = min(vals.values()) if lower_is_better else max(vals.values())
    return {a for a, v in vals.items() if v == target}


def markdown_tables(rows: list[dict], reference: str) -> str:
    algorithms, probs, cell = _grid(rows)
    head = "| Problem | " + " | ".join(algorithms) + " |"
    sep = "|---|" + "---|" * len(algorithms)
    out = []

    out += ["## Final error (mean ± std)",
            "",
            f"Best mean in **bold**. {BETTER} / {TIE} / {WORSE}: significantly better than, no different from, or "
            f"worse than **{reference}** (Mann-Whitney U, α = {ALPHA}).",
            "", head, sep]
    for fn, dim in probs:
        best = _best_in_row(cell, algorithms, fn, dim, "mean_error", True)
        cells = []
        for a in algorithms:
            r = cell.get((a, fn, dim))
            if r is None:
                cells.append("–")
                continue
            text = f"{r['mean_error']:.2e} ± {r['std_error']:.1e}"
            text = f"**{text}**" if a in best else text
            cells.append(f"{text} {r['vs_reference']}".rstrip())
        out.append(f"| {fn} {dim}-D | " + " | ".join(cells) + " |")

    wtl = win_tie_loss(rows, reference)
    if wtl:
        out += ["", f"Win / tie / loss against {reference}: " +
                ", ".join(f"{a} {w}/{t}/{l}" for a, (w, t, l) in wtl.items())]

    out += ["", "## Success rate and AOCC", "",
            "Success = final error ≤ 1e-8. AOCC = area over the convergence curve, 0 to 1, higher is better "
            "(rewards both speed and final quality).", "", head, sep]
    for fn, dim in probs:
        best = _best_in_row(cell, algorithms, fn, dim, "mean_aocc", False)
        cells = []
        for a in algorithms:
            r = cell.get((a, fn, dim))
            if r is None:
                cells.append("–")
                continue
            aocc = f"**{r['mean_aocc']:.3f}**" if a in best else f"{r['mean_aocc']:.3f}"
            cells.append(f"{r['success_rate']:.0%} / {aocc}")
        out.append(f"| {fn} {dim}-D | " + " | ".join(cells) + " |")

    out += ["", "## Compute cost per run", "",
            "Wall time in seconds, and in brackets the part spent on the algorithm's own work (excluding "
            "benchmark evaluations). Every algorithm gets the same evaluation budget.", "", head, sep]
    for fn, dim in probs:
        cells = []
        for a in algorithms:
            r = cell.get((a, fn, dim))
            cells.append("–" if r is None else f"{r['mean_runtime_s']:.3f} ({r['mean_overhead_s']:.3f})")
        out.append(f"| {fn} {dim}-D | " + " | ".join(cells) + " |")

    failures = [r for r in rows if r["failures"]]
    if failures:
        out += ["", "Runs that crashed: " +
                ", ".join(f"{r['algorithm']} on {r['function']} {r['dim']}-D: {r['failures']}/{r['runs']}"
                          for r in failures)]
    return "\n".join(out) + "\n"


def _latex_sci(x: float) -> str:
    if not np.isfinite(x):
        return r"\text{--}"
    if x == 0:
        return "0"
    mantissa, exp = f"{x:.2e}".split("e")
    return rf"{mantissa}\mathrm{{e}}{{{int(exp)}}}"


def _tex_escape(text: str) -> str:
    for ch in "&%$#_{}":
        text = text.replace(ch, "\\" + ch)
    return text


def latex_error_table(rows: list[dict], reference: str) -> str:
    """Final-error table as a booktabs LaTeX table, ready for the paper."""
    algorithms, probs, cell = _grid(rows)
    ref_tex = _tex_escape(reference)
    sign = {BETTER: "$^{+}$", WORSE: "$^{-}$", TIE: "$^{\\approx}$", "": ""}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Final error (mean $\pm$ std) over independent runs. Best mean in bold. "
        rf"Superscripts: significantly better ($+$), worse ($-$) or not different ($\approx$) "
        rf"compared with {ref_tex} (Mann--Whitney U, $\alpha={ALPHA}$).}}",
        r"\label{tab:final-error}",
        r"\begin{tabular}{l" + "c" * len(algorithms) + "}",
        r"\toprule",
        "Problem & " + " & ".join(_tex_escape(a) for a in algorithms) + r" \\",
        r"\midrule",
    ]
    for fn, dim in probs:
        best = _best_in_row(cell, algorithms, fn, dim, "mean_error", True)
        cells = []
        for a in algorithms:
            r = cell.get((a, fn, dim))
            if r is None:
                cells.append("--")
                continue
            body = f"{_latex_sci(r['mean_error'])} \\pm {_latex_sci(r['std_error'])}"
            body = rf"\mathbf{{{body}}}" if a in best else body
            cells.append(f"${body}${sign[r['vs_reference']]}")
        lines.append(f"{fn.capitalize()} ({dim}-D) & " + " & ".join(cells) + r" \\")
    wtl = win_tie_loss(rows, reference)
    if wtl:
        lines.append(r"\midrule")
        lines.append(rf"W/T/L vs {ref_tex} & " + " & ".join(
            "--" if a == reference else "{}/{}/{}".format(*wtl[a]) for a in algorithms) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# --- Convergence curves -------------------------------------------------------------

def _dodge(values: list[float], min_gap: float) -> list[float]:
    """Spread label positions apart (in log10 space) so direct labels don't overlap."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    out = np.empty_like(placed)
    out[order] = placed
    return list(out)


def plot_convergence(suites: Sequence[SuiteResult], function: str, dim: int, path: str | Path,
                     colors: dict[str, str] | None = None) -> None:
    """Median best-so-far error over runs, with the interquartile range as a band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = colors or {s.name: PALETTE[i % len(PALETTE)] for i, s in enumerate(suites)}
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ends, lowest = [], np.inf
    for s in suites:
        runs = [r for r in s.runs if r.function == function and r.dim == dim and r.evals]
        if not runs:
            continue
        budget = runs[0].budget
        curves = np.empty((len(runs), budget))
        for i, r in enumerate(runs):
            curves[i, : r.evals] = r.history
            curves[i, r.evals:] = r.history[-1]
        curves = np.maximum(curves, 1e-16)  # exact zeros can't go on a log axis
        evals = np.arange(1, budget + 1)
        q25, med, q75 = np.percentile(curves, [25, 50, 75], axis=0)
        ax.fill_between(evals, q25, q75, color=colors[s.name], alpha=0.15, linewidth=0)
        ax.plot(evals, med, color=colors[s.name], linewidth=2, label=s.name, solid_capstyle="round")
        ends.append((s.name, evals[-1], med[-1]))
        lowest = min(lowest, q25.min())

    ax.set_yscale("log")
    # Only show the success threshold when some curve gets near it; otherwise it would
    # stretch the axis over decades no algorithm reaches.
    if lowest < 100 * SUCCESS_ERROR:
        ax.axhline(SUCCESS_ERROR, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0.01, SUCCESS_ERROR, " success threshold (1e-8)", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=7.5, color=INK_SECONDARY)

    # Direct labels at the line ends (up to 4 series), nudged apart; the legend is always there too.
    if 2 <= len(ends) <= 4:
        lo, hi = (np.log10(v) for v in ax.get_ylim())
        ys = _dodge([np.log10(e[2]) for e in ends], 0.06 * (hi - lo))
        for (name, x, _), y in zip(ends, ys):
            ax.annotate(name, (x, 10**y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK, annotation_clip=False)

    ax.set_xlabel("Function evaluations", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Best-so-far error  (f − f*)", color=INK_SECONDARY, fontsize=9)
    n_runs = max((len(final_errors(s, function, dim)) for s in suites), default=0)
    ax.set_title(f"{function.capitalize()}, {dim}-D: median of {n_runs} runs, band = interquartile range",
                 fontsize=10, loc="left", color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    if len(ends) >= 2:
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
