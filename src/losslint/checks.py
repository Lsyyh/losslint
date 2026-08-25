"""Lint checks: pure functions from a parsed RunLog to Findings."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from losslint.accel import finite_pairs, median, nonfinite_positions, spike_candidates
from losslint.parsing import RunLog
from losslint.report import Finding

EVAL_HINTS = ("eval", "val", "test")
MAX_EXAMPLES = 5


@dataclass(frozen=True)
class CheckOptions:
    """Tunable thresholds for the checks (all have conservative defaults)."""

    spike_factor: float = 10.0
    spike_window: int = 25
    divergence_ratio: float = 1.25
    divergence_min_points: int = 20
    overfit_min_worsen: int = 3
    stagnation_min_improvement: float = 0.01
    stagnation_min_points: int = 20


def find_series(run: RunLog, role: str) -> tuple[str, list[float | None]] | None:
    """Locate the train or eval loss series by conventional naming."""
    if role == "train":
        if "loss" in run.series:
            return "loss", run.series["loss"]
        for name, values in run.series.items():
            lowered = name.lower()
            if "loss" in lowered and not any(h in lowered for h in EVAL_HINTS):
                return name, values
        return None
    if "eval_loss" in run.series:
        return "eval_loss", run.series["eval_loss"]
    for name, values in run.series.items():
        lowered = name.lower()
        if "loss" in lowered and any(h in lowered for h in EVAL_HINTS):
            return name, values
    return None


def check_nan_inf(run: RunLog, options: CheckOptions | None = None) -> list[Finding]:
    """Flag NaN/Inf values in any numeric series (they poison later checkpoints)."""
    findings: list[Finding] = []
    for name, values in run.series.items():
        bad_positions = nonfinite_positions(values)
        bad_steps = [run.steps[i] for i in bad_positions]
        if bad_steps:
            shown = ", ".join(str(s) for s in bad_steps[:MAX_EXAMPLES])
            findings.append(
                Finding(
                    check="nan_inf",
                    severity="error",
                    message=(
                        f"series '{name}' has {len(bad_steps)} NaN/Inf value(s) "
                        f"(first steps: {shown}) — every metric after this is suspect"
                    ),
                    details={
                        "series": name,
                        "steps": bad_steps[:MAX_EXAMPLES],
                        "count": len(bad_steps),
                    },
                    source=run.source,
                )
            )
    return findings


def check_step_backtrack(run: RunLog, options: CheckOptions | None = None) -> list[Finding]:
    """Flag strictly decreasing step indices (resume/logging bugs).

    Duplicate steps are deliberately allowed: HF ``log_history`` interleaves
    train and eval entries that share a step.
    """
    backtracks = [
        (i, run.steps[i]) for i in range(1, len(run.steps)) if run.steps[i] < run.steps[i - 1]
    ]
    if not backtracks:
        return []
    return [
        Finding(
            check="step_backtrack",
            severity="warning",
            message=(
                f"step index moved backwards {len(backtracks)} time(s) "
                "(first: position {pos} -> step {step}) — check resume/logging logic"
            ).format(pos=backtracks[0][0], step=backtracks[0][1]),
            details={"backtracks": backtracks[:MAX_EXAMPLES], "count": len(backtracks)},
            source=run.source,
        )
    ]


def check_loss_spike(run: RunLog, options: CheckOptions | None = None) -> list[Finding]:
    """Flag isolated loss spikes above ``spike_factor`` x the local rolling median.

    Exact medians are only computed for candidates that first exceed
    ``spike_factor`` x the sliding window minimum — the window median is always
    >= its minimum, so the pre-filter can never hide a real spike, yet it turns
    the common all-quiet case into a single O(n) sweep.
    """
    opts = options or CheckOptions()
    entry = find_series(run, "train")
    if entry is None:
        return []
    positions, vals = finite_pairs(entry[1])
    if len(vals) < 6:
        return []
    steps = [run.steps[i] for i in positions]
    half = opts.spike_window // 2
    spikes: list[tuple[int, float]] = []
    for pos in spike_candidates(vals, half, opts.spike_factor):
        value = vals[pos]
        neighbors = vals[max(0, pos - half) : pos] + vals[pos + 1 : pos + 1 + half]
        if len(neighbors) < 5:
            continue
        local_median = median(neighbors)
        if local_median > 0 and value > opts.spike_factor * local_median:
            spikes.append((steps[pos], value))
    if not spikes:
        return []
    shown = ", ".join(f"step {s} (loss {v:.3g})" for s, v in spikes[:MAX_EXAMPLES])
    return [
        Finding(
            check="loss_spike",
            severity="warning",
            message=(
                f"train loss spiked {len(spikes)} time(s) above "
                f"{opts.spike_factor}x its local median "
                f"({shown}) — data shard or LR bug?"
            ),
            details={
                "steps": [step for step, _ in spikes[:MAX_EXAMPLES]],
                "count": len(spikes),
            },
            source=run.source,
        )
    ]


def check_divergence(run: RunLog, options: CheckOptions | None = None) -> list[Finding]:
    """Flag runs whose train loss ends materially higher than it started."""
    opts = options or CheckOptions()
    entry = find_series(run, "train")
    if entry is None:
        return []
    _, vals = finite_pairs(entry[1])
    if len(vals) < opts.divergence_min_points:
        return []
    decile = max(1, len(vals) // 10)
    first_median = median(vals[:decile])
    last_median = median(vals[-decile:])
    if first_median <= 0 or last_median <= opts.divergence_ratio * first_median:
        return []
    return [
        Finding(
            check="divergence",
            severity="error",
            message=(
                f"train loss diverged: last-decile median {last_median:.3g} is "
                f"{last_median / first_median:.2f}x the first-decile median {first_median:.3g}"
            ),
            details={
                "first_median": first_median,
                "last_median": last_median,
                "ratio": last_median / first_median,
            },
            source=run.source,
        )
    ]


def check_overfit_onset(run: RunLog, options: CheckOptions | None = None) -> list[Finding]:
    """Flag eval loss worsening at the tail while train loss still improves."""
    opts = options or CheckOptions()
    eval_entry = find_series(run, "eval")
    train_entry = find_series(run, "train")
    if eval_entry is None or train_entry is None:
        return []
    eval_positions, eval_vals = finite_pairs(eval_entry[1])
    _, train_vals = finite_pairs(train_entry[1])
    if len(eval_vals) < opts.overfit_min_worsen + 1 or len(train_vals) < 8:
        return []
    eval_steps = [run.steps[i] for i in eval_positions]
    worsen = 0
    for prev, cur in itertools.pairwise(eval_vals):
        worsen = worsen + 1 if cur > prev else 0
    if worsen < opts.overfit_min_worsen:
        return []
    quarter = max(1, len(train_vals) // 4)
    early_median = median(train_vals[:quarter])
    late_median = median(train_vals[-quarter:])
    if (
        early_median <= 0
        or early_median - late_median < opts.stagnation_min_improvement * early_median
    ):
        return []
    start_step = eval_steps[-worsen - 1]
    return [
        Finding(
            check="overfit_onset",
            severity="info",
            message=(
                f"eval loss has worsened for the last {worsen} eval point(s) "
                f"(since step {start_step}) while train loss still improves — "
                "best checkpoint is probably behind you"
            ),
            details={"consecutive_worsening": worsen, "since_step": start_step},
            source=run.source,
        )
    ]


def check_stagnation(run: RunLog, options: CheckOptions | None = None) -> list[Finding]:
    """Flag runs whose train loss stopped improving over the last half-quarter window."""
    opts = options or CheckOptions()
    entry = find_series(run, "train")
    if entry is None:
        return []
    _, vals = finite_pairs(entry[1])
    if len(vals) < opts.stagnation_min_points:
        return []
    quarter = max(1, len(vals) // 4)
    previous_median = median(vals[-2 * quarter : -quarter])
    last_median = median(vals[-quarter:])
    if previous_median <= 0:
        return []
    improvement = (previous_median - last_median) / previous_median
    if improvement >= opts.stagnation_min_improvement:
        return []
    return [
        Finding(
            check="stagnation",
            severity="info",
            message=(
                f"train loss has plateaued: {improvement:.1%} improvement over the "
                "last quarter of the run — consider stopping or raising the LR"
            ),
            details={"recent_improvement": improvement},
            source=run.source,
        )
    ]


ALL_CHECKS = (
    check_nan_inf,
    check_step_backtrack,
    check_loss_spike,
    check_divergence,
    check_overfit_onset,
    check_stagnation,
)


def run_all_checks(run: RunLog, options: CheckOptions | None = None) -> list[Finding]:
    """Run every check plus parse-issue reporting, ordered by severity."""
    findings: list[Finding] = []
    for issue in run.parse_issues:
        findings.append(
            Finding(
                check="parse",
                severity="warning",
                message=f"skipped a malformed log entry: {issue}",
                details={"issue": issue},
                source=run.source,
            )
        )
    for check in ALL_CHECKS:
        findings.extend(check(run, options))
    return findings
