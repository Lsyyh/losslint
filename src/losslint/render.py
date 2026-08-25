"""Human-readable terminal rendering: per-run groups, loss sparklines, colors.

The renderer is dependency-free ANSI: colors are applied only when the caller
enables them (the CLI ties that to ``--color`` / TTY / ``NO_COLOR``), and the
sparkline falls back to an ASCII ramp when the output stream is not UTF-8, so
redirected output and legacy Windows code pages never break.

Layout contract: every series row renders ``label │ sparkline │ first → last``
into fixed-width columns — the sparkline box is always ``SPARK_BINS`` cells and
the numeric spans are right-aligned — so rows from the same report line up
vertically regardless of how sparse a series is.
"""

from __future__ import annotations

import math

from losslint.checks import find_series
from losslint.report import SEVERITIES, SEVERITY_RANK, Finding, RunReport

SPARK_BINS = 24
LABEL_WIDTH_MAX = 12
LABEL_WIDTH_MIN = 5
BLOCK_RAMPS = "▁▂▃▄▅▆▇█"
# ASCII fallback ramp. Every glyph here either rests on the text baseline or
# floats *above* it (never dips below, unlike ``_``): the lower envelope of the
# curve therefore stays flat instead of ragged. Order is by increasing ink
# height so the ramp still reads as a rising curve.
ASCII_RAMPS = ".:-=+*#%@"


class _Palette:
    """Minimal ANSI wrapper; every method is a no-op when disabled."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def error(self, text: str) -> str:
        return self._wrap("1;31", text)

    def warning(self, text: str) -> str:
        return self._wrap("0;33", text)

    def info(self, text: str) -> str:
        return self._wrap("0;36", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def clean(self, text: str) -> str:
        return self._wrap("0;32", text)


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.3g}"


def _sparkline(values: list[float | None], *, unicode_ok: bool, bins: int = SPARK_BINS) -> str:
    """Compress a series into up to ``bins`` block characters.

    The bins are laid over the *present* points only: an eval series logged
    every Nth step renders as a continuous curve of its own progress instead
    of a handful of blocks scattered across blank cells. Bins that contain a
    NaN/Inf value are marked so the break is visible at a glance, and the
    value range is clipped to the 2nd/98th percentile so one giant spike
    cannot flatten the whole curve into the bottom block.
    """
    present = [v for v in values if v is not None]
    finite = sorted(v for v in present if math.isfinite(v))
    if not finite:
        return ""
    bins = max(1, min(bins, len(present)))
    lo = finite[max(0, len(finite) // 50)]
    hi = finite[min(len(finite) - 1, len(finite) - 1 - len(finite) // 50)]
    ramp = BLOCK_RAMPS if unicode_ok else ASCII_RAMPS
    span = hi - lo
    out: list[str] = []
    for b in range(bins):
        start = b * len(present) // bins
        end = (b + 1) * len(present) // bins or len(present)
        chunk = present[start:end]
        if any(not math.isfinite(v) for v in chunk):
            out.append("!")
            continue
        average = sum(chunk) / len(chunk)
        if span <= 0:
            level = len(ramp) - 1
        else:
            ratio = (average - lo) / span
            level = round(min(1.0, max(0.0, ratio)) * (len(ramp) - 1))
        out.append(ramp[level])
    return "".join(out)


def _fit_label(name: str, width: int, unicode_ok: bool) -> str:
    """Trim a series name to the label column, marking truncation."""
    if len(name) <= width:
        return name
    marker = "…" if unicode_ok else "~"
    return name[: width - 1] + marker


def _series_line(
    label: str,
    values: list[float | None],
    steps: list[int],
    *,
    palette: _Palette,
    unicode_ok: bool,
    show_best: bool,
    label_width: int,
) -> str:
    spark = _sparkline(values, unicode_ok=unicode_ok)
    present = [v for v in values if v is not None]
    head = _fmt_number(present[0] if present else None)
    tail = _fmt_number(present[-1] if present else None)
    arrow = "→" if unicode_ok else "->"
    bar = "│" if unicode_ok else "|"
    spark_field = f"{spark:<{SPARK_BINS}}"
    span = f"{head:>8} {arrow} {tail:>8}"
    parts = [
        f"  {_fit_label(label, label_width, unicode_ok):<{label_width}} "
        f"{palette.dim(bar)}{palette.dim(spark_field)}{palette.dim(bar)}  {span}"
    ]
    if show_best:
        best_index = min(
            (i for i, v in enumerate(values) if v is not None and math.isfinite(v)),
            key=lambda i: values[i],  # type: ignore[arg-type, return-value]
            default=None,
        )
        if best_index is not None:
            best = values[best_index]
            assert best is not None
            parts.append(palette.dim(f"best {_fmt_number(best)} @ step {steps[best_index]}"))
    return "  ".join(parts).rstrip()


def _finding_line(finding: Finding, palette: _Palette) -> str:
    color = {"error": palette.error, "warning": palette.warning, "info": palette.info}[
        finding.severity
    ]
    return f"  {color(f'{finding.severity:<8}')} {finding.check:<14} {finding.message}"


def render_report(
    reports: list[RunReport],
    *,
    fail_severity: str = "error",
    color: bool = False,
    unicode_ok: bool = True,
    version: str = "",
) -> str:
    """Render the lint report grouped by run, with one sparkline per loss series."""
    palette = _Palette(color)
    findings = [f for report in reports for f in report.findings]
    counts = {name: sum(1 for f in findings if f.severity == name) for name in SEVERITIES}

    header_bits = [
        palette.bold(f"losslint {version}".rstrip()),
        f"{len(reports)} run(s)",
    ]
    if findings:
        header_bits.append(palette.bold(f"{len(findings)} finding(s)"))
    lines = [" · ".join(header_bits), ""]

    names = [report.run.source.name for report in reports]
    unique_names = len(names) == len(set(names))

    label_width = LABEL_WIDTH_MIN
    for report in reports:
        for role in ("train", "eval"):
            entry = find_series(report.run, role)
            if entry is not None:
                label_width = max(label_width, min(LABEL_WIDTH_MAX, len(entry[0])))

    for report in reports:
        run = report.run
        label = run.source.name if unique_names else str(run.source)
        run_counts = {
            name: sum(1 for f in report.findings if f.severity == name) for name in SEVERITIES
        }
        tally = " · ".join(f"{run_counts[n]} {n}" for n in SEVERITIES if run_counts[n])
        status = tally if tally else "clean"
        styled_status = palette.clean(status) if not report.findings else status
        lines.append(f"{palette.bold(label)} · {len(run.steps)} points · {styled_status}")

        for role, show_best in (("train", False), ("eval", True)):
            entry = find_series(run, role)
            if entry is None:
                continue
            lines.append(
                _series_line(
                    entry[0],
                    entry[1],
                    run.steps,
                    palette=palette,
                    unicode_ok=unicode_ok,
                    show_best=show_best,
                    label_width=label_width,
                )
            )
        for finding in report.findings:
            lines.append(_finding_line(finding, palette))
        lines.append("")

    summary_bits = [f"{len(findings)} finding(s)"] if findings else []
    summary_bits += [f"{counts[n]} {n}" for n in SEVERITIES if counts[n]]
    threshold = SEVERITY_RANK[fail_severity]
    code = int(any(SEVERITY_RANK[f.severity] >= threshold for f in findings))
    summary_bits.append(f"exit {code} (fail-severity: {fail_severity})")
    sep = " · " if unicode_ok else " | "
    lines.append(palette.dim(sep.join(summary_bits)))
    return "\n".join(lines) + "\n"
