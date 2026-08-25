"""Command-line interface: ``losslint check``, ``losslint watch`` and ``losslint demo``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from losslint import __version__
from losslint.checks import CheckOptions, run_all_checks
from losslint.demo import write_demo_logs
from losslint.parsing import ColumnOverrides, load_run
from losslint.render import render_report
from losslint.report import SEVERITIES, RunReport, exit_code, render_json
from losslint.watch import WatchOptions, run_watch

# Directory-scan suffixes: every format load_run understands, matched against
# the filename (event files carry no fixed suffix).
DIR_SCAN_RULES = (
    ("*.jsonl",),
    ("*.csv",),
    ("trainer_state.json",),
    ("*tfevents*",),
)


def _expand_inputs(raw_inputs: list[str]) -> list[Path]:
    """Resolve files and directories into a concrete list of log files.

    Directories are scanned recursively for the supported formats (``*.jsonl``,
    ``*.csv``, ``trainer_state.json``, TensorBoard event files), skipping
    hidden folders — so ``losslint check runs/`` works on shells without
    glob expansion.
    """
    files: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw)
        if path.is_dir():
            for pattern in DIR_SCAN_RULES:
                files.extend(
                    p
                    for p in path.rglob(pattern[0])
                    if not p.is_dir() and not _under_hidden_dir(path, p)
                )
        else:
            files.append(path)
    # De-duplicate while preserving order (rglob patterns can overlap).
    return list(dict.fromkeys(files))


def _under_hidden_dir(root: Path, candidate: Path) -> bool:
    """True when the candidate sits inside a dot-directory (.git, .venv, ...)."""
    return any(part.startswith(".") for part in candidate.relative_to(root).parts[:-1])


def _stdout_supports_unicode() -> bool:
    """True when stdout can render block characters (UTF-8/UTF-16 consoles)."""
    encoding = (sys.stdout.encoding or "").lower()
    return "utf" in encoding


def _color_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="losslint",
        description="Lint training logs after the run: NaN, divergence, spikes, "
        "overfitting onset and step bugs in trainer_state.json / CSV / JSONL / "
        "TensorBoard event files.",
    )
    parser.add_argument("--version", action="version", version=f"losslint {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="lint one or more training log files")
    p_check.add_argument(
        "files",
        nargs="+",
        help="log files or directories (json/jsonl/csv/tfevents; "
        "directories are scanned recursively)",
    )
    _add_check_thresholds(p_check)
    _add_column_overrides(p_check)
    p_check.add_argument(
        "--format", choices=("text", "json"), default="text", help="report format (default: text)"
    )
    p_check.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI colors in text output; auto disables them when not a TTY or "
        "NO_COLOR is set (default: auto)",
    )
    p_check.add_argument("--json-out", type=Path, help="also write a JSON report here")

    p_watch = sub.add_parser(
        "watch",
        help="follow a live training log and lint it as it grows",
        description="Follow one training log file, re-lint on every change and write "
        "an evidence snapshot (series tails incl. lr/grad_norm, raw log tail, host "
        "state) whenever a new finding appears. Text formats are tailed "
        "incrementally; trainer_state.json / TensorBoard event files are reloaded "
        "in full on change.",
    )
    p_watch.add_argument("file", type=Path, help="the log file to watch")
    p_watch.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="seconds between polls (default: 5; text formats only parse new bytes)",
    )
    p_watch.add_argument(
        "--snapshot-dir",
        type=Path,
        help="directory for snapshots (default: <log dir>/losslint-snapshots)",
    )
    p_watch.add_argument(
        "--snapshot-cooldown",
        type=float,
        default=60.0,
        help="seconds to wait before re-snapshotting the same check (default: 60)",
    )
    p_watch.add_argument(
        "--snapshot-on",
        choices=SEVERITIES,
        default="warning",
        help="minimum severity that triggers a snapshot (default: warning)",
    )
    p_watch.add_argument(
        "--exit-on-finding",
        action="store_true",
        help="exit with code 1 as soon as a finding reaches --fail-severity "
        "(for scripted early-stop reactions)",
    )
    p_watch.add_argument(
        "--once",
        action="store_true",
        help="run a single poll pass and exit (lint the log as it stands now)",
    )
    p_watch.add_argument("--verbose", action="store_true", help="log every poll")
    _add_check_thresholds(p_watch)
    _add_column_overrides(p_watch)
    p_watch.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="ANSI colors in text output (default: auto)",
    )

    p_demo = sub.add_parser("demo", help="write deterministic demo logs to a directory")
    p_demo.add_argument("outdir", type=Path, help="output directory (created if missing)")
    return parser


def _add_check_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fail-severity",
        choices=SEVERITIES,
        default="error",
        help="exit 1 when a finding is at or above this severity (default: error)",
    )
    parser.add_argument(
        "--spike-factor", type=float, default=10.0, help="loss spike threshold multiplier"
    )
    parser.add_argument(
        "--divergence-ratio",
        type=float,
        default=1.25,
        help="last/first decile median ratio that counts as divergence",
    )


def _add_column_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--step-col", help="override the CSV step column name")
    parser.add_argument("--loss-col", help="override the CSV train-loss column name")
    parser.add_argument("--eval-loss-col", help="override the CSV eval-loss column name")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = _build_parser().parse_args(argv)

    if args.command == "demo":
        paths = write_demo_logs(args.outdir)
        print(f"wrote {len(paths)} demo log(s) to {args.outdir}:")
        for path in paths:
            print(f"  {path.name}")
        print(f"\ntry: losslint check {args.outdir}")
        return 0

    overrides = ColumnOverrides(
        step=args.step_col, loss=args.loss_col, eval_loss=args.eval_loss_col
    )
    options = CheckOptions(spike_factor=args.spike_factor, divergence_ratio=args.divergence_ratio)

    if args.command == "watch":
        watch_options = WatchOptions(
            interval=args.interval,
            snapshot_dir=args.snapshot_dir,
            snapshot_cooldown=args.snapshot_cooldown,
            snapshot_severity=args.snapshot_on,
            fail_severity=args.fail_severity,
            exit_on_finding=args.exit_on_finding,
            once=args.once,
            verbose=args.verbose,
        )
        try:
            return run_watch(args.file, watch_options, options, overrides)
        except (FileNotFoundError, ValueError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            return 0

    files = _expand_inputs(args.files)
    if not files:
        print("error: no log files found in the given path(s)", file=sys.stderr)
        return 2

    reports: list[RunReport] = []
    for file in files:
        try:
            run = load_run(file, overrides)
        except (FileNotFoundError, ValueError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        reports.append(RunReport(run=run, findings=run_all_checks(run, options)))

    findings = [f for report in reports for f in report.findings]
    code = exit_code(findings, args.fail_severity)

    json_text = render_json(reports, code)
    if args.format == "json":
        print(json_text, end="")
    else:
        print(
            render_report(
                reports,
                fail_severity=args.fail_severity,
                color=_color_enabled(args.color),
                unicode_ok=_stdout_supports_unicode(),
                version=__version__,
            ),
            end="",
        )
    if args.json_out:
        args.json_out.write_text(json_text, encoding="utf-8")
        print(f"wrote JSON report to {args.json_out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
