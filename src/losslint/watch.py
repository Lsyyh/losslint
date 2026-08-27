"""Live monitoring: tail a growing training log and lint it as it grows.

Text formats (JSONL / CSV) are tailed incrementally — only bytes appended
since the last poll are parsed, so watching a million-step log costs a few
milliseconds per poll. Blob formats (``trainer_state.json``, TensorBoard
event files) are reloaded in full whenever size or mtime changes.

Whenever a *new* finding at or above the snapshot severity appears, a JSON
snapshot is written beside the log: the finding, the tail of every logged
series (lr, grad_norm, ... — whatever the trainer writes), the raw tail of
the log file, and the host state (load, memory, nvidia-smi when present).
That is the evidence you want when scrambling to explain a mid-run anomaly.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from losslint.checks import CheckOptions, run_all_checks
from losslint.parsing import (
    JSONL_STEP_CANDIDATES,
    ColumnOverrides,
    RunLog,
    _is_usable_number,
    _resolve_columns,
    _to_float,
    load_run,
    sniff_format,
)
from losslint.report import SEVERITY_RANK, Finding, exit_code

SNAPSHOT_DIR_NAME = "losslint-snapshots"
LOG_TAIL_MAX_BYTES = 16 * 1024
NVIDIA_SMI_TIMEOUT = 5.0


@dataclass
class WatchOptions:
    """Tunables for the watch loop."""

    interval: float = 5.0
    snapshot_dir: Path | None = None
    snapshot_cooldown: float = 60.0
    snapshot_severity: str = "warning"
    tail_lines: int = 40
    tail_values: int = 20
    fail_severity: str = "error"
    exit_on_finding: bool = False
    once: bool = False
    verbose: bool = False
    color: bool = False


# ---------------------------------------------------------------------------
# incremental tails


class _JsonlTail:
    """Incremental JSONL reader: parses only lines appended since last poll."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._carry = ""
        self._steps: list[int] = []
        self._series: dict[str, list[float | None]] = {}
        self._issues: list[str] = []
        self._step_key: str | None = None
        self._line_no = 0

    def poll(self) -> RunLog | None:
        with self._path.open("rb") as fh:
            fh.seek(self._offset)
            raw = fh.read()
        if not raw:
            return None
        self._offset += len(raw)
        text = self._carry + raw.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._carry = lines.pop()  # last fragment may be an unfinished line
        for line in lines:
            self._line_no += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                self._issues.append(f"line {self._line_no}: invalid JSON ({err.msg})")
                continue
            if not isinstance(record, dict):
                self._issues.append(
                    f"line {self._line_no}: expected a JSON object, got {type(record).__name__}"
                )
                continue
            self._feed(record)
        if not self._steps:
            return None
        return RunLog(
            steps=self._steps,
            series=self._series,
            source=self._path,
            parse_issues=self._issues,
        )

    def _feed(self, record: dict[str, Any]) -> None:
        if self._step_key is None and not self._steps:
            for candidate in JSONL_STEP_CANDIDATES:
                if candidate in record:
                    self._step_key = candidate
                    break
        step_value = _to_float(record.get(self._step_key)) if self._step_key else None
        if self._step_key is not None and not _is_usable_number(step_value):
            self._issues.append(
                f"line {self._line_no}: invalid step value {record.get(self._step_key)!r}"
            )
            return
        for key, raw in record.items():
            if key == self._step_key or not isinstance(raw, bool | int | float | str):
                continue
            if key not in self._series:
                self._series[key] = [None] * len(self._steps)
        if self._step_key is not None:
            assert step_value is not None
            self._steps.append(int(step_value))
        else:
            self._steps.append(len(self._steps))
        for name, values in self._series.items():
            values.append(_to_float(record.get(name)))


class _CsvTail:
    """Incremental CSV reader over a fixed header (columns resolved once)."""

    def __init__(self, path: Path, overrides: ColumnOverrides) -> None:
        self._path = path
        self._offset = 0
        self._carry = ""
        self._steps: list[int] = []
        self._issues: list[str] = []
        with path.open("rb") as fh:
            header_bytes = fh.readline()
            self._offset = len(header_bytes)
        if not header_bytes.strip():
            raise ValueError(f"empty CSV file: {path}")
        header = [name.strip() for name in header_bytes.decode("utf-8-sig").split(",")]
        step_col, loss_col, eval_col = _resolve_columns(header, overrides)
        self._step_idx = header.index(step_col) if step_col is not None else None
        roles: dict[int, str] = {}
        if loss_col is not None:
            roles[header.index(loss_col)] = "loss"
        if eval_col is not None:
            roles[header.index(eval_col)] = "eval_loss"
        # (header column index -> series values list); the canonical loss /
        # eval_loss names alias their source columns, mirroring _load_csv.
        self._columns: list[tuple[int, list[float | None]]] = []
        self._series: dict[str, list[float | None]] = {}
        for idx, name in enumerate(header):
            canonical = roles.get(idx, name)
            if idx == self._step_idx or canonical in self._series:
                continue
            values: list[float | None] = []
            self._series[canonical] = values
            self._columns.append((idx, values))

    def poll(self) -> RunLog | None:
        with self._path.open("rb") as fh:
            fh.seek(self._offset)
            raw = fh.read()
        if not raw:
            return None
        self._offset += len(raw)
        text = self._carry + raw.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._carry = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            cells = next(csv.reader([line]))
            if self._step_idx is not None:
                raw_step = cells[self._step_idx] if self._step_idx < len(cells) else None
                step_value = _to_float(raw_step)
                if _is_usable_number(step_value):
                    assert step_value is not None
                    self._steps.append(int(step_value))
                else:
                    self._issues.append(f"invalid step value {raw_step!r}")
                    continue
            else:
                self._steps.append(len(self._steps))
            for idx, values in self._columns:
                cell = cells[idx] if idx < len(cells) else None
                values.append(_to_float(cell))
        if not self._steps:
            return None
        return RunLog(
            steps=self._steps, series=self._series, source=self._path, parse_issues=self._issues
        )


class _BlobReload:
    """Full reload on change (trainer_state.json / TensorBoard event files)."""

    def __init__(self, path: Path, overrides: ColumnOverrides) -> None:
        self._path = path
        self._overrides = overrides
        self._stamp: tuple[int, int] | None = None

    def poll(self) -> RunLog | None:
        stat = self._path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._stamp:
            return None
        self._stamp = stamp
        return load_run(self._path, self._overrides)  # may raise mid-write; retried next poll


def make_tail(path: Path, overrides: ColumnOverrides | None = None) -> Any:
    """Build the incremental reader matching the file's format."""
    if not path.is_file():
        raise FileNotFoundError(f"log file does not exist: {path}")
    overrides = overrides or ColumnOverrides()
    fmt = sniff_format(path)
    if fmt == "jsonl":
        return _JsonlTail(path)
    if fmt == "csv":
        return _CsvTail(path, overrides)
    return _BlobReload(path, overrides)


# ---------------------------------------------------------------------------
# evidence snapshots


def system_snapshot() -> dict[str, Any]:
    """Best-effort host state; optional tools (psutil, nvidia-smi) enrich it."""
    info: dict[str, Any] = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        info["loadavg"] = list(os.getloadavg())
    except (OSError, AttributeError):  # Windows has no loadavg
        info["loadavg"] = None
    try:
        import psutil

        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        info["memory_percent"] = memory.percent
        info["memory_used_gb"] = round(memory.used / 1e9, 2)
        info["memory_total_gb"] = round(memory.total / 1e9, 2)
    except ImportError:
        pass
    gpus = _nvidia_smi()
    if gpus is not None:
        info["gpu"] = gpus
    return info


def _nvidia_smi() -> list[dict[str, str]] | None:
    query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT,
            stdin=subprocess.DEVNULL,  # nvidia-smi can block reading a dead console
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if proc.returncode != 0:
        return None
    keys = query.split(",")
    gpus: list[dict[str, str]] = []
    for line in proc.stdout.strip().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == len(keys):
            gpus.append(dict(zip(keys, cells, strict=True)))
    return gpus or None


def _log_tail(path: Path, lines: int) -> list[str]:
    """Last text lines of the log (bounded read); empty for binary formats."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - LOG_TAIL_MAX_BYTES))
        text = fh.read().decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


def _series_tail(run: RunLog, count: int) -> dict[str, dict[str, list[Any]]]:
    lo = max(0, len(run.steps) - count)
    return {
        name: {"steps": run.steps[lo:], "values": values[lo:]}
        for name, values in run.series.items()
    }


def write_snapshot(
    run: RunLog,
    finding: Finding,
    options: WatchOptions,
    *,
    is_text_log: bool,
) -> Path:
    """Write one JSON evidence snapshot; returns its path."""
    directory = options.snapshot_dir or run.source.parent / SNAPSHOT_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(run.source),
        "finding": {
            "check": finding.check,
            "severity": finding.severity,
            "message": finding.message,
            "details": finding.details,
        },
        "series_tail": _series_tail(run, options.tail_values),
        "log_tail": _log_tail(run.source, options.tail_lines) if is_text_log else [],
        "system": system_snapshot(),
    }
    path = directory / f"{stamp}_{finding.check}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# watch loop


@dataclass
class _WatchState:
    reported: set[str] = field(default_factory=set)
    last_snapshot_at: dict[str, float] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    warned_error: str | None = None


def run_watch(
    path: Path,
    options: WatchOptions,
    check_options: CheckOptions | None = None,
    overrides: ColumnOverrides | None = None,
    *,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Watch one log file; returns the process exit code.

    A watch is normally started before the trainer writes its first line, so
    a missing or still-empty file is not an error: the loop waits (and says
    so once) until the file exists and its format can be sniffed.
    ``--once`` keeps the strict behaviour and raises instead.
    """
    path = Path(path)
    if options.once and not path.is_file():
        raise FileNotFoundError(f"log file does not exist: {path}")
    state = _WatchState()
    tail = None
    is_text_log = True

    out(
        f"losslint watch · {path} · poll {options.interval:g}s"
        + (" · single pass" if options.once else " · Ctrl-C to stop")
    )
    while True:
        if tail is None:
            try:
                tail = make_tail(path, overrides)
                is_text_log = isinstance(tail, _JsonlTail | _CsvTail)
            except (FileNotFoundError, ValueError) as err:
                if options.once:
                    raise
                if state.warned_error != str(err):
                    state.warned_error = str(err)
                    out(f"[{datetime.now():%H:%M:%S}] waiting for a readable log: {err}")
                sleep(options.interval)
                continue
        try:
            run = tail.poll()
        except (ValueError, FileNotFoundError) as err:
            if isinstance(err, FileNotFoundError):
                tail = None  # file vanished (rotation?); re-sniff when it returns
            if state.warned_error != str(err):
                state.warned_error = str(err)
                out(f"[{datetime.now():%H:%M:%S}] waiting: {err}")
            run = None
        if run is not None:
            state.findings = run_all_checks(run, check_options)
            _announce_new(run, state, options, is_text_log=is_text_log, out=out)
        if options.exit_on_finding and exit_code(state.findings, options.fail_severity):
            return 1
        if options.once:
            break
        if options.verbose and run is not None:
            out(f"[{datetime.now():%H:%M:%S}] {len(run.steps)} points, still watching")
        sleep(options.interval)
    return exit_code(state.findings, options.fail_severity)


def _announce_new(
    run: RunLog,
    state: _WatchState,
    options: WatchOptions,
    *,
    is_text_log: bool,
    out: Callable[[str], None],
) -> None:
    for finding in state.findings:
        key = f"{finding.check}|{finding.message}"
        if key in state.reported:
            continue
        state.reported.add(key)
        stamp = datetime.now().strftime("%H:%M:%S")
        out(f"[{stamp}] {finding.severity} {finding.check}: {finding.message}")
        if SEVERITY_RANK[finding.severity] < SEVERITY_RANK[options.snapshot_severity]:
            continue
        now = time.monotonic()
        if now - state.last_snapshot_at.get(finding.check, float("-inf")) < (
            options.snapshot_cooldown
        ):
            continue
        state.last_snapshot_at[finding.check] = now
        snap = write_snapshot(run, finding, options, is_text_log=is_text_log)
        out(f"[{stamp}]   snapshot: {snap}")
