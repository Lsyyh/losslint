"""Parse training log artifacts into RunLogs.

Supported inputs: HF ``trainer_state.json``, metric JSONL, step/loss CSV and
TensorBoard event files (with the optional ``losslint[tb]`` extra).
"""

from __future__ import annotations

import csv
import json
import math
import struct
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

STEP_CANDIDATES = ("step", "steps", "global_step", "iteration", "iter", "epoch")
JSONL_STEP_CANDIDATES = ("step", "steps", "global_step", "iteration")
LOSS_CANDIDATES = ("loss", "train_loss", "training_loss")
EVAL_LOSS_CANDIDATES = ("eval_loss", "val_loss", "validation_loss", "test_loss")
EVAL_HINTS = ("eval", "val", "test")


@dataclass(frozen=True)
class ColumnOverrides:
    """Explicit column names that bypass CSV auto-detection."""

    step: str | None = None
    loss: str | None = None
    eval_loss: str | None = None


@dataclass
class RunLog:
    """One parsed training log: step axis plus named numeric series.

    Series values are aligned to ``steps``; entries where a series was absent
    hold ``None``.
    """

    steps: list[int]
    series: dict[str, list[float | None]]
    source: Path
    parse_issues: list[str] = field(default_factory=list)


def _to_float(value: Any) -> float | None:
    """Coerce a scalar to float; None for non-numeric input (NaN/Inf pass through)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _is_usable_number(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


TFRECORD_MAX_FIRST_RECORD = 4096
SNIFF_PREFIX_BYTES = 64 * 1024


def _looks_like_tfrecord(path: Path) -> bool:
    """Heuristic TFRecord detection via the leading little-endian length header.

    Every TFRecord frame starts with a uint64 record length; for an event file
    the first frame (the file-version record) is tiny. Text formats decode to
    huge values here (every ASCII byte is >= 0x20), so the check is reliable.
    """
    with path.open("rb") as fh:
        header = fh.read(8)
    if len(header) < 8:
        return False
    (length,) = struct.unpack("<Q", header)
    if length <= 0 or length > TFRECORD_MAX_FIRST_RECORD:
        return False
    try:
        return path.stat().st_size >= 12 + length
    except OSError:
        return False


def sniff_format(path: Path) -> str:
    """Return "trainer_state", "jsonl", "csv" or "tfevent" for a log file.

    Only a bounded prefix is read: ``trainer_state.json`` files of long runs
    can reach hundreds of MB, and sniffing must not cost a second full read on
    top of the loader's own.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if _looks_like_tfrecord(path):
        return "tfevent"
    with path.open("rb") as fh:
        prefix = fh.read(SNIFF_PREFIX_BYTES)
    text = prefix.decode("utf-8", errors="replace").lstrip()
    if text.startswith("["):
        return "jsonl"
    if suffix == ".json":
        return "trainer_state"
    if not text:
        raise ValueError(f"cannot sniff format of empty file: {path}")
    first_line = text.split("\n", 1)[0].strip()
    if first_line.startswith("{"):
        if first_line.endswith("}"):
            try:
                first = json.loads(first_line)
            except json.JSONDecodeError:
                return "csv"
            if not isinstance(first, dict):
                return "jsonl"
            return "trainer_state" if "log_history" in first else "jsonl"
        # The first line did not terminate within the prefix: a single-line
        # JSON document. A log_history key anywhere in the head means
        # trainer_state; otherwise treat it as one very long JSONL record.
        return "trainer_state" if '"log_history"' in text else "jsonl"
    return "csv"


def _series_from_records(
    records: list[dict], step_key_candidates: tuple[str, ...] | None, source: Path
) -> RunLog:
    """Build a RunLog from an ordered list of dict records (JSONL/log_history)."""
    issues: list[str] = []
    kept: list[dict[str, Any]] = []
    for line_no, record in enumerate(records, start=1):
        if "step" in record and _to_float(record["step"]) is None:
            issues.append(f"line {line_no}: unparseable step value {record['step']!r}")
            continue
        kept.append(record)

    if not kept:
        raise ValueError(f"no usable rows in {source}")

    step_key: str | None = None
    if step_key_candidates is not None:
        for candidate in step_key_candidates:
            if candidate in kept[0]:
                step_key = candidate
                break

    steps: list[int] = []
    for index, record in enumerate(kept):
        if step_key is not None and _is_usable_number(_to_float(record.get(step_key))):
            steps.append(int(_to_float(record[step_key])))
        else:
            steps.append(index)

    series: dict[str, list[float | None]] = {}
    candidate_names: dict[str, None] = {}  # ordered set of numeric-looking keys
    for record in kept:
        for key, raw in record.items():
            if key == step_key:
                continue
            if isinstance(raw, bool | int | float | str):
                candidate_names[key] = None  # lists/dicts/nulls are not numeric series
    for name in candidate_names:
        values = [_to_float(record.get(name)) for record in kept]
        if any(v is not None for v in values):
            series[name] = values

    return RunLog(steps=steps, series=series, source=source, parse_issues=issues)


def _load_trainer_state(path: Path) -> RunLog:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("log_history")
    if not isinstance(history, list):
        raise ValueError(f"expected a 'log_history' list inside {path}")
    run = _series_from_records(
        [entry for entry in history if isinstance(entry, dict)], ("step",), path
    )
    return run


def _load_jsonl(path: Path) -> RunLog:
    records: list[dict] = []
    issues: list[str] = []
    # Stream line by line: JSONL logs of long runs can be large, and a single
    # read_text().splitlines() would double the peak memory for no benefit.
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as err:
                issues.append(f"line {line_no}: invalid JSON ({err.msg})")
                continue
            if not isinstance(entry, dict):
                issues.append(f"line {line_no}: expected a JSON object, got {type(entry).__name__}")
                continue
            records.append(entry)
    run = _series_from_records(records, JSONL_STEP_CANDIDATES, path)
    run.parse_issues = issues + run.parse_issues
    return run


def _load_tfevent(path: Path) -> RunLog:
    """Load scalar series from a TensorBoard event file.

    Keeps first-seen (wall-clock) step order, so step regressions in the file
    still surface through the ``step_backtrack`` check. Both the legacy
    ``simple_value`` field (torch ``SummaryWriter``) and the tensor format
    with scalar plugin metadata (TF2/Keras) are read.
    """
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        from tensorboard.util.tensor_util import make_ndarray
    except ImportError as err:  # pragma: no cover - exercised via monkeypatched import
        raise ValueError(
            f"reading TensorBoard event files requires the optional extra: "
            f"pip install 'losslint[tb]' ({path})"
        ) from err

    def scalar_of(value: Any) -> float | None:
        if value.HasField("simple_value"):
            return float(value.simple_value)
        if value.HasField("tensor"):
            try:
                array = make_ndarray(value.tensor)
            except Exception:
                return None
            if array.size != 1:
                return None
            return float(array.reshape(-1)[0])
        return None

    steps: list[int] = []
    seen_steps: set[int] = set()
    points: dict[str, dict[int, float]] = {}
    for event in EventFileLoader(str(path)).Load():
        step = int(event.step)
        if step not in seen_steps:
            seen_steps.add(step)
            steps.append(step)
        for value in event.summary.value:
            number = scalar_of(value)
            if number is None:
                continue  # histograms/images/other non-scalar summaries
            points.setdefault(value.tag, {})[step] = number

    if not points:
        raise ValueError(f"no scalar series found in TensorBoard event file: {path}")
    series = {tag: [mapped.get(step) for step in steps] for tag, mapped in points.items()}
    return RunLog(steps=steps, series=series, source=path)


def _resolve_columns(
    header: list[str], overrides: ColumnOverrides
) -> tuple[str | None, str, str | None]:
    for role in ("step", "loss", "eval_loss"):
        override = getattr(overrides, role)
        if override is not None and override not in header:
            raise ValueError(f"override column {override!r} not found in header {header}")

    def find(
        candidates: tuple[str, ...],
        *,
        contains: str | None = None,
        exclude_hints: bool = False,
        require_hint: bool = False,
    ) -> str | None:
        for candidate in candidates:
            if candidate in header:
                return candidate
        if contains is not None:
            for name in header:
                lowered = name.lower()
                if contains not in lowered:
                    continue
                has_hint = any(h in lowered for h in EVAL_HINTS)
                if (exclude_hints and has_hint) or (require_hint and not has_hint):
                    continue
                return name
        return None

    step_col = overrides.step or find(STEP_CANDIDATES)
    loss_col = overrides.loss or find(LOSS_CANDIDATES, contains="loss", exclude_hints=True)
    eval_col = overrides.eval_loss or find(EVAL_LOSS_CANDIDATES, contains="loss", require_hint=True)
    if loss_col is not None and eval_col == loss_col:
        eval_col = None  # a lone train-loss column must not double as eval loss
    if loss_col is None and eval_col is None:
        raise ValueError(
            f"no loss-like column found in header {header}; use --loss-col to specify one"
        )
    return step_col, loss_col, eval_col  # type: ignore[return-value]


def _load_csv(path: Path, overrides: ColumnOverrides) -> RunLog:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration:
            raise ValueError(f"empty CSV file: {path}") from None
        rows = [row for row in reader if row]

    step_col, loss_col, eval_col = _resolve_columns(header, overrides)
    step_idx = header.index(step_col) if step_col is not None else None

    issues: list[str] = []
    steps: list[int] = []
    kept_rows: list[list[str]] = []
    for row_no, row in enumerate(rows, start=2):
        if step_idx is not None:
            raw = row[step_idx] if step_idx < len(row) else None
            step_value = _to_float(raw)
            if step_value is None:
                issues.append(f"row {row_no}: unparseable step value {raw!r}")
                continue
            steps.append(int(step_value))
        else:
            steps.append(len(steps))
        kept_rows.append(row)

    if not kept_rows:
        raise ValueError(f"no usable rows in {path}")

    # Column-major parse: one transpose instead of one dict per row, and
    # float coercion runs per column in a single list comprehension. Rows can
    # be shorter than the header, so pad missing columns before the zip.
    columns = list(zip_longest(*kept_rows, fillvalue=None))[: len(header)]
    columns.extend([()] * (len(header) - len(columns)))
    parsed = {
        name: [_to_float(cell) for cell in column]
        for name, column in zip(header, columns, strict=True)
    }

    series: dict[str, list[float | None]] = {}
    if loss_col is not None:
        series["loss"] = parsed[loss_col]
    if eval_col is not None:
        series["eval_loss"] = parsed[eval_col]
    for name in header:
        if name in (step_col, loss_col, eval_col) or name in series:
            continue
        values = parsed[name]
        if all(v is None for v in values):
            continue  # fully non-numeric column (labels etc.)
        series[name] = values

    return RunLog(steps=steps, series=series, source=path, parse_issues=issues)


def load_run(path: Path, overrides: ColumnOverrides | None = None) -> RunLog:
    """Load one log file into a RunLog, sniffing the format when needed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"log file does not exist: {path}")
    fmt = sniff_format(path)
    if fmt == "trainer_state":
        return _load_trainer_state(path)
    if fmt == "jsonl":
        return _load_jsonl(path)
    if fmt == "tfevent":
        return _load_tfevent(path)
    return _load_csv(path, overrides or ColumnOverrides())
