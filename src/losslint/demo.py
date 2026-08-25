"""Deterministic synthetic training logs with planted defects for demos and tests."""

from __future__ import annotations

import json
from pathlib import Path

DEMO_FILES = (
    "healthy.csv",
    "nan.jsonl",
    "diverging.csv",
    "spike.jsonl",
    "trainer_state.json",
)

TB_DEMO_FILE = "events.out.tfevents.1700000000.losslint-demo"


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _healthy_csv() -> str:
    lines = ["step,loss,eval_loss"]
    eval_value = 1.9
    for i in range(60):
        loss = 2.0 * 0.96**i + 0.02
        eval_cell = _fmt(eval_value) if i % 6 == 5 else ""
        if i % 6 == 5:
            eval_value *= 0.9
        lines.append(f"{i * 10},{_fmt(loss)},{eval_cell}")
    return "\n".join(lines) + "\n"


def _nan_jsonl() -> str:
    rows = []
    for i in range(40):
        loss = float("nan") if i == 30 else 1.8 * 0.95**i
        rows.append(json.dumps({"step": i, "loss": loss}))
    return "\n".join(rows) + "\n"


def _diverging_csv() -> str:
    lines = ["step,loss"]
    for i in range(40):
        lines.append(f"{i * 10},{_fmt(1.0 + 0.03 * i)}")
    return "\n".join(lines) + "\n"


def _spike_jsonl() -> str:
    rows = []
    for i in range(30):
        loss = 25.0 if i == 15 else 1.5 * 0.97**i
        rows.append(json.dumps({"step": i, "loss": loss}))
    return "\n".join(rows) + "\n"


def _trainer_state() -> str:
    history: list[dict] = []
    eval_schedule = {4 * (k + 1): k for k in range(12)}
    for i in range(50):
        history.append({"step": i, "loss": 2.0 * 0.95**i, "lr": 1e-3})
        if i in eval_schedule:
            k = eval_schedule[i]
            eval_loss = 1.9 * 0.93**k if k < 8 else [1.2, 1.35, 1.55, 1.8][k - 8]
            # one mislogged step planted mid-history: fires step_backtrack
            logged_step = 7 if i == 40 else i
            history.append({"step": logged_step, "eval_loss": eval_loss})
    return json.dumps({"log_history": history, "best_metric": 1.147}, indent=None)


_BUILDERS = {
    "healthy.csv": _healthy_csv,
    "nan.jsonl": _nan_jsonl,
    "diverging.csv": _diverging_csv,
    "spike.jsonl": _spike_jsonl,
    "trainer_state.json": _trainer_state,
}


def _write_tb_demo(outdir: Path) -> Path | None:
    """Write a TensorBoard event log with a planted NaN; None without tensorboard."""
    try:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
    except ImportError:
        return None

    staging = outdir / "_tb_staging"
    staging.mkdir(exist_ok=True)
    writer = EventFileWriter(str(staging))
    for step in range(40):
        event = Event(wall_time=1700000000.0 + step, step=step)
        loss = float("nan") if step == 30 else 1.8 * 0.95**step
        event.summary.value.add(tag="train/loss", simple_value=loss)
        event.summary.value.add(tag="Loss/val", simple_value=1.9 * 0.93 ** (step // 6))
        writer.add_event(event)
    writer.flush()
    writer.close()
    target = outdir / TB_DEMO_FILE
    next(staging.iterdir()).replace(target)
    staging.rmdir()
    return target


def write_demo_logs(outdir: Path) -> list[Path]:
    """Write the demo logs and return their paths.

    The five text logs are always written; the TensorBoard event demo is added
    when the optional ``losslint[tb]`` extra is installed.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in DEMO_FILES:
        path = outdir / name
        path.write_text(_BUILDERS[name](), encoding="utf-8")
        paths.append(path)
    tb_path = _write_tb_demo(outdir)
    if tb_path is not None:
        paths.append(tb_path)
    return paths
