from __future__ import annotations

from losslint.parsing import load_run
from losslint.watch import make_tail


def test_csv_infinite_step_is_parse_issue_not_overflow(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("Step,Loss\n1,1.0\ninf,0.9\n2,0.8\n", encoding="utf-8")

    run = load_run(path)

    assert run.steps == [1, 2]
    assert "invalid step value 'inf'" in run.parse_issues[0]


def test_jsonl_infinite_step_is_parse_issue_not_overflow(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "loss": 1.0}\n{"step": 1e999, "loss": 0.9}\n', encoding="utf-8")

    run = load_run(path)

    assert run.steps == [1]
    assert "invalid step value inf" in run.parse_issues[0]


def test_csv_column_detection_is_case_insensitive(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("GLOBAL_STEP,TRAIN_LOSS,VAL_LOSS\n1,1.0,1.2\n", encoding="utf-8")

    run = load_run(path)

    assert run.steps == [1]
    assert run.series["loss"] == [1.0]
    assert run.series["eval_loss"] == [1.2]


def test_watch_tail_skips_infinite_jsonl_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"global_step": 1, "loss": 1.0}\n{"global_step": 1e999, "loss": 0.9}\n')

    run = make_tail(path).poll()

    assert run is not None
    assert run.steps == [1]
    assert "invalid step value inf" in run.parse_issues[0]
