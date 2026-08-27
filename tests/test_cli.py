from __future__ import annotations

import json

from losslint.cli import main


def test_json_stdout_stays_valid_when_writing_report(tmp_path, capsys):
    path = tmp_path / "metrics.csv"
    output = tmp_path / "nested" / "report.json"
    path.write_text("step,loss\n1,1\n2,0.9\n", encoding="utf-8")

    assert main(["check", str(path), "--format", "json", "--json-out", str(output)]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["schema_version"] == 1
    assert "wrote JSON report" in captured.err
    assert json.loads(output.read_text(encoding="utf-8"))["runs"][0]["points"] == 2
