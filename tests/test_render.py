from __future__ import annotations

import json
from pathlib import Path

from losslint.parsing import RunLog
from losslint.render import render_github, render_markdown
from losslint.report import Finding, RunReport, render_json


def _report() -> RunReport:
    run = RunLog(steps=[1], series={"loss": [1.0]}, source=Path("run|one.csv"))
    finding = Finding("parse", "warning", "bad | value\nnext", source=run.source)
    return RunReport(run=run, findings=[finding])


def test_json_report_has_versioned_schema():
    assert json.loads(render_json([_report()], 0))["schema_version"] == 1


def test_github_renderer_escapes_workflow_command_content():
    text = render_github([_report()])
    assert "::warning file=run|one.csv,title=losslint parse::bad | value%0Anext" in text


def test_markdown_renderer_escapes_cells():
    text = render_markdown([_report()])
    assert "run\\|one.csv" in text
    assert "bad \\| value<br>next" in text
