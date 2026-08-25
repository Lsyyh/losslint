"""Findings model, JSON renderer and exit-code computation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from losslint.parsing import RunLog

SEVERITIES = ("error", "warning", "info")
SEVERITY_RANK = {"info": 1, "warning": 2, "error": 3}


@dataclass(frozen=True)
class Finding:
    """One lint finding for one run."""

    check: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None


@dataclass
class RunReport:
    """One parsed run together with the findings it produced."""

    run: RunLog
    findings: list[Finding]


def exit_code(findings: list[Finding], fail_severity: str = "error") -> int:
    """Return 1 when any finding is at or above the failure threshold, else 0."""
    threshold = SEVERITY_RANK[fail_severity]
    return int(any(SEVERITY_RANK[f.severity] >= threshold for f in findings))


def _summary(findings: list[Finding]) -> dict[str, int]:
    return {name: sum(1 for f in findings if f.severity == name) for name in SEVERITIES}


def render_json(reports: list[RunReport], exit_code: int) -> str:
    """Render the machine-readable report (schema mirrored in README)."""
    from losslint import __version__

    findings = [f for report in reports for f in report.findings]
    payload = {
        "losslint_version": __version__,
        "runs": [
            {
                "source": str(report.run.source),
                "points": len(report.run.steps),
                "series": sorted(report.run.series),
                "findings": len(report.findings),
            }
            for report in reports
        ],
        "findings": [
            {
                "check": f.check,
                "severity": f.severity,
                "message": f.message,
                "details": f.details,
                "source": str(f.source) if f.source is not None else None,
            }
            for f in findings
        ],
        "summary": _summary(findings),
        "exit_code": exit_code,
    }
    return json.dumps(payload, indent=2)
