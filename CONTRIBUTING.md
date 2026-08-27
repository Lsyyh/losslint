# Contributing to losslint

Thank you for helping make training failures easier to diagnose.

## Development

Use Python 3.10 or newer, then install the project and its test extras:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

Please include a focused regression test for every behaviour change. Keep rules deterministic, describe heuristic conclusions as evidence rather than certainty, and avoid collecting or uploading training logs.

## Reporting a finding

Provide a minimal, sanitized log fixture, the command you ran, the expected result, and the actual result. Never include credentials, private dataset examples, or unredacted experiment metadata.
