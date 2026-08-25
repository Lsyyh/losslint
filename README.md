# losslint

Lint your training logs — live or after the run. Point losslint at the metrics
file your trainer already writes (a HuggingFace `trainer_state.json`, a CSV of
`step,loss,lr`, a JSONL of metric dicts, or a TensorBoard event file) and get a
lint report: NaN poisoning, divergence, loss spikes, overfitting onset, and
step-logging bugs — with CI-friendly exit codes. No instrumentation, no cloud,
CPU-only.

## Install

```bash
pip install losslint                  # zero-runtime-dependency core
pip install "losslint[fast]"          # + numpy-accelerated checks
pip install "losslint[tb]"            # + TensorBoard event file support
```

Requires Python ≥ 3.10.

## Quick start

```bash
losslint check runs/                        # lint every supported log under a directory
losslint check metrics.csv --loss-col l --eval-loss-col e
losslint watch runs/exp1/metrics.jsonl --interval 5   # tail a live run
losslint demo demo_logs && losslint check demo_logs   # try the planted-defect demos
```

Text report (fixed-grid sparklines with a flat baseline):

```
losslint 0.4.1 · 1 run(s) · 1 finding(s)

nan.jsonl · 40 points · 1 error
  loss  │█▇▇▆▆▅▅▄▄▄▃▃▃▃▂▂▂▂!▂▁▁▁▁│       1.8 →    0.243
  error    nan_inf   series 'loss' has 1 NaN/Inf value(s) (first steps: 30) — every metric after this is suspect

1 finding(s) · 1 error · exit 1 (fail-severity: error)
```

The `!` marks a NaN/Inf break before you read a single finding.

## Checks

| check | severity | what it flags |
| --- | --- | --- |
| `nan_inf` | error | NaN/Inf in any numeric series |
| `divergence` | error | train loss ends materially higher than it started |
| `loss_spike` | warning | isolated point above 10× the local rolling median |
| `step_backtrack` | warning | strictly decreasing step indices |
| `overfit_onset` | info | eval loss worsening while train loss still improves |
| `stagnation` | info | < 1% train-loss improvement over the last quarter |
| `parse` | warning | malformed log lines skipped |

Exit codes: `0` clean, `1` findings at/above `--fail-severity`, `2` usage/parse error.

## In CI

```yaml
- run: pip install losslint && losslint check runs/ --fail-severity error
```

Or use the bundled action:

```yaml
- uses: Lsyyh/losslint@v0.4
  with:
    files: runs/
```

## License

MIT
