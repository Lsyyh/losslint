# losslint

[![CI](https://github.com/Lsyyh/losslint/actions/workflows/ci.yml/badge.svg)](https://github.com/Lsyyh/losslint/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/losslint/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Lint your training logs — live or after the run.** Point it at the metrics
file your trainer *already writes* — a HuggingFace `trainer_state.json`, a CSV
of `step,loss,lr`, a JSONL of metric dicts, or a TensorBoard event file — and
get a lint report: NaN poisoning, divergence, loss spikes, overfitting onset
and step-logging bugs, with CI-friendly exit codes. No instrumentation, no
cloud, CPU-only.

```
$ losslint check examples/demo_logs/nan.jsonl
losslint 0.4.1 · 1 run(s) · 1 finding(s)

nan.jsonl · 40 points · 1 error
  loss   │█▇▇▆▆▅▅▄▄▄▃▃▃▃▂▂▂▂!▂▁▁▁▁│       1.8 →    0.243
  error    nan_inf        series 'loss' has 1 NaN/Inf value(s) (first steps: 30) — every metric after this is suspect

1 finding(s) · 1 error · exit 1 (fail-severity: error)
```

Every series row renders into the same fixed-width grid — label, a framed
24-cell sparkline, right-aligned first → last values — so train and eval
curves line up however sparsely eval is logged. The `!` above is the NaN
break, visible before you read a single finding. The sparkline uses bottom-aligned
Unicode block characters, so the lower edge of every curve is a flat, aligned
baseline; on Windows the console is switched to the UTF-8 code page automatically
so blocks render there too. Colors are enabled on TTYs only
(`--color auto|always|never`, honors `NO_COLOR`); the few non-UTF-8 streams
that cannot be reconfigured fall back to an ASCII sparkline whose lower edge
still stays flat.

## Why

Training runs fail silently, and the evidence lands in logs nobody reads
closely:

- a NaN appears mid-run and every later number is garbage — HuggingFace TRL
  [issue #6702](https://github.com/huggingface/trl/issues/6702) shows NaN steps
  can be *invisible* in `log_history`: the plotted curve still looks plausible;
- the run diverges after a scheduler hiccup or bad data shard and burns
  GPU-hours overnight;
- an isolated loss spike (see [*Loss Spike in Training Neural Networks*](
  https://arxiv.org/abs/2305.12133)) corrupts a checkpoint that later
  "mysteriously" underperforms;
- eval loss turns upward while train loss keeps falling — the best checkpoint
  is already behind you;
- resume bugs duplicate or rewind the step index.

Dashboards (TensorBoard, W&B) only help if you watch them. `terminate_on_nan`
only helps during the run, and only for NaN. **losslint audits the artifact
that already exists** — after the run, from CI, or on a borrowed laptop — and
gives you a verdict, not a chart. And since v0.4 it can also *tail* that
artifact while the run is live: `losslint watch` re-lints on every change and
captures an evidence snapshot the moment something goes wrong.

## Install

```bash
pip install losslint                    # once published on PyPI
pip install "losslint[fast]"            # + numpy-accelerated checks
pip install "losslint[tb]"              # + TensorBoard event file support
pip install -e ".[dev]"                 # development, from a clone
```

Requires Python ≥ 3.10. **Zero runtime dependencies** — the text-format checks
are stdlib-only. Optional extras: `[fast]` vectorizes the checks with numpy
(~2x on the checks stage of a 500k-point log; identical findings either way),
`[tb]` reads TensorBoard event files (`tensorboard`, no TensorFlow required).

## Usage

### Check logs

```bash
losslint check runs/*/trainer_state.json      # HF Trainer output
losslint check metrics.csv                    # any step/loss CSV
losslint check events.jsonl                   # one metric dict per line
losslint check runs/                          # a directory: scanned recursively
losslint check runs/tb/events.out.tfevents.*  # TensorBoard logs (with [tb])
```

Format is sniffed automatically (TensorBoard files are recognized by their
binary TFRecord header, so renamed files work too); CSV columns are
auto-detected (`step`/`global_step`/`iteration`/`epoch`, `loss`/`train_loss`,
`eval_loss`/`val_loss`) and can be overridden:

```bash
losslint check metrics.csv --step-col t --loss-col l --eval-loss-col e
```

Directory inputs pick up `*.jsonl`, `*.csv`, `trainer_state.json` and
TensorBoard event files (hidden folders are skipped) — handy on shells
without glob expansion.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--fail-severity` | `error` | exit 1 at/above this severity (`error`/`warning`/`info`) |
| `--spike-factor` | `10.0` | loss-spike threshold vs local rolling median |
| `--divergence-ratio` | `1.25` | last/first-decile median ratio counted as divergence |
| `--format` | `text` | `text` (sparkline report) or `json` (schema below) on stdout |
| `--color` | `auto` | ANSI colors: auto/always/never (auto = TTY and no `NO_COLOR`) |
| `--json-out report.json` | — | also write a machine-readable report |

Exit codes: `0` clean, `1` findings at/above the threshold, `2` usage/parse
error — designed for CI.

### Watch a live run

```bash
losslint watch runs/exp1/metrics.jsonl --interval 5
```

`watch` follows one log file while the trainer writes it, re-lints on every
change, and prints each *new* finding with a timestamp. Whenever a new finding
reaches the snapshot severity (`--snapshot-on`, default `warning`) it writes an
**evidence snapshot** to `<log dir>/losslint-snapshots/`:

```json
{
  "finding":      { "check": "loss_spike", "severity": "warning", "...": "..." },
  "series_tail":  { "loss": {"steps": "...", "values": "..."},
                    "lr": {"steps": "...", "values": "..."},
                    "grad_norm": {"steps": "...", "values": "..."} },
  "log_tail":     ["... the last 40 raw lines of the log ..."],
  "system":       { "hostname": "...", "cpu_percent": "...", "gpu": ["nvidia-smi query"] }
}
```

That is the context you otherwise lose by the time you notice the anomaly:
what the learning rate and gradient norm looked like, what the raw log said,
and what the host (and GPUs, when `nvidia-smi` exists) were doing — captured
at the moment of detection, ready for post-mortem.

Text formats (JSONL / CSV) are tailed incrementally — only appended bytes are
parsed, so a poll costs milliseconds even on a million-step log.
`trainer_state.json` and TensorBoard event files are reloaded in full when
they change. Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--interval` | `5` | seconds between polls |
| `--snapshot-dir` | `<log dir>/losslint-snapshots` | where snapshots go |
| `--snapshot-cooldown` | `60` | seconds before the same check re-snapshots |
| `--snapshot-on` | `warning` | minimum severity that triggers a snapshot |
| `--exit-on-finding` | off | exit 1 as soon as `--fail-severity` is reached (scripted early stop) |
| `--once` | off | single poll pass, then exit (lint the log as it stands) |

`--once` makes `watch` scriptable: run it from cron, a wrapper script, or CI
against a still-growing log without staying attached.

### Run it in CI

One step in a GitHub Actions workflow:

```yaml
- run: pip install losslint && losslint check runs/ --fail-severity error
```

Or use the bundled action (repo root `action.yml`, needs the PyPI release):

```yaml
- uses: Lsyyh/losslint@v0.2
  with:
    files: runs/
```

### Generate demo logs

```bash
losslint demo examples/demo_logs
losslint check examples/demo_logs
```

Six deterministic logs with planted defects:

| file | planted defect | detected as |
|---|---|---|
| `healthy.csv` | none | clean |
| `nan.jsonl` | NaN at step 30 | `nan_inf` (error) |
| `diverging.csv` | loss climbs 1.0 → 2.2 | `divergence` (error) |
| `spike.jsonl` | 25.0 spike at step 15 | `loss_spike` (warning) |
| `trainer_state.json` | eval upturn + mislogged step | `overfit_onset` (info), `step_backtrack` (warning) |
| `events.out.tfevents…` | NaN in a TensorBoard log | `nan_inf` (error; needs `[tb]`) |

## Checks

| check | severity | what it flags |
|---|---|---|
| `nan_inf` | error | NaN/Inf in any numeric series — poisons later checkpoints |
| `divergence` | error | train loss ends materially higher than it started (last-decile median > ratio × first-decile median) |
| `loss_spike` | warning | isolated points above `--spike-factor` × the local rolling median |
| `step_backtrack` | warning | strictly decreasing step indices (resume/logging bugs; duplicates are allowed — HF logs interleave train/eval entries at the same step) |
| `overfit_onset` | info | eval loss worsening for ≥ 3 consecutive evals while train loss still improves |
| `stagnation` | info | < 1 % train-loss improvement over the last quarter |
| `parse` | warning | malformed log lines skipped while reading |

## Architecture

```
src/losslint/
├── parsing.py   # format sniffing + HF trainer_state / JSONL / CSV /
│                #   TensorBoard event loaders → RunLog
├── checks.py    # pure functions RunLog → list[Finding] (thresholds in CheckOptions)
├── accel.py     # optional numpy fast paths (stdlib fallbacks return identical results)
├── report.py    # Finding/RunReport models, JSON renderer, exit-code computation
├── render.py    # terminal renderer: fixed-grid sparklines, ANSI colors
├── watch.py     # incremental tailing, new-finding detection, evidence snapshots
├── demo.py      # deterministic synthetic logs with planted defects
└── cli.py       # argparse wiring: check | watch | demo
```

`RunLog` holds a step axis plus named numeric series aligned to it (`None`
where an entry lacks the key), so interleaved HF `log_history` entries and
TensorBoard tags with different logging cadences fall out naturally. Checks
are pure and independently testable; new checks are one function added to
`ALL_CHECKS`.

## Development

```bash
pip install -e ".[dev]"
losslint demo demo_logs && losslint check demo_logs   # smoke round trip
ruff check . && ruff format --check .
```

## Roadmap

- regex-based column mapping for exotic text logs
- oscillation / non-convergence check (a high-LR run that never descends is
  currently surfaced as `loss_spike` + `stagnation`, not as divergence)
- numeric-metric checks (accuracy/F1 trends), NaN-corrupted checkpoint detection
- W&B export / MLflow artifact input

## Contributing

Bug reports and new checks are welcome — a check is a pure function over a
parsed `RunLog` (see `checks.py`). Please open an issue before large changes.

## License

[MIT](LICENSE)
