# losslint

训练日志事后审计 linter：指向 trainer 已经写出的指标文件（HuggingFace
`trainer_state.json`、CSV、`JSONL`、TensorBoard event 文件），即可拿到一份 lint
报告——NaN 污染、发散、loss 尖峰、过拟合起点、step 记录异常，并带 CI 友好的退出码。
无需埋点、无需上云、纯 CPU。

## 安装

```bash
pip install losslint                  # 核心，零运行时依赖
pip install "losslint[fast]"          # + numpy 加速
pip install "losslint[tb]"            # + TensorBoard 支持
```

需要 Python ≥ 3.10。

## 快速开始

```bash
losslint check runs/                        # 递归检查目录下所有支持的日志
losslint check metrics.csv --loss-col l --eval-loss-col e
losslint watch runs/exp1/metrics.jsonl --interval 5   # 实时监控正在写入的日志
losslint demo demo_logs && losslint check demo_logs   # 试用内置缺陷样例
```

文本报告采用固定宽度网格，曲线下缘为齐平基线：

```
losslint 0.4.1 · 1 run(s) · 1 finding(s)

nan.jsonl · 40 points · 1 error
  loss  │█▇▇▆▆▅▅▄▄▄▃▃▃▃▂▂▂▂!▂▁▁▁▁│       1.8 →    0.243
  error    nan_inf   series 'loss' has 1 NaN/Inf value(s) (first steps: 30) — every metric after this is suspect

1 finding(s) · 1 error · exit 1 (fail-severity: error)
```

`!` 标记 NaN/Inf 断点，读数前即可看见。

## 检查项

| check | 级别 | 触发条件 |
| --- | --- | --- |
| `nan_inf` | error | 任意数值序列出现 NaN/Inf |
| `divergence` | error | train loss 末端显著高于起点 |
| `loss_spike` | warning | 孤立点高于局部滚动中位数的 10× |
| `step_backtrack` | warning | step 严格递减（resume/日志 bug） |
| `overfit_onset` | info | eval loss 转差而 train loss 仍在下降 |
| `stagnation` | info | 末段 train loss 改善 < 1% |
| `parse` | warning | 跳过的非法日志行 |

退出码：`0` 正常、`1` 达到/超过 `--fail-severity`、`2` 用法/解析错误。

## 在 CI 中使用

```yaml
- run: pip install losslint && losslint check runs/ --fail-severity error
```

或使用内置 Action：

```yaml
- uses: Lsyyh/losslint@v0.4
  with:
    files: runs/
```

## 许可证

MIT
