# Roadmap

losslint is a local, deterministic linter for machine-learning training logs.

## Next

- Add `.losslint.toml` for rule thresholds, severity overrides, aliases, and ignored paths.
- Compare a candidate run with a baseline using stable, documented metrics.
- Publish sanitized fixtures for Hugging Face, Lightning, Keras, and MMEngine.
- Gather false-positive and missed-failure examples to calibrate conservative defaults.

New rules should state their evidence and known limitations; they should not claim to identify a root cause without verification.
