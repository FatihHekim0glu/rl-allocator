# Contributing to rl-allocator

Thanks for your interest. This project is a research-grade, honest-NULL benchmark; the
bar is rigour and reproducibility, not feature count.

## Development setup

```bash
uv venv && uv pip install -e '.[data,viz,dev]'   # lean stack (NO torch/sb3/gymnasium)
uv pip install -e '.[train]'                       # only if touching the offline path
```

## Quality gates (all must pass before a PR)

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest -q -m "not slow" --cov=rlallocator --cov-fail-under=85
```

- **Import purity** is non-negotiable: `import rlallocator` must load **no** torch /
  stable-baselines3 / gymnasium / onnxruntime / plotly. Heavy deps are imported lazily
  inside the functions that need them. The import-purity smoke test enforces this in a
  fresh interpreter.
- **Strict mypy** and **ruff** must be clean.
- Coverage gate: **≥ 85%**.
- The PURE honesty kernels (DSR / PSR, DM, PBO, seed-lottery, the verdict) must not
  regress — they carry parity tests against independent references.

## House conventions

- **No AI attribution anywhere** — commit messages, code comments, docs, and files must
  not contain `Co-Authored-By: Claude`, "Generated with Claude", or similar markers. A
  CI guard (`.github/workflows/no-ai-attribution.yml`) rejects them.
- Strictly causal data handling: the weights set at `t` earn `t → t+1` returns; the
  observation at `t` uses only data `≤ t`. Baselines estimate covariance on the train
  window only.
- The verdict is **derived from evidence, never narrated** — no profit claims.
- Commit clean, focused changes with conventional, descriptive messages.

## Commit hygiene

Keep commits atomic and messages imperative ("Add the CSCV PBO kernel"). Do **not**
push to `main` directly; open a PR. Do not add AI-attribution trailers or files.
