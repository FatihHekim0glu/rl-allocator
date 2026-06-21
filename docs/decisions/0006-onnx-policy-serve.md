# ADR-0006 — ONNX-policy serve (no torch at request time)

- Status: Accepted
- Context: the serve path must be lean and import-pure; training is heavy and offline.

## Context

PPO needs torch, stable-baselines3 and gymnasium to train. Shipping those into the serve
container is wasteful (hundreds of MB) and breaks the import-purity rule — `import
rlallocator` must have zero heavy import-time side effects, and the deployed `[serve]`
extra must not pull torch. But the served metrics still need the trained policy's
decisions, evaluated over the purged walk-forward.

## Decision

Train offline, serve via ONNX.

- `agents/ppo.py` (the `[train]` extra) trains SB3 PPO, extracts **only** the policy MLP
  (`obs → per-asset scores`), and exports that network to ONNX with
  `torch.onnx.export(..., dynamo=False)`, validated **1e-4** against the torch forward
  pass at export time.
- `agents/onnx_policy.py` loads the committed `artifacts/policy.onnx` **lazily** via
  `onnxruntime` and projects the scores onto the simplex — it never imports torch / sb3 /
  gymnasium.
- The container `[serve]` extra is `onnxruntime` only; `[train]` (torch + sb3 + gymnasium)
  is offline. The request path reads precomputed OOS metrics + the committed ONNX policy;
  it **never trains per request**.
- A fallback is documented: if SB3 → ONNX export is troublesome, a hand-rolled plain-torch
  policy MLP (ONNX-clean) ships instead; the honest-NULL deliverable holds either way and
  the README/CHANGELOG record which shipped.

## Consequences

- The lean serve install imports onnxruntime only; an import-purity smoke test
  (`tests/unit/test_import_purity.py`) and a torch-free end-to-end test
  (`tests/integration/test_serve_end_to_end.py`) pin it.
- The 1e-4 export-time parity gate guarantees the served policy is the trained policy.
- The committed `policy.onnx` is `< 10 MB`, tracked in git, and ships in the wheel so the
  deployed serve path is self-contained.
