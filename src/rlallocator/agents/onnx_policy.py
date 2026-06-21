"""ONNX policy inference — the SERVE path (onnxruntime, NEVER torch).

TYPED STUB (SCAFFOLD). The container and the FastAPI router run the trained PPO
policy through this module ONLY. It loads the committed ``artifacts/policy.onnx`` MLP
with onnxruntime (the ``[serve]`` extra = numpy + onnxruntime) and runs a forward
pass mapping an observation (look-back window + current weights) to per-asset SCORES,
which are projected onto the long-only weight SIMPLEX via
:func:`rlallocator._validation.project_to_simplex`; torch / stable-baselines3 /
gymnasium are NEVER imported here. onnxruntime is imported LAZILY inside the methods
so that ``import rlallocator`` stays free of any inference engine.

:func:`default_artifact_path` (pure path arithmetic, no I/O) IS implemented so the
serve / CLI path can probe for the committed artifact and gracefully degrade to a
baselines-only comparison while the offline-trained policy is a separate deliverable.
The inference methods raise ``NotImplementedError`` until the artifact + onnxruntime
wiring is committed.

Importing this module has no side effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rlallocator._exceptions import ArtifactError
from rlallocator._typing import FloatArray, ObservationMatrix
from rlallocator._validation import project_to_simplex

#: Directory holding the committed, shipped ONNX artifact(s).
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent.parent / "artifacts"

#: The committed policy artifact filename (the exported PPO policy MLP).
POLICY_ARTIFACT_FILENAME: str = "policy.onnx"


def default_artifact_path() -> Path:
    """Return the filesystem path of the shipped policy ONNX artifact.

    Pure path arithmetic — does NOT check existence and imports nothing heavy, so it
    is safe to call at import-time of a caller. The serve / CLI path uses
    ``default_artifact_path().is_file()`` to decide whether the committed policy is
    available (and otherwise degrades to a baselines-only comparison honestly).

    Returns
    -------
    pathlib.Path
        ``<package>/artifacts/policy.onnx``.
    """
    return ARTIFACTS_DIR / POLICY_ARTIFACT_FILENAME


class OnnxPolicy:
    """A thin onnxruntime wrapper that serves the committed PPO policy artifact.

    TYPED STUB (SCAFFOLD). The onnxruntime session is created LAZILY on first use,
    so constructing this object is cheap and import-pure; torch / stable-baselines3 /
    gymnasium are never imported on this path. The forward pass binds the per-bar
    OBSERVATION (look-back window + current weights) to the exported policy graph and
    projects the per-asset score output onto the long-only weight simplex. The
    inference methods raise ``NotImplementedError`` until the artifact + onnxruntime
    wiring is committed.
    """

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        """Record the artifact path; defer session creation to :meth:`load`.

        Parameters
        ----------
        artifact_path:
            Explicit path to the ``.onnx`` file. Defaults to the shipped
            ``artifacts/policy.onnx`` (:func:`default_artifact_path`).
        """
        self._artifact_path = Path(artifact_path) if artifact_path else default_artifact_path()
        self._session: object | None = None

    @property
    def artifact_path(self) -> Path:
        """Return the resolved artifact path this wrapper will load."""
        return self._artifact_path

    def load(self) -> OnnxPolicy:
        """Create the onnxruntime inference session (lazy, idempotent).

        LAZY IMPORT: ``onnxruntime`` will be imported inside this method. NO torch /
        sb3 / gymnasium import occurs anywhere on this path. The artifact file is
        checked BEFORE onnxruntime is imported, so a missing artifact raises
        :class:`ArtifactError` without loading any inference engine.

        Returns
        -------
        OnnxPolicy
            ``self``, with an initialized session.

        Raises
        ------
        ArtifactError
            If the artifact file is missing.
        NotImplementedError
            Until the onnxruntime session wiring is committed.
        """
        if not self._artifact_path.is_file():
            raise ArtifactError(
                f"OnnxPolicy.load: policy ONNX artifact not found at {self._artifact_path}."
            )
        raise NotImplementedError(
            "OnnxPolicy.load is a scaffold stub; the onnxruntime session wiring is "
            "committed alongside the offline-trained policy.onnx artifact."
        )

    def predict_scores(self, observations: ObservationMatrix) -> FloatArray:
        """Return the raw per-asset score logits for a batch of observations (ONNX, no torch).

        The lower-level forward pass behind :meth:`predict_weights`: binds
        ``observations`` to the exported graph and returns the raw
        ``(batch, n_assets)`` per-asset scores. Used by the ONNX-vs-torch parity test
        (validated to 1e-4 against the SB3 policy). NO torch on this path.

        Parameters
        ----------
        observations:
            A ``(batch, obs_dim)`` observation matrix.

        Returns
        -------
        FloatArray
            The ``(batch, n_assets)`` per-asset score logits.

        Raises
        ------
        NotImplementedError
            Until the onnxruntime forward pass is committed.
        """
        raise NotImplementedError(
            "OnnxPolicy.predict_scores is a scaffold stub; implemented with the "
            "committed policy.onnx + onnxruntime session."
        )

    def predict_weights(self, observations: ObservationMatrix) -> FloatArray:
        """Map a batch of observations to per-bar weight SIMPLICES via the committed policy.

        Runs :meth:`predict_scores` and projects each row of per-asset scores onto the
        long-only weight simplex via
        :func:`rlallocator._validation.project_to_simplex`, returning a
        ``(batch, n_assets)`` weight path that is a valid simplex EVERY bar. The
        backtester then scores this weight path. NO torch on this path.

        Parameters
        ----------
        observations:
            A ``(batch, obs_dim)`` observation matrix (look-back window + weights).

        Returns
        -------
        FloatArray
            A ``(batch, n_assets)`` per-bar weight path (each row a valid simplex).

        Raises
        ------
        NotImplementedError
            Until the onnxruntime forward pass is committed.
        """
        scores = self.predict_scores(observations)
        rows = [
            project_to_simplex(np.asarray(row, dtype="float64")) for row in np.atleast_2d(scores)
        ]
        return np.asarray(rows, dtype="float64")


def score_weights_from_onnx(
    observations: ObservationMatrix,
    artifact_path: str | Path | None = None,
) -> FloatArray:
    """Serve the committed policy's per-bar weight simplices from its ONNX artifact (no torch).

    A thin convenience wrapper over :class:`OnnxPolicy` for the backend: load the
    committed ``policy.onnx`` lazily (onnxruntime, NO torch) and map a batch of
    observations to a per-bar weight path (each row a valid simplex).

    Parameters
    ----------
    observations:
        A ``(batch, obs_dim)`` observation matrix.
    artifact_path:
        Explicit artifact path; ``None`` => the shipped ``artifacts/policy.onnx``.

    Returns
    -------
    FloatArray
        A ``(batch, n_assets)`` per-bar weight path.

    Raises
    ------
    ArtifactError
        If the artifact is missing/corrupt or its signature mismatches the inputs.
    NotImplementedError
        Until the onnxruntime forward pass is committed.
    """
    return OnnxPolicy(artifact_path).predict_weights(observations)
