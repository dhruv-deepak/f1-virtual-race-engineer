"""
Phase 6 -- recover each trained model's predictions.

The per-model metrics JSON records confusion matrices and curves' summary
statistics but not the probabilities themselves, so the combined ROC and
precision-recall figures cannot be drawn from it. This module reloads each
trained model and writes its validation and test probabilities to
``reports/metrics/<model>_probs.npz``.

Predictions are recovered rather than regenerated
-------------------------------------------------
The models are *reloaded from the checkpoints the training runs saved*, not
retrained. Retraining would risk small numerical drift and leave the figures
disagreeing with the numbers already reported. To prove the reload is faithful,
each model's metrics are recomputed from the recovered probabilities and
checked against the stored JSON; a mismatch beyond floating-point tolerance
aborts rather than writing anything.

Usage
-----
    python -m src.evaluation.predict
"""

from __future__ import annotations

import json
import sys
import warnings

import keras
import numpy as np

from src import config
from src.evaluation import metrics as M
from src.models.architectures import WeightedBCE  # noqa: F401  (registers the loss)

warnings.filterwarnings("ignore")

SINGLE_INPUT_MODELS = ["bilstm", "cnn_bilstm"]
MTL_MODEL = "mtl_attn"
TOLERANCE = 1e-6


def _load_metrics(name: str) -> dict:
    path = config.METRICS_DIR / f"{name}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- train {name} first")
    with open(path) as fh:
        return json.load(fh)


def predict_single(name: str, seq) -> tuple[np.ndarray, np.ndarray]:
    """Reload a single-input model (Models 1 and 2) and score val + test."""
    ckpt = config.MODELS_DIR / f"{name}.keras"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found -- train {name} first")
    model = keras.models.load_model(ckpt)
    val = model.predict(seq["X_val"], batch_size=512, verbose=0).ravel()
    test = model.predict(seq["X_test"], batch_size=512, verbose=0).ravel()
    return val, test


def predict_mtl(seq) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild Model 3, load its weights, and scatter grids back to rows."""
    from src.features.build_mtl_dataset import MAX_DRIVERS  # noqa: F401
    from src.models.mtl_architecture import build_mtl_attention
    from src.models.train_mtl import grid_arrays, load_grids, scatter_to_rows

    stored = _load_metrics(MTL_MODEL)
    _, mtl, spec, X, n_train, n_val = load_grids()
    n_rows = len(X)

    model = build_mtl_attention(
        seq_len=spec["seq_len"], n_features=spec["n_features"],
        max_drivers=spec["max_drivers"], n_compounds=spec["n_compound_classes"],
        n_outcomes=spec["n_outcome_classes"], pos_weight=stored["pos_weight"],
    )
    weights = config.MODELS_DIR / f"{MTL_MODEL}.weights.h5"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found -- train {MTL_MODEL} first")
    model.load_weights(weights)

    out = {}
    for split, slc in (("val", slice(n_train, n_train + n_val)),
                       ("test", slice(n_train + n_val, n_rows))):
        x, _, grid, mask = grid_arrays(mtl, X, split, spec)
        pred = model.predict(x, batch_size=64, verbose=0)["pit"][..., 0]
        rows = scatter_to_rows(pred, grid, mask, n_rows)[slc]
        if np.isnan(rows).any():
            raise ValueError(f"{split}: some rows received no prediction")
        out[split] = rows
    return out["val"], out["test"]


def verify(name: str, y_test: np.ndarray, test_prob: np.ndarray, stored: dict) -> None:
    """Recomputed metrics must reproduce the stored ones, or we abort."""
    recomputed = M.compute(y_test, test_prob, stored["tuned_threshold"])
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        delta = abs(recomputed[key] - stored["test"][key])
        if delta > TOLERANCE:
            raise ValueError(
                f"{name}: recovered {key} = {recomputed[key]:.6f} but the stored run "
                f"reported {stored['test'][key]:.6f} (delta {delta:.2e}). The reloaded "
                "checkpoint is not the model that produced the published numbers."
            )
    if recomputed["confusion_matrix"] != stored["test"]["confusion_matrix"]:
        raise ValueError(f"{name}: recovered confusion matrix differs from the stored one")


def main() -> int:
    config.ensure_dirs()
    config.set_seed()

    seq = np.load(config.DATA_PROCESSED / "sequences.npz")
    y_val, y_test = seq["y_val"], seq["y_test"]

    for name in SINGLE_INPUT_MODELS + [MTL_MODEL]:
        stored = _load_metrics(name)
        val_prob, test_prob = (predict_mtl(seq) if name == MTL_MODEL
                               else predict_single(name, seq))
        verify(name, y_test, test_prob, stored)

        out = config.METRICS_DIR / f"{name}_probs.npz"
        np.savez_compressed(out, val=val_prob.astype(np.float32),
                            test=test_prob.astype(np.float32))
        print(f"  {name:<12} verified against the stored run  ->  {out.name}  "
              f"(ROC-AUC {stored['test']['roc_auc']:.4f}, PR-AUC {stored['test']['pr_auc']:.4f})")

    print(f"\nall {len(SINGLE_INPUT_MODELS) + 1} models reproduce their published metrics exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
