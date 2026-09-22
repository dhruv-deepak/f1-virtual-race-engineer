"""
Regression tests for the weighted loss.

These exist because of a bug that cost a full training run. Keras passes
``y_true`` into ``Loss.call`` with shape ``(batch,)`` but leaves ``y_pred`` at
``(batch, 1)``. An arithmetic expression combining the two broadcasts to a
``(batch, batch)`` outer product rather than a ``(batch,)`` vector. Nothing
raises: training runs to completion, the loss even decreases slightly, and the
model collapses to a constant 0.5 prediction with ROC-AUC 0.50. The only symptom
is a model that silently refuses to learn.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.architectures import WeightedBCE


def test_call_returns_one_loss_per_sample():
    """The shape contract the original bug violated."""
    loss = WeightedBCE(pos_weight=5.0)
    y_true = np.array([1.0, 0.0, 1.0, 0.0])          # (4,)   as Keras passes it
    y_pred = np.array([[0.9], [0.1], [0.8], [0.2]])  # (4, 1) as the model emits it

    per_sample = np.asarray(loss.call(y_true, y_pred))

    assert per_sample.shape == (4,), (
        f"expected one loss per sample, got {per_sample.shape} -- "
        "a (4, 4) result means y_true and y_pred broadcast against each other"
    )


def test_matches_hand_computed_weighted_bce():
    """Value check against the definition, independent of any Keras internals."""
    pos_weight = 5.0
    loss = WeightedBCE(pos_weight=pos_weight)
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([[0.8], [0.3]])

    expected = np.array([
        pos_weight * -np.log(0.8),   # positive sample, up-weighted
        1.0 * -np.log(1.0 - 0.3),    # negative sample, unweighted
    ])
    actual = np.asarray(loss.call(y_true, y_pred))

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_positive_class_is_weighted_more_than_negative():
    """An equally-wrong positive must cost pos_weight times an equally-wrong negative."""
    loss = WeightedBCE(pos_weight=10.0)
    per_sample = np.asarray(
        loss.call(np.array([1.0, 0.0]), np.array([[0.2], [0.8]]))
    )
    np.testing.assert_allclose(per_sample[0] / per_sample[1], 10.0, rtol=1e-5)


def test_pos_weight_of_one_equals_plain_bce():
    loss = WeightedBCE(pos_weight=1.0)
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([[0.7], [0.4], [0.55]])

    expected = -np.log([0.7, 1 - 0.4, 0.55])
    np.testing.assert_allclose(
        np.asarray(loss.call(y_true, y_pred)), expected, rtol=1e-5
    )


def test_survives_a_serialisation_round_trip():
    """The loss is saved inside the .keras checkpoint, so it must rebuild."""
    original = WeightedBCE(pos_weight=7.5)
    restored = WeightedBCE.from_config(original.get_config())
    assert restored.pos_weight == pytest.approx(7.5)
