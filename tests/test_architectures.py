"""
Contract tests for the model registry.

The Review 3 comparison is only fair if the three models differ in architecture
and in nothing else. These tests pin that down: every registered builder must
accept the same input shape, emit one probability per sample, and compile with
the same weighted loss and the same metric set. A model that quietly diverged on
any of those would make its accuracy and confusion matrix incomparable with the
others while still training and reporting perfectly happily.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.models import architectures as A

INPUT_SHAPE = (config.SEQ_LEN, 55)
EXPECTED_METRICS = {"accuracy", "precision", "recall", "roc_auc", "pr_auc"}


@pytest.fixture(params=sorted(A.BUILDERS))
def model_name(request):
    return request.param


def _build(name):
    return A.build(name, INPUT_SHAPE, pos_weight=9.93)


def test_registry_is_not_empty():
    assert A.BUILDERS, "no architectures registered"


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        A.build("not_a_model", INPUT_SHAPE)


def test_accepts_the_shared_input_shape(model_name):
    model = _build(model_name)
    assert tuple(model.input_shape[1:]) == INPUT_SHAPE


def test_emits_one_probability_per_sample(model_name):
    """A single sigmoid output in [0, 1] -- the contract train.py assumes."""
    model = _build(model_name)
    assert tuple(model.output_shape[1:]) == (1,)

    x = np.random.randn(8, *INPUT_SHAPE).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (8, 1)
    assert np.all((y >= 0.0) & (y <= 1.0)), "output is not a probability"


def test_uses_the_shared_weighted_loss(model_name):
    model = _build(model_name)
    assert isinstance(model.loss, A.WeightedBCE)
    assert model.loss.pos_weight == pytest.approx(9.93)


def test_trains_for_one_step_and_reports_the_shared_metrics(model_name):
    """
    Catches shape faults that only surface once gradients flow, and checks the
    metric set at the same time.

    The metrics are read from the fit history rather than from ``model.metrics``
    because Keras only instantiates the compiled metrics once a train step has
    run -- before that the list is empty for every model.
    """
    model = _build(model_name)
    x = np.random.randn(32, *INPUT_SHAPE).astype(np.float32)
    y = np.random.randint(0, 2, size=32).astype(np.int8)

    history = model.fit(x, y, epochs=1, batch_size=16, verbose=0)

    assert np.isfinite(history.history["loss"][0]), "loss is not finite"
    missing = EXPECTED_METRICS - set(history.history)
    assert not missing, f"{model_name} is missing metrics: {sorted(missing)}"
