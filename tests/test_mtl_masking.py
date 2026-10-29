"""
Masking tests for Model 3.

A real race rarely has 20 cars running at a given lap, so most grids are padded.
If padding leaks, two things go wrong and neither raises: a driver attends to a
car that is not on track, and the model is trained against invented targets. The
symptom would be a model that trains cleanly and scores worse than it should,
which is close to undiagnosable after the fact.

The test strategy is to put *different garbage* in the padded slots of two
otherwise identical batches. Every masked quantity must come out bit-identical.

Run with:  python -m pytest tests/test_mtl_masking.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.models.mtl_architecture import build_mtl_attention

D, T, F = 8, config.SEQ_LEN, 55
N_REAL = 5          # first five slots hold real drivers, the rest are padding
B = 3


@pytest.fixture(scope="module")
def model():
    config.set_seed()
    return build_mtl_attention(seq_len=T, n_features=F, max_drivers=D, pos_weight=9.93)


def _mask():
    m = np.zeros((B, D), dtype="float32")
    m[:, :N_REAL] = 1.0
    return m


def _batch(seed, mask):
    """Identical real drivers; padded slots filled with seed-dependent noise."""
    rng = np.random.default_rng(0)
    real = rng.standard_normal((B, D, T, F)).astype("float32")
    noise = np.random.default_rng(seed).standard_normal((B, D, T, F)).astype("float32") * 50.0
    windows = np.where(mask[:, :, None, None] > 0, real, noise)

    rng_y = np.random.default_rng(1)
    y = {
        "pit": rng_y.integers(0, 2, (B, D)).astype("float32"),
        "compound": rng_y.integers(0, 5, (B, D)).astype("int32"),
        "compound_mask": np.ones((B, D), "float32"),
        "laptime": rng_y.standard_normal((B, D)).astype("float32"),
        "laptime_mask": np.ones((B, D), "float32"),
        "outcome": rng_y.integers(0, 3, (B, D)).astype("int32"),
        "outcome_mask": np.ones((B, D), "float32"),
        "driver_mask": mask,
    }
    # Padded slots also get garbage targets.
    noise_y = np.random.default_rng(seed + 99)
    pad = mask == 0
    y["pit"] = np.where(pad, noise_y.integers(0, 2, (B, D)), y["pit"]).astype("float32")
    y["laptime"] = np.where(pad, noise_y.standard_normal((B, D)) * 50.0, y["laptime"]).astype("float32")
    y["outcome"] = np.where(pad, noise_y.integers(0, 3, (B, D)), y["outcome"]).astype("int32")

    return {"windows": windows, "driver_mask": mask}, y


def test_padded_drivers_do_not_change_real_predictions(model):
    """Attention must not let a real driver see a padded slot."""
    mask = _mask()
    x_a, _ = _batch(seed=10, mask=mask)
    x_b, _ = _batch(seed=777, mask=mask)

    pit_a = model.predict(x_a, verbose=0)["pit"][:, :N_REAL]
    pit_b = model.predict(x_b, verbose=0)["pit"][:, :N_REAL]

    np.testing.assert_allclose(
        pit_a, pit_b, rtol=1e-5, atol=1e-6,
        err_msg="real drivers' predictions changed with the padding -- attention is leaking",
    )


def test_padded_targets_do_not_change_the_loss(model):
    """Every head must ignore targets sitting in padded slots."""
    mask = _mask()
    x_a, y_a = _batch(seed=11, mask=mask)
    x_b, y_b = _batch(seed=888, mask=mask)

    total_a, losses_a = model.compute_task_losses(y_a, model(x_a, training=False))
    total_b, losses_b = model.compute_task_losses(y_b, model(x_b, training=False))

    np.testing.assert_allclose(float(total_a), float(total_b), rtol=1e-5, atol=1e-6)
    for task in losses_a:
        np.testing.assert_allclose(
            float(losses_a[task]), float(losses_b[task]), rtol=1e-5, atol=1e-6,
            err_msg=f"the {task} head is being supervised on padded slots",
        )


def test_a_head_is_not_supervised_where_its_own_mask_is_zero(model):
    """compound_mask=0 rows must drop out even though the driver is real."""
    mask = _mask()
    x, y = _batch(seed=12, mask=mask)
    y_zero = dict(y)
    y_zero["compound_mask"] = np.zeros((B, D), "float32")

    y_pred = model(x, training=False)
    _, losses_on = model.compute_task_losses(y, y_pred)
    _, losses_off = model.compute_task_losses(y_zero, y_pred)

    assert float(losses_off["compound"]) == pytest.approx(0.0, abs=1e-6)
    assert float(losses_on["compound"]) > 0.0
    # Switching one head off must leave the others untouched.
    for task in ("pit", "laptime", "outcome"):
        np.testing.assert_allclose(
            float(losses_on[task]), float(losses_off[task]), rtol=1e-6, atol=1e-7
        )


def test_attention_actually_couples_drivers_and_the_ablation_removes_it():
    """
    The ablation's whole claim, tested directly.

    With attention on, changing one driver's window must move another driver's
    prediction -- otherwise the mechanism is inert and any score difference
    comes from somewhere else. With attention off, it must not move at all,
    otherwise the control is not a control.
    """
    mask = np.ones((1, D), dtype="float32")
    rng = np.random.default_rng(3)
    base = rng.standard_normal((1, D, T, F)).astype("float32")

    perturbed = base.copy()
    perturbed[0, 1] = rng.standard_normal((T, F)).astype("float32") * 3.0

    def driver0_pit(use_attention):
        config.set_seed()
        m = build_mtl_attention(seq_len=T, n_features=F, max_drivers=D,
                                pos_weight=9.93, use_attention=use_attention)
        a = m.predict({"windows": base, "driver_mask": mask}, verbose=0)["pit"][0, 0, 0]
        b = m.predict({"windows": perturbed, "driver_mask": mask}, verbose=0)["pit"][0, 0, 0]
        return float(a), float(b)

    on_a, on_b = driver0_pit(True)
    off_a, off_b = driver0_pit(False)

    assert abs(on_a - on_b) > 1e-6, (
        "attention is on but driver 0 ignored a change to driver 1 -- "
        "the cross-driver mechanism is inert"
    )
    assert off_a == pytest.approx(off_b, abs=1e-7), (
        "attention is off yet driver 0 still saw driver 1 -- "
        "the ablation control leaks"
    )


def test_an_all_padding_grid_yields_finite_loss(model):
    """A grid with no real drivers must not divide by zero."""
    mask = np.zeros((B, D), dtype="float32")
    x, y = _batch(seed=13, mask=mask)
    total, losses = model.compute_task_losses(y, model(x, training=False))
    assert np.isfinite(float(total))
    for task, value in losses.items():
        assert np.isfinite(float(value)), f"{task} loss is not finite on an empty grid"
