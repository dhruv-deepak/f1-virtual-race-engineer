"""
Model architectures for the pit-stop prediction task.

Every architecture here takes the same input shape ``(SEQ_LEN, n_features)`` and
returns a single sigmoid probability, so the three models compared in Review 3
can be swapped in and out without touching the data pipeline, the training loop
or the evaluation code. That interchangeability is what makes their accuracy
scores and confusion matrices genuinely comparable.

Registry
--------
    bilstm      Model 1 -- Bi-LSTM sequential baseline          (Review 2)
    cnn_bilstm  Model 2 -- reserved                             (Phase 4)
    mtl_attn    Model 3 -- reserved                             (Phase 5)
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from src import config


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="f1vre")
class WeightedBCE(keras.losses.Loss):
    """
    Binary cross-entropy with the positive class up-weighted.

    Only ~13% of laps are followed by a pit stop. Plain cross-entropy lets the
    model collapse onto "never pit" and still score ~87% accuracy, so the
    positive class is scaled by ``pos_weight``.

    This is applied as a *loss function* rather than via Keras' ``class_weight``
    argument on purpose: ``class_weight`` is not applied to the validation set,
    which would leave the training and validation loss curves on different
    scales and make the loss graph impossible to read. Baking the weight into
    the loss keeps both curves directly comparable.
    """

    def __init__(self, pos_weight: float = 1.0, name: str = "weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = float(pos_weight)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        weights = y_true * self.pos_weight + (1.0 - y_true)
        return tf.reduce_mean(weights * bce)

    def get_config(self):
        return {**super().get_config(), "pos_weight": self.pos_weight}


def standard_metrics() -> list:
    """
    The metric set every model reports.

    Accuracy alone is misleading under 13% positives, so precision, recall and
    both AUCs are tracked from the first epoch. PR-AUC is the headline number:
    it is the one that actually degrades when a model stops predicting pit stops.
    """
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="roc_auc"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]


# ---------------------------------------------------------------------------
# Model 1 -- Bi-LSTM baseline
# ---------------------------------------------------------------------------
def build_bilstm(input_shape: tuple[int, int], pos_weight: float = 1.0,
                 learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """
    Stacked bidirectional LSTM.

    Why bidirectional: the input is a closed 10-lap window that is already in
    the past at prediction time, not a live stream. The model is therefore free
    to read it in both directions, which lets the later laps of the window
    inform how the earlier ones are interpreted -- a stint's degradation trend
    reads more clearly backwards than forwards.

    The stack narrows 64 -> 32 -> 32: the first layer returns the full sequence
    so the second can compress it into one summary vector per window.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape, name="lap_window"),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_1"),
            layers.Dropout(0.3, name="dropout_1"),
            layers.Bidirectional(layers.LSTM(32), name="bilstm_2"),
            layers.Dropout(0.3, name="dropout_2"),
            layers.Dense(32, activation="relu", name="dense"),
            layers.Dense(1, activation="sigmoid", name="pit_probability"),
        ],
        name="bilstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=WeightedBCE(pos_weight=pos_weight),
        metrics=standard_metrics(),
    )
    return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BUILDERS = {
    "bilstm": build_bilstm,
}


def build(name: str, input_shape: tuple[int, int], **kwargs) -> keras.Model:
    if name not in BUILDERS:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(BUILDERS)}")
    return BUILDERS[name](input_shape, **kwargs)
