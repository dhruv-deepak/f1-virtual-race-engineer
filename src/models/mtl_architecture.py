"""
Model 3 -- multi-task network with cross-driver attention.

This is the project's contribution, and the one thing neither Model 1 nor Model
2 can express. Those two see a single driver's 10-lap window and must infer
competitive pressure from hand-built summary features (gap to leader, how much
of the field has stopped, tyre age relative to the field). Model 3 is handed
every driver on track at that lap and *learns* which rivals matter.

Shape of the network
--------------------
    windows      (B, D, T, F)   D drivers, T laps, F features
    driver_mask  (B, D)         1 where the slot holds a real driver

    1. A shared per-driver encoder runs the same Bi-LSTM over each driver's
       window independently -> (B, D, d). Sharing the weights is what makes it
       an encoder of "a driver's recent form" rather than of a grid position.
    2. Multi-head attention across the DRIVER axis, masked to real drivers, with
       a residual connection and layer norm. Each driver's representation is
       updated by the drivers it attends to.
    3. A position-wise feed-forward block, again residual -- the standard
       transformer block, applied over drivers instead of over tokens.
    4. Four task heads, each a dense layer applied per driver.

Why this is not in the single-output BUILDERS registry
------------------------------------------------------
Models 1 and 2 satisfy a contract the registry's tests enforce: one window in,
one probability out. Model 3 takes a grid plus a mask and emits four heads, so
it deliberately lives apart rather than weakening that contract. It is still
*evaluated* through the identical metrics on the identical test rows -- its pit
head is flattened back to per-driver-lap predictions for scoring.

Masking, which is the easy thing to get wrong
---------------------------------------------
A real race rarely has the full 20 cars running at a given lap: retirements,
and the 10-lap warm-up each driver needs, leave gaps. Padded slots must be
excluded from the attention (or a driver attends to a car that is not there)
*and* from every loss (or the model is trained on invented targets). Both are
done explicitly below.
"""

from __future__ import annotations

import keras
from keras import layers, ops

# Relative weight of each task in the combined loss. The pit head dominates
# because it is the task the Review 3 comparison scores; the auxiliary heads are
# there to shape the shared representation, not to be optimised for their own
# sake. The compound head is weighted lowest of the three because it is
# supervised on only ~8.6% of rows.
DEFAULT_TASK_WEIGHTS = {"pit": 1.0, "compound": 0.2, "laptime": 0.3, "outcome": 0.2}


@keras.saving.register_keras_serializable(package="f1vre")
class MultiTaskRaceEngineer(keras.Model):
    """Functional graph plus a custom masked multi-task training step."""

    def __init__(self, inputs=None, outputs=None, pos_weight: float = 1.0,
                 task_weights: dict | None = None, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.pos_weight = float(pos_weight)
        self.task_weights = dict(task_weights or DEFAULT_TASK_WEIGHTS)

        self._loss_trackers = {
            name: keras.metrics.Mean(name=f"{name}_loss")
            for name in ("total", "pit", "compound", "laptime", "outcome")
        }
        self._pit_pr_auc = keras.metrics.AUC(name="pr_auc", curve="PR")
        self._pit_roc_auc = keras.metrics.AUC(name="roc_auc")
        self._pit_acc = keras.metrics.BinaryAccuracy(name="accuracy")

    # -- metrics ------------------------------------------------------------
    @property
    def metrics(self):
        return [*self._loss_trackers.values(),
                self._pit_acc, self._pit_roc_auc, self._pit_pr_auc]

    # -- loss ---------------------------------------------------------------
    @staticmethod
    def _masked_mean(values, mask):
        """Mean over the entries the mask keeps, safe when nothing is kept."""
        total = ops.sum(values * mask)
        count = ops.maximum(ops.sum(mask), 1.0)
        return total / count

    def compute_task_losses(self, y, y_pred):
        """
        One masked loss per head.

        Every head multiplies its own availability mask by ``driver_mask``, so a
        padded slot can never contribute to any task, and a head is only ever
        supervised where its target is genuinely defined.
        """
        driver_mask = ops.cast(y["driver_mask"], "float32")

        # --- pit: weighted binary cross-entropy ----------------------------
        pit_true = ops.cast(y["pit"], "float32")
        pit_pred = ops.squeeze(y_pred["pit"], axis=-1)
        pit_bce = ops.binary_crossentropy(pit_true, pit_pred)
        pit_w = pit_true * self.pos_weight + (1.0 - pit_true)
        pit_loss = self._masked_mean(pit_bce * pit_w, driver_mask)

        # --- compound: 5-class, only where a stop is actually coming -------
        comp_mask = ops.cast(y["compound_mask"], "float32") * driver_mask
        comp_ce = ops.sparse_categorical_crossentropy(
            ops.cast(y["compound"], "int32"), y_pred["compound"]
        )
        compound_loss = self._masked_mean(comp_ce, comp_mask)

        # --- laptime: Huber, which the safety-car tail would wreck as MSE --
        lt_mask = ops.cast(y["laptime_mask"], "float32") * driver_mask
        lt_err = ops.squeeze(y_pred["laptime"], axis=-1) - ops.cast(y["laptime"], "float32")
        abs_err = ops.abs(lt_err)
        huber = ops.where(abs_err <= 1.0, 0.5 * ops.square(lt_err), abs_err - 0.5)
        laptime_loss = self._masked_mean(huber, lt_mask)

        # --- outcome: 3-class, constant per driver-race --------------------
        out_mask = ops.cast(y["outcome_mask"], "float32") * driver_mask
        out_ce = ops.sparse_categorical_crossentropy(
            ops.cast(y["outcome"], "int32"), y_pred["outcome"]
        )
        outcome_loss = self._masked_mean(out_ce, out_mask)

        losses = {"pit": pit_loss, "compound": compound_loss,
                  "laptime": laptime_loss, "outcome": outcome_loss}
        total = sum(self.task_weights[k] * v for k, v in losses.items())
        return total, losses

    def _update_metrics(self, y, y_pred, total, losses):
        self._loss_trackers["total"].update_state(total)
        for name, value in losses.items():
            self._loss_trackers[name].update_state(value)

        # Pit metrics are scored only over real drivers, via sample_weight.
        dm = ops.reshape(ops.cast(y["driver_mask"], "float32"), (-1,))
        yt = ops.reshape(ops.cast(y["pit"], "float32"), (-1,))
        yp = ops.reshape(ops.squeeze(y_pred["pit"], axis=-1), (-1,))
        for metric in (self._pit_acc, self._pit_roc_auc, self._pit_pr_auc):
            metric.update_state(yt, yp, sample_weight=dm)

    # -- train / eval steps -------------------------------------------------
    def train_step(self, data):
        x, y = data
        import tensorflow as tf

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            total, losses = self.compute_task_losses(y, y_pred)
            total = total + sum(self.losses)          # regularisation terms

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self._update_metrics(y, y_pred, total, losses)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        total, losses = self.compute_task_losses(y, y_pred)
        self._update_metrics(y, y_pred, total, losses)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        return {**super().get_config(),
                "pos_weight": self.pos_weight,
                "task_weights": self.task_weights}


def build_mtl_attention(seq_len: int, n_features: int, max_drivers: int,
                        n_compounds: int = 5, n_outcomes: int = 3,
                        pos_weight: float = 1.0, encoder_units: int = 32,
                        attn_heads: int = 4, attn_key_dim: int = 16,
                        ff_units: int = 64, dropout: float = 0.4,
                        learning_rate: float = 5e-4,
                        task_weights: dict | None = None,
                        use_attention: bool = True) -> MultiTaskRaceEngineer:
    """
    Assemble the encoder, the cross-driver attention block and the four heads.

    ``use_attention=False`` removes the attention sub-block and nothing else --
    the encoder, the feed-forward block, the normalisations and all four heads
    are untouched, so each driver is then encoded in complete isolation. That
    switch exists for the ablation: it is the only honest way to attribute a
    score difference to the cross-driver mechanism rather than to the extra
    depth and parameters that arrive with it.
    """
    windows = keras.Input(shape=(max_drivers, seq_len, n_features), name="windows")
    driver_mask = keras.Input(shape=(max_drivers,), name="driver_mask")

    # 1. Shared per-driver encoder. TimeDistributed applies one Bi-LSTM to each
    #    driver's window with the same weights, so the encoder learns "recent
    #    form" rather than anything tied to a slot in the grid.
    encoded = layers.TimeDistributed(
        layers.Bidirectional(layers.LSTM(encoder_units)), name="driver_encoder"
    )(windows)                                                   # (B, D, 2*units)
    encoded = layers.Dropout(dropout, name="encoder_dropout")(encoded)

    # 2. Cross-driver attention. The mask is (B, D, D): entry (i, j) says whether
    #    driver i may attend driver j. Both axes are masked, so neither a padded
    #    query nor a padded key takes part.
    if use_attention:
        mask_q = ops.cast(driver_mask, "bool")[:, :, None]
        mask_k = ops.cast(driver_mask, "bool")[:, None, :]
        attn_mask = ops.logical_and(mask_q, mask_k)

        attended = layers.MultiHeadAttention(
            num_heads=attn_heads, key_dim=attn_key_dim, dropout=dropout,
            name="cross_driver_attention",
        )(query=encoded, value=encoded, key=encoded, attention_mask=attn_mask)
        x = layers.Add(name="attention_residual")([encoded, attended])
        x = layers.LayerNormalization(name="attention_norm")(x)
    else:
        # Ablation control: every driver stays isolated from every other.
        # The mask is still applied -- zeroing padded slots, which leaves real
        # drivers untouched -- both because it is correct and because the
        # functional graph requires the driver_mask input to be consumed.
        x = layers.Multiply(name="mask_padding")(
            [encoded, ops.expand_dims(driver_mask, axis=-1)]
        )
        x = layers.LayerNormalization(name="attention_norm")(x)

    # 3. Position-wise feed-forward block -- the standard transformer second
    #    half, applied across drivers rather than across tokens.
    ff = layers.Dense(ff_units, activation="relu", name="ff_1")(x)
    ff = layers.Dropout(dropout, name="ff_dropout")(ff)
    ff = layers.Dense(x.shape[-1], name="ff_2")(ff)
    x = layers.Add(name="ff_residual")([x, ff])
    x = layers.LayerNormalization(name="ff_norm")(x)

    # 4. Four heads sharing that one representation.
    outputs = {
        "pit": layers.Dense(1, activation="sigmoid", name="pit")(x),
        "compound": layers.Dense(n_compounds, activation="softmax", name="compound")(x),
        "laptime": layers.Dense(1, name="laptime")(x),
        "outcome": layers.Dense(n_outcomes, activation="softmax", name="outcome")(x),
    }

    model = MultiTaskRaceEngineer(
        inputs={"windows": windows, "driver_mask": driver_mask},
        outputs=outputs,
        pos_weight=pos_weight,
        task_weights=task_weights,
        name="mtl_attn",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model
