"""
replica_surrogate_lib.py
------------------------
Utility functions for the replica surrogate model.

The dominant fix for the flat BSA error band is z-score standardization of
both output observables so that under unweighted MSE they contribute equally
to the gradient.  Without standardization, ``unp_beam_unp_target_xsec``
(O(5)) contributes ~600× more loss signal than ``unp_target_bsa`` (O(0.2)),
causing the BSA head to collapse to a near-trivial constant fit.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def fit_scalers(x_train, y_train):
    """Fit StandardScalers on training data only and return them.

    Parameters
    ----------
    x_train : array-like of shape (n_samples, n_features)
        Training inputs (kinematics: t, x_b, q_squared, phi).
    y_train : array-like of shape (n_samples, n_outputs)
        Training targets (unp_beam_unp_target_xsec, unp_target_bsa).

    Returns
    -------
    x_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for input features.
    y_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for output targets.

    Notes
    -----
    Both scalers are fitted **only** on ``x_train`` / ``y_train``.  The
    caller is responsible for applying ``transform`` to validation and test
    sets using the scalers returned here, which prevents data leakage.
    """
    x_scaler = StandardScaler()
    x_scaler.fit(x_train)

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    return x_scaler, y_scaler


def ensemble_predict(models, x_raw, x_scaler, y_scaler):
    """Run an ensemble of models on raw (un-standardized) inputs and return
    the mean and standard-deviation of predictions in physical units.

    Parameters
    ----------
    models : list of objects with a ``.predict(x)`` method
        Trained replica models.  Each must accept a 2-D array of
        *standardized* inputs and return a 2-D array of *standardized*
        outputs of shape ``(n_samples, n_outputs)``.
    x_raw : array-like of shape (n_samples, n_features)
        Raw (un-standardized) input features.
    x_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for input features (returned by :func:`fit_scalers`).
    y_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for output targets (returned by :func:`fit_scalers`).

    Returns
    -------
    mean_physical : numpy.ndarray of shape (n_samples, n_outputs)
        Ensemble mean prediction in physical units
        (``y_scaler.inverse_transform`` applied to the mean of the
        standardized predictions).
    std_physical : numpy.ndarray of shape (n_samples, n_outputs)
        Ensemble standard deviation in physical units.  Computed as
        ``std_standardized * y_scaler.scale_`` (element-wise), which is
        the correct formula for propagating a std through a linear
        rescaling — **not** ``inverse_transform(std_standardized)``, which
        would incorrectly add the mean offset.

    Notes
    -----
    The ``std_physical`` formula follows from the fact that
    ``inverse_transform`` is the affine map ``z → z * scale_ + mean_``.
    For a collection of standardized replica outputs ``{z_r}``:

        std(z_r * scale_ + mean_) = scale_ * std(z_r)

    so multiplying the standardized std by ``scale_`` gives the correct
    physical std without any double-counting of the mean offset.
    """
    x_scaled = x_scaler.transform(x_raw)

    # Stack per-replica standardized predictions: shape (n_replicas, n_samples, n_outputs)
    preds_scaled = np.array([model.predict(x_scaled) for model in models])

    mean_scaled = np.mean(preds_scaled, axis=0)   # (n_samples, n_outputs)
    std_scaled  = np.std(preds_scaled, axis=0)    # (n_samples, n_outputs)

    mean_physical = y_scaler.inverse_transform(mean_scaled)
    std_physical  = std_scaled * y_scaler.scale_  # element-wise; scale_ shape: (n_outputs,)

    return mean_physical, std_physical


def build_replica_model(seed: int) -> tf.keras.Model:
    """Build and compile a seeded replica model with deterministic initialization.

    Each Dense layer receives its own ``GlorotNormal`` initializer instance
    with a unique seed derived from ``seed``.  Using one shared initializer
    instance across layers triggers a Keras warning and causes correlated
    weight draws; separate instances with offset seeds avoid both issues.

    Architecture: Input(4) → Dense(48, tanh) × 3 → Dense(2, linear)

    Parameters
    ----------
    seed : int
        Base seed for weight initialization.  Layers use seeds
        ``seed``, ``seed+1000``, ``seed+2000``, and ``seed+3000``
        so that within one model the hidden layers have meaningfully
        different weight draws despite sharing the same shape.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model with Adam(lr=3e-4) and MSE loss.
    """
    # One separate GlorotNormal instance per layer — avoids the
    # "initializer instance reused" Keras warning.
    init_1 = tf.keras.initializers.GlorotNormal(seed=seed)
    init_2 = tf.keras.initializers.GlorotNormal(seed=seed + 1000)
    init_3 = tf.keras.initializers.GlorotNormal(seed=seed + 2000)
    init_out = tf.keras.initializers.GlorotNormal(seed=seed + 3000)

    model_inputs = tf.keras.Input(shape=(4,), name="input_values")

    hidden = tf.keras.layers.Dense(
        48, kernel_initializer=init_1, activation="tanh")(model_inputs)
    hidden = tf.keras.layers.Dense(
        48, kernel_initializer=init_2, activation="tanh")(hidden)
    hidden = tf.keras.layers.Dense(
        48, kernel_initializer=init_3, activation="tanh")(hidden)

    model_output = tf.keras.layers.Dense(
        2, kernel_initializer=init_out, activation="linear")(hidden)

    model = tf.keras.Model(inputs=model_inputs, outputs=model_output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="mse",
    )

    return model
