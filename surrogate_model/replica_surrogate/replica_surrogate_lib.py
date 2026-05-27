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
