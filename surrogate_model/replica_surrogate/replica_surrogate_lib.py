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
