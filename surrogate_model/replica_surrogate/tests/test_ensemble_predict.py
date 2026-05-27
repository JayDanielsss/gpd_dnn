"""
test_ensemble_predict.py
------------------------
Pytest tests for replica_surrogate_lib.ensemble_predict.

Test 1 – Constant output (single model):
    A single mock model returning a fixed standardized output → the returned
    mean is that output correctly inverse-transformed, and the std is zero.

Test 2 – Non-zero std (two models with different outputs):
    Two mock models returning different standardized outputs → the returned
    std is non-zero.  The expected std equals scale_ * std of the
    standardized predictions — this test fails if double-scaling is
    introduced or the scale_ multiplication is forgotten.
"""

import numpy as np
import pytest
import sys
import os

# Ensure the parent package is importable when running pytest from any cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from surrogate_model.replica_surrogate.replica_surrogate_lib import (
    fit_scalers,
    ensemble_predict,
)


# ---------------------------------------------------------------------------
# Helper: simple mock model
# ---------------------------------------------------------------------------

class _ConstantModel:
    """Mock model whose .predict() always returns the same 2-D array."""

    def __init__(self, output):
        """
        Parameters
        ----------
        output : array-like of shape (n_outputs,)
            The constant row that will be broadcast to every sample.
        """
        self._output = np.asarray(output, dtype=float)

    def predict(self, x):
        n_samples = x.shape[0]
        return np.tile(self._output, (n_samples, 1))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scalers():
    """Return (x_scaler, y_scaler) fitted on small deterministic data."""
    rng = np.random.default_rng(seed=0)

    x_train = rng.uniform(
        low=[-0.5, 0.1, 1.0, -3.14],
        high=[0.0,  0.4, 4.0,  3.14],
        size=(20, 4),
    )
    y_train = np.column_stack([
        rng.uniform(0.5, 10.0, size=20),   # xsec  (O(5))
        rng.uniform(-0.5, 0.5, size=20),   # BSA   (O(0.2))
    ])

    return fit_scalers(x_train, y_train)


@pytest.fixture
def x_raw():
    """Five raw input rows to use as query points."""
    rng = np.random.default_rng(seed=1)
    return rng.uniform(
        low=[-0.5, 0.1, 1.0, -3.14],
        high=[0.0,  0.4, 4.0,  3.14],
        size=(5, 4),
    )


# ---------------------------------------------------------------------------
# Test 1: constant output — mean is correctly inverse-transformed, std == 0
# ---------------------------------------------------------------------------

def test_constant_output(scalers, x_raw):
    """Single model always returning the same standardized output.

    The ensemble mean must equal y_scaler.inverse_transform of that constant,
    and the ensemble std must be exactly zero (only one replica → no spread).
    """
    x_scaler, y_scaler = scalers

    # Arbitrary standardized output that is NOT [0, 0]
    standardized_output = np.array([1.5, -0.8])

    model = _ConstantModel(standardized_output)
    mean_phys, std_phys = ensemble_predict([model], x_raw, x_scaler, y_scaler)

    # --- mean check ---
    expected_mean = y_scaler.inverse_transform(standardized_output.reshape(1, -1))
    # expected_mean shape: (1, 2) → broadcast against (5, 2)
    np.testing.assert_allclose(
        mean_phys,
        np.tile(expected_mean, (x_raw.shape[0], 1)),
        rtol=1e-6, atol=1e-10,
        err_msg="Mean in physical units does not match inverse_transform of standardized output",
    )

    # --- std check: one model → std should be zero ---
    np.testing.assert_allclose(
        std_phys,
        np.zeros_like(std_phys),
        atol=1e-10,
        err_msg="Std should be zero for a single-model ensemble",
    )


# ---------------------------------------------------------------------------
# Test 2: two models, different outputs → non-zero std; correct formula
# ---------------------------------------------------------------------------

def test_nonzero_std(scalers, x_raw):
    """Two models returning different standardized outputs.

    The physical std must be non-zero and must equal
        std_standardized * y_scaler.scale_
    This test fails if:
      - scale_ multiplication is omitted (std would be in standardized units)
      - double-scaling is applied (e.g. inverse_transform used on std, adding mean)
    """
    x_scaler, y_scaler = scalers

    # Two clearly different standardized outputs
    out_a = np.array([1.0,  0.5])
    out_b = np.array([3.0, -1.5])

    models = [_ConstantModel(out_a), _ConstantModel(out_b)]
    mean_phys, std_phys = ensemble_predict(models, x_raw, x_scaler, y_scaler)

    # --- std must be non-zero ---
    assert np.all(std_phys > 0), "Std should be non-zero when two models disagree"

    # --- std must equal scale_ * std_standardized ---
    # std of [out_a, out_b] for each element:
    std_scaled_expected = np.std(np.array([out_a, out_b]), axis=0)  # shape (2,)
    std_phys_expected   = std_scaled_expected * y_scaler.scale_     # element-wise

    # Every row of std_phys should equal std_phys_expected (models are constant)
    np.testing.assert_allclose(
        std_phys,
        np.tile(std_phys_expected, (x_raw.shape[0], 1)),
        rtol=1e-6, atol=1e-10,
        err_msg=(
            "Physical std does not match std_standardized * y_scaler.scale_. "
            "Check for double-scaling or missing scale_ multiplication."
        ),
    )

    # --- mean check: mean of out_a and out_b inverse-transformed ---
    mean_scaled_expected = (out_a + out_b) / 2.0
    mean_phys_expected   = y_scaler.inverse_transform(mean_scaled_expected.reshape(1, -1))

    np.testing.assert_allclose(
        mean_phys,
        np.tile(mean_phys_expected, (x_raw.shape[0], 1)),
        rtol=1e-6, atol=1e-10,
        err_msg="Mean in physical units does not match expected value",
    )
