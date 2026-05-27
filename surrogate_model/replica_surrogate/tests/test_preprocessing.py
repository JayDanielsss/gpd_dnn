"""
test_preprocessing.py
---------------------
Pytest tests for replica_surrogate_lib.fit_scalers.

Test 1 – Round-trip:
    inverse_transform(transform(x)) ≈ x  (both for x and y scalers)

Test 2 – Fit-only-on-training:
    The scalers' mean_ and scale_ match the statistics of the training
    subset and are NOT influenced by the non-training (held-out) rows.
"""

import numpy as np
import pytest
import sys
import os

# Ensure the parent package is importable when running pytest from any cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from surrogate_model.replica_surrogate.replica_surrogate_lib import fit_scalers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Return deterministic (x_train, y_train, x_held_out, y_held_out)."""
    rng = np.random.default_rng(seed=42)

    # Simulate kinematics: t, x_b, q_squared, phi
    # Use very different scales to stress-test the scaler.
    x_train = rng.uniform(
        low=[-0.5, 0.1, 1.0, -3.14],
        high=[0.0, 0.4, 4.0,  3.14],
        size=(20, 4)
    )

    # Simulate outputs: xsec (O(5)) and BSA (O(0.2))
    y_train = np.column_stack([
        rng.uniform(0.5, 10.0, size=20),   # xsec
        rng.uniform(-0.5, 0.5, size=20),   # BSA
    ])

    # Held-out data with deliberately different statistics
    x_held_out = rng.uniform(
        low=[0.1, 0.5, 5.0, -3.14],
        high=[0.5, 0.9, 8.0,  3.14],
        size=(10, 4)
    )
    y_held_out = np.column_stack([
        rng.uniform(50.0, 100.0, size=10),  # very different scale from train
        rng.uniform(-5.0, -4.0, size=10),   # very different mean from train
    ])

    return x_train, y_train, x_held_out, y_held_out


# ---------------------------------------------------------------------------
# Test 1: Round-trip inverse_transform(transform(x)) ≈ x
# ---------------------------------------------------------------------------

def test_roundtrip(sample_data):
    """Applying transform then inverse_transform recovers the original data."""
    x_train, y_train, _, _ = sample_data

    x_scaler, y_scaler = fit_scalers(x_train, y_train)

    x_transformed = x_scaler.transform(x_train)
    x_recovered   = x_scaler.inverse_transform(x_transformed)
    np.testing.assert_allclose(
        x_recovered, x_train, rtol=1e-5, atol=1e-8,
        err_msg="x_scaler round-trip failed"
    )

    y_transformed = y_scaler.transform(y_train)
    y_recovered   = y_scaler.inverse_transform(y_transformed)
    np.testing.assert_allclose(
        y_recovered, y_train, rtol=1e-5, atol=1e-8,
        err_msg="y_scaler round-trip failed"
    )


# ---------------------------------------------------------------------------
# Test 2: Scaler statistics match training data; ignore held-out data
# ---------------------------------------------------------------------------

def test_scaler_fit_only_on_training(sample_data):
    """
    The scaler mean_ and scale_ must match the training-data statistics
    and must NOT be influenced by held-out rows.
    """
    x_train, y_train, x_held_out, y_held_out = sample_data

    x_scaler, y_scaler = fit_scalers(x_train, y_train)

    # ---- Check that mean_ and scale_ match training statistics ----
    np.testing.assert_allclose(
        x_scaler.mean_, np.mean(x_train, axis=0), rtol=1e-5,
        err_msg="x_scaler.mean_ does not match training mean"
    )
    np.testing.assert_allclose(
        x_scaler.scale_, np.std(x_train, axis=0, ddof=0), rtol=1e-5,
        err_msg="x_scaler.scale_ does not match training std"
    )
    np.testing.assert_allclose(
        y_scaler.mean_, np.mean(y_train, axis=0), rtol=1e-5,
        err_msg="y_scaler.mean_ does not match training mean"
    )
    np.testing.assert_allclose(
        y_scaler.scale_, np.std(y_train, axis=0, ddof=0), rtol=1e-5,
        err_msg="y_scaler.scale_ does not match training std"
    )

    # ---- Check that held-out data has visibly different statistics ----
    # This confirms the held-out fixture is meaningfully different, so the
    # test above is a genuine check against data leakage.
    assert not np.allclose(
        np.mean(y_held_out, axis=0), y_scaler.mean_, atol=1.0
    ), (
        "Held-out y mean is unexpectedly close to training mean — "
        "the fixture may not be testing data-leakage resistance"
    )
