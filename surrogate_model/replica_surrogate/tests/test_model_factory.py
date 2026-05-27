"""
test_model_factory.py
---------------------
Pytest tests for replica_surrogate_lib.build_replica_model.

Test 1 – Reproducibility:
    build_replica_model(seed=1) called twice → bit-identical initial weights
    in every layer.

Test 2 – Seed diversity:
    build_replica_model(seed=1) vs build_replica_model(seed=2) → at least one
    layer has measurably different initial weights.

Test 3 – Output shape:
    model.predict(np.zeros((5, 4))) → shape is (5, 2).
"""

import numpy as np
import pytest
import sys
import os

# Ensure the parent package is importable when running pytest from any cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from surrogate_model.replica_surrogate.replica_surrogate_lib import build_replica_model


# ---------------------------------------------------------------------------
# Test 1: Reproducibility — same seed → bit-identical initial weights
# ---------------------------------------------------------------------------

def test_reproducibility():
    """Two models built with the same seed must have identical initial weights."""
    model_a = build_replica_model(seed=1)
    model_b = build_replica_model(seed=1)

    for layer_a, layer_b in zip(model_a.layers, model_b.layers):
        weights_a = layer_a.get_weights()
        weights_b = layer_b.get_weights()
        assert len(weights_a) == len(weights_b), (
            f"Layer {layer_a.name} has different number of weight tensors "
            f"between the two models"
        )
        for w_a, w_b in zip(weights_a, weights_b):
            np.testing.assert_array_equal(
                w_a, w_b,
                err_msg=(
                    f"Layer {layer_a.name}: initial weights differ between "
                    f"two models built with seed=1"
                ),
            )


# ---------------------------------------------------------------------------
# Test 2: Seed diversity — different seeds → at least one layer differs
# ---------------------------------------------------------------------------

def test_seed_diversity():
    """Models built with different seeds must have at least one differing layer."""
    model_1 = build_replica_model(seed=1)
    model_2 = build_replica_model(seed=2)

    all_equal = True
    for layer_1, layer_2 in zip(model_1.layers, model_2.layers):
        weights_1 = layer_1.get_weights()
        weights_2 = layer_2.get_weights()
        for w1, w2 in zip(weights_1, weights_2):
            if not np.array_equal(w1, w2):
                all_equal = False
                break
        if not all_equal:
            break

    assert not all_equal, (
        "build_replica_model(seed=1) and build_replica_model(seed=2) produced "
        "identical weights in every layer — seed is not affecting initialization"
    )


# ---------------------------------------------------------------------------
# Test 3: Output shape — predict on 5 samples → shape (5, 2)
# ---------------------------------------------------------------------------

def test_output_shape():
    """Model output on 5 input samples must have shape (5, 2)."""
    model = build_replica_model(seed=42)
    x = np.zeros((5, 4))
    output = model.predict(x, verbose=0)
    assert output.shape == (5, 2), (
        f"Expected output shape (5, 2), got {output.shape}"
    )
