"""
verify_phase1.py
----------------
Phase-1 structural verification of replica-ensemble fix on the 18-row burner dataset.

Runs four structural checks and prints clear PASS/FAIL for each.
"""

import os
import sys
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

# Add repo root to path so imports resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import train_test_split
from surrogate_model.replica_surrogate.replica_surrogate_lib import (
    fit_scalers,
    ensemble_predict,
    build_replica_model,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BURNER_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'burner_data.csv')
X_COLS = ['t', 'x_b', 'q_squared', 'phi']
Y_COLS = ['unp_beam_unp_target_xsec', 'unp_target_bsa']
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(BURNER_CSV)
    x_data = df[X_COLS].values
    y_data = df[Y_COLS].values

    x_rem, x_test, y_rem, y_test = train_test_split(
        x_data, y_data, test_size=0.1, shuffle=True, random_state=RANDOM_STATE
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_rem, y_rem, test_size=0.1, shuffle=True, random_state=RANDOM_STATE
    )

    x_scaler, y_scaler = fit_scalers(x_train, y_train)
    return df, x_data, y_data, x_train, x_val, y_train, y_val, x_scaler, y_scaler


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
def train_ensemble(n_replicas, x_train, y_train, x_val, y_val, x_scaler, y_scaler):
    """Train an ensemble of n_replicas models, each seeded by its replica number."""
    models = []

    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_val_scaled = x_scaler.transform(x_val)
    y_val_scaled = y_scaler.transform(y_val)

    for replica_number in range(1, n_replicas + 1):
        tf.random.set_seed(replica_number)
        np.random.seed(replica_number)
        tf.keras.backend.clear_session()
        gc.collect()

        model = build_replica_model(seed=replica_number)

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=60, restore_best_weights=True
            ),
        ]

        model.fit(
            x_train_scaled, y_train_scaled,
            validation_data=(x_val_scaled, y_val_scaled),
            epochs=500,
            batch_size=8,
            callbacks=callbacks,
            verbose=0,
        )

        models.append(model)

    return models


# ---------------------------------------------------------------------------
# Smooth phi curve helper
# ---------------------------------------------------------------------------
def make_smooth_phi_curve(df):
    """Return x_smooth at the first unique kinematic group over 361 phi points."""
    first_group = df.groupby(['t', 'x_b', 'q_squared']).first().reset_index().iloc[0]
    t_val, xb_val, q2_val = first_group['t'], first_group['x_b'], first_group['q_squared']
    phi_smooth = np.linspace(-np.pi, np.pi, 361)
    x_smooth = np.column_stack([
        np.full_like(phi_smooth, t_val),
        np.full_like(phi_smooth, xb_val),
        np.full_like(phi_smooth, q2_val),
        phi_smooth,
    ])
    return x_smooth


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase-1 Structural Verification")
    print("Dataset: 18-row burner dataset")
    print("=" * 60)

    df, x_data, y_data, x_train, x_val, y_train, y_val, x_scaler, y_scaler = load_data()

    results = {}

    # ------------------------------------------------------------------
    # Check 1 — BSA phi-dependent structure (N=5 and N=50)
    # ------------------------------------------------------------------
    print("\n--- Check 1: BSA phi-dependent structure (N=5 and N=50) ---")

    print("  Training N=5 ensemble...")
    models_5 = train_ensemble(5, x_train, y_train, x_val, y_val, x_scaler, y_scaler)

    print("  Training N=50 ensemble...")
    models_50 = train_ensemble(50, x_train, y_train, x_val, y_val, x_scaler, y_scaler)

    x_smooth = make_smooth_phi_curve(df)

    bsa_mean_5, bsa_std_5 = ensemble_predict(models_5, x_smooth, x_scaler, y_scaler)
    bsa_mean_50, bsa_std_50 = ensemble_predict(models_50, x_smooth, x_scaler, y_scaler)

    # BSA is the second output column (index 1)
    std_N5 = np.std(bsa_mean_5[:, 1])
    std_N50 = np.std(bsa_mean_50[:, 1])

    print(f"  std(BSA_mean) — N=5:  {std_N5:.6f}")
    print(f"  std(BSA_mean) — N=50: {std_N50:.6f}")

    check1_pass = (std_N5 > 0.001) and (std_N50 > 0.001)
    results['check1'] = check1_pass
    print(f"  Check 1: {'PASS' if check1_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 — N=5 vs N=50 band shapes differ
    # ------------------------------------------------------------------
    print("\n--- Check 2: N=5 vs N=50 BSA band shapes differ ---")

    # std bands for BSA (index 1)
    std_band_5  = bsa_std_5[:, 1]
    std_band_50 = bsa_std_50[:, 1]

    mean_abs_diff = np.mean(np.abs(std_band_5 - std_band_50))
    print(f"  Mean abs diff of std bands: {mean_abs_diff:.8f}")

    check2_pass = mean_abs_diff > 1e-6
    results['check2'] = check2_pass
    print(f"  Check 2: {'PASS' if check2_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 3 — Reproducibility (same seeds → bit-identical predictions)
    # ------------------------------------------------------------------
    print("\n--- Check 3: Reproducibility (same seeds → bit-identical predictions) ---")

    print("  Training N=3 ensemble (run 1)...")
    models_run1 = train_ensemble(3, x_train, y_train, x_val, y_val, x_scaler, y_scaler)
    pred_run1, _ = ensemble_predict(models_run1, x_data, x_scaler, y_scaler)

    print("  Training N=3 ensemble (run 2, same seeds)...")
    models_run2 = train_ensemble(3, x_train, y_train, x_val, y_val, x_scaler, y_scaler)
    pred_run2, _ = ensemble_predict(models_run2, x_data, x_scaler, y_scaler)

    bit_identical = np.array_equal(pred_run1, pred_run2)
    print(f"  bit-identical: {bit_identical}")

    check3_pass = bit_identical
    results['check3'] = check3_pass
    print(f"  Check 3: {'PASS' if check3_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 4 — Spot-check de-standardization round-trip
    # ------------------------------------------------------------------
    print("\n--- Check 4: Spot-check de-standardization round-trip ---")

    # Train a fresh single replica (seed=99)
    tf.random.set_seed(99)
    np.random.seed(99)
    tf.keras.backend.clear_session()
    gc.collect()

    single_model = build_replica_model(seed=99)
    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_val_scaled = x_scaler.transform(x_val)
    y_val_scaled = y_scaler.transform(y_val)

    single_model.fit(
        x_train_scaled, y_train_scaled,
        validation_data=(x_val_scaled, y_val_scaled),
        epochs=500,
        batch_size=8,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=30, min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=60, restore_best_weights=True
            ),
        ],
        verbose=0,
    )

    # Use the first row of training data
    x_row = x_train[[0], :]
    y_train_row = y_train[0]

    pred_mean, _ = ensemble_predict([single_model], x_row, x_scaler, y_scaler)
    pred_vals = pred_mean[0]

    print(f"  Predicted: {pred_vals}")
    print(f"  Actual:    {y_train_row}")

    ratio = np.abs(pred_vals / y_train_row)
    print(f"  |pred/actual|: {ratio}")
    check4_pass = bool(np.all(ratio < 10))
    results['check4'] = check4_pass
    print(f"  Check 4: {'PASS' if check4_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = all(results.values())
    for i, (key, val) in enumerate(results.items(), 1):
        print(f"  Check {i}: {'PASS' if val else 'FAIL'}")
    print()
    if all_pass:
        print("All 4 checks PASS.")
    else:
        failing = [str(i) for i, (k, v) in enumerate(results.items(), 1) if not v]
        print(f"FAILING checks: {', '.join(failing)}")

    # Print values for PHASE1_VERIFICATION.md
    print("\n--- Values for PHASE1_VERIFICATION.md ---")
    print(f"std(BSA_mean) N=5:  {std_N5:.6f}")
    print(f"std(BSA_mean) N=50: {std_N50:.6f}")
    print(f"mean abs diff std bands: {mean_abs_diff:.8f}")
    print(f"bit-identical: {bit_identical}")
    print(f"pred={list(pred_vals)}, actual={list(y_train_row)}")

    return results


if __name__ == '__main__':
    main()
