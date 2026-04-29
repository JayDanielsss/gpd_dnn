##########################################
# FILE INFORMATION:
# Purpose: runs *a* replica based on a single
# DNN architecture.
# Created: 20260107
# Last changed: 20260429
##########################################

print("[INFO]: Script began running!")

##########################################
# Importing Python Libraries
##########################################

import sys
import gc
import json

import pandas as pd
import numpy as np
import tensorflow as tf

from simultaneous_fit_dnn_config import _INITIALIZER_MINIMUM_VALUE
from simultaneous_fit_dnn_config import _INITIALIZER_MAXMIMUM_VALUE

from simultaneous_fit_dnn_config import _NUMBER_NODES_HIDDEN_1
from simultaneous_fit_dnn_config import _NUMBER_NODES_HIDDEN_2
from simultaneous_fit_dnn_config import _NUMBER_NODES_HIDDEN_3
from simultaneous_fit_dnn_config import _NUMBER_NODES_HIDDEN_4

from simultaneous_fit_dnn_config import compute_fe
from simultaneous_fit_dnn_config import compute_fg
from simultaneous_fit_dnn_config import compute_f2
from simultaneous_fit_dnn_config import compute_f1
from simultaneous_fit_dnn_config import compute_epsilon
from simultaneous_fit_dnn_config import compute_y
from simultaneous_fit_dnn_config import compute_skewness
from simultaneous_fit_dnn_config import compute_t_min
from simultaneous_fit_dnn_config import compute_t_prime
from simultaneous_fit_dnn_config import compute_k_tilde
from simultaneous_fit_dnn_config import compute_k
from simultaneous_fit_dnn_config import compute_k_dot_delta
from simultaneous_fit_dnn_config import prop_1
from simultaneous_fit_dnn_config import prop_2

from simultaneous_fit_dnn_config import bkm10_cross_section
from simultaneous_fit_dnn_config import bkm10_bsa

print("[INFO]: Libraries imported!")

##########################################
# [IMPORTANT]: Static quantities parametrizing
# the program. Change these if you need!
##########################################

# verify this is what you want
SCRATCH_PATH = 'placeholder!'

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

TEST_LEPTON_HELICITY = 0.0
TEST_TARGET_POLARIZATION = 0.0

print(f"[INFO]: Detected lepton helicity to be: {'unpolarized' if TEST_LEPTON_HELICITY == 0.0 else 'polarized'}")

NUMBER_OF_REPLICAS = 100
_NUMBER_OF_EPOCHS = 750
_BATCH_SIZE = 1

print(f"[INFO]: Each replica has a batch size of {_BATCH_SIZE}")
print(f"[INFO]: Each replica will run for {_NUMBER_OF_EPOCHS} if assuming no EarlyStopping.")

##########################################
# Reading the Datafile:
##########################################

try:
    with open(
        f"./pseudodata_slurm_logs/version_{MAJOR_MINOR_NUMBER}/valid_kinematic_sets_v{MAJOR_MINOR_NUMBER}.txt", 
        "r",
        encoding = "utf-8") as good_kinematic_sets_file:
        # This creates a list like [2, 9, 15, ...]
        valid_kinematic_sets = [ int(line.strip()) for line in good_kinematic_sets_file if line.strip() ]
except FileNotFoundError:
    print(f"[ERROR]: Could not find pseudodata_slurm_logs/version_{MAJOR_MINOR_NUMBER}!")
    sys.exit(1)

print(f"[INFO]: Good kinematic sets were: {valid_kinematic_sets}")

# extract kinematic set number
task_id = int(sys.argv[1]) - 1
set_index = task_id // NUMBER_OF_REPLICAS
replica_number = (task_id % NUMBER_OF_REPLICAS) + 1
kinematic_set_number = valid_kinematic_sets[set_index]

print(f"[INFO]: Now running Kinematic Set #{kinematic_set_number}")

# this reads the pseudodata!
dnn_replica_data = pd.read_csv(
    filepath_or_buffer = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/dnn_data_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.csv"
)

TOTAL_DATA_SIZE = len(dnn_replica_data)

FIXED_BEAM_ENERGY = dnn_replica_data["k"].iloc[0]
FIXED_Q_SQUARED = dnn_replica_data["q_squared"].iloc[0]
FIXED_X_BJORKEN = dnn_replica_data["x_b"].iloc[0]
FIXED_T_VALUE = dnn_replica_data["t"].iloc[0]

print(f"[INFO]: Selected beam energy k = {FIXED_BEAM_ENERGY}")
print(f"[INFO]: Selected Q^2 = {FIXED_Q_SQUARED}")
print(f"[INFO]: Selected xB = {FIXED_X_BJORKEN}")
print(f"[INFO]: Selected t = {FIXED_T_VALUE}")

CFF_REAL_H_KM15 = dnn_replica_data["Re[H]"].iloc[0]
CFF_IMAG_H_KM15 = dnn_replica_data["Im[H]"].iloc[0]
CFF_REAL_E_KM15 = dnn_replica_data["Re[E]"].iloc[0]
CFF_IMAG_E_KM15 = dnn_replica_data["Im[E]"].iloc[0]
CFF_REAL_HT_KM15 = dnn_replica_data["Re[Ht]"].iloc[0]
CFF_IMAG_HT_KM15 = dnn_replica_data["Im[Ht]"].iloc[0]
CFF_REAL_ET_KM15 = dnn_replica_data["Re[Et]"].iloc[0]
CFF_IMAG_ET_KM15 = dnn_replica_data["Im[Et]"].iloc[0]

CFF_H_KM15 = complex(CFF_REAL_H_KM15, CFF_IMAG_H_KM15)
CFF_H_TILDE_KM15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
CFF_E_KM15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
CFF_E_TILDE_KM15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

print(f"[INFO]: Selected CFF H = {CFF_H_KM15}")
print(f"[INFO]: Selected CFF E = {CFF_E_KM15}")
print(f"[INFO]: Selected CFF Ht = {CFF_H_TILDE_KM15}")
print(f"[INFO]: Selected CFF Et = {CFF_E_TILDE_KM15}")

training_df = dnn_replica_data[dnn_replica_data["split"] == "train"]
validation_df = dnn_replica_data[dnn_replica_data["split"] == "validation"]
testing_df = dnn_replica_data[dnn_replica_data["split"] == "test"]

number_of_dnn_training_points = len(training_df)
number_of_dnn_validation_points = len(validation_df)
number_of_dnn_testing_points = len(testing_df)

x_training = training_df[["t", "x_b", "q_squared", "phi"]]
y_training = training_df[["unp_beam_unp_target_xsec", "unp_target_bsa"]]

x_validation = validation_df[["t", "x_b", "q_squared", "phi"]]
y_validation = validation_df[["unp_beam_unp_target_xsec", "unp_target_bsa"]]

x_testing = testing_df[["t", "x_b", "q_squared", "phi"]]
y_testing = testing_df[["unp_beam_unp_target_xsec", "unp_target_bsa"]]

if number_of_dnn_training_points <= _BATCH_SIZE:
    print(f"[WARN]: Number of training points is less than or equal to the batch size. Setting batch size equal to {number_of_dnn_training_points}.")
    _BATCH_SIZE = number_of_dnn_training_points

##########################################
# Tensorflow Configuration
##########################################

print(f"[INFO]: Physical devices available to TF are: {tf.config.list_physical_devices()}")
print(f"[INFO]: Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"[INFO]: Number of CPUs Available: {len(tf.config.list_physical_devices('CPU'))}")
print(f"[INFO]: Checking the GPU name: {tf.test.gpu_device_name()}")

class SimultaneousObservablesLoss(tf.keras.losses.Loss):
    def __init__(self, name = "simultaneous_observables_loss"):
        super().__init__(name = name)

        self._OBSERVABLE_WEIGHT_1 = 0.5 * 1.0
        self._OBSERVABLE_WEIGHT_2 = 0.5 * 1.0
        self._OBSERVABLE_WEIGHT_3 = 0.0 * 0.5
        self._OBSERVABLE_WEIGHT_4 = 0.0 * 0.5
        
    def call(self, true_values, predicted_values):
        
        cff_h_real_tf = predicted_values[:, 0]
        cff_h_imag_tf = predicted_values[:, 1]
        t_tf = predicted_values[:, 2]
        xb_tf = predicted_values[:, 3]
        q_squared_tf = predicted_values[:, 4]
        phi_values = predicted_values[:, 5]

        fe_tf = compute_fe(t_tf)
        fg_tf = compute_fg(fe_tf) 
        f2_tf = compute_f2(t_tf, fe_tf, fg_tf)
        f1_tf = compute_f1(fg_tf, f2_tf)
        
        epsilon_tf = compute_epsilon(xb_tf, q_squared_tf)
        y_lep_tf = compute_y(FIXED_BEAM_ENERGY, q_squared_tf, epsilon_tf)
        xi_tf = compute_skewness(xb_tf, t_tf, q_squared_tf)
        tmin_tf = compute_t_min(xb_tf, q_squared_tf, epsilon_tf)
        tprime_tf = compute_t_prime(t_tf, tmin_tf) # used in interference only
        ktilde_tf = compute_k_tilde(xb_tf, q_squared_tf, t_tf, tmin_tf, epsilon_tf)
        k_tf = compute_k(q_squared_tf, y_lep_tf, epsilon_tf, ktilde_tf)
        kdd_tf = compute_k_dot_delta(q_squared_tf, xb_tf, t_tf, phi_values, epsilon_tf, y_lep_tf, k_tf)
        p1_tf = prop_1(q_squared_tf, kdd_tf)
        p2_tf = prop_2(q_squared_tf, t_tf, kdd_tf)
            
        true_cross_section = true_values[:, 0]
        true_bsa = true_values[:, 1]

        cross_section = bkm10_cross_section(
            TEST_LEPTON_HELICITY, TEST_TARGET_POLARIZATION,
            q_squared_tf, xb_tf, t_tf, epsilon_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_values, p1_tf, p2_tf,
            cff_h_real_tf, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag_tf, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)
        
        # compute cross-section residuals:
        residuals_cross_section = true_cross_section - cross_section

        predicted_bsa = bkm10_bsa(
            TEST_TARGET_POLARIZATION,
            q_squared_tf, xb_tf, t_tf, epsilon_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_values, p1_tf, p2_tf,
            cff_h_real_tf, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag_tf, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)
        
        # compute BSA residuals:
        residuals_bsa = true_bsa - predicted_bsa

        # compute the MSE:
        mean_squared_error = (
            self._OBSERVABLE_WEIGHT_1 * tf.reduce_mean(tf.square(residuals_cross_section)) +
            self._OBSERVABLE_WEIGHT_2 * tf.reduce_mean(tf.square(residuals_bsa))
            )

        return mean_squared_error
    
def cff_h_model():
    initializer = tf.keras.initializers.RandomUniform(
        minval = _INITIALIZER_MINIMUM_VALUE,
        maxval = _INITIALIZER_MAXMIMUM_VALUE,
        seed = None)
    
    all_model_inputs = tf.keras.Input(shape = (4,), name = "input_values") 
    kinematic_inputs = tf.keras.layers.Lambda(lambda x: x[:, :3], name = 'input_kinematics')(all_model_inputs) # takes t, xb, qsquared

    # [NOTE]: ONLY kinematic inputs goes into the DNN!
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_1, kernel_initializer = initializer, activation = "relu")(kinematic_inputs)
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_2, kernel_initializer = initializer, activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_3, kernel_initializer = initializer, activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_4, kernel_initializer = initializer, activation = "relu")(hidden)

    # linear activation is default activation if `activation` key is not specified: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    cff_outputs = tf.keras.layers.Dense(2, activation = "linear", name = "cff_h")(hidden) # Re[H] and Im[H]
    
    full_model_outputs = tf.keras.layers.Concatenate(name = "kinematics_and_cffs")(
        [cff_outputs, all_model_inputs])
    
    model = tf.keras.Model(
        inputs = all_model_inputs,
        outputs = full_model_outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = SimultaneousObservablesLoss())
    
    return model

##########################################
# DNN Model Setup
##########################################

dnn_model = cff_h_model()

dnn_model_history = dnn_model.fit(
    x_training,
    y_training,
    validation_data = (x_validation, y_validation),
    epochs = _NUMBER_OF_EPOCHS,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 25,
            restore_best_weights = True
        )
    ],
    batch_size = _BATCH_SIZE,
    verbose = 0
    )

number_of_epochs_run = len(dnn_model_history.epoch)
print(f"The model ran for {number_of_epochs_run} epochs before early stopping.")

dnn_model.save(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas/replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras")

metadata = {
    "replica_id": replica_number,
    "version": MAJOR_MINOR_NUMBER,
    "batch_size": _BATCH_SIZE,
    "max_epochs": _NUMBER_OF_EPOCHS,
    "actual_epochs": len(dnn_model_history.epoch),
    "training_points": len(x_training),
    "features": list(x_training.columns)
}
with open(
    file = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/replica_{replica_number}_metadata.json",
    mode = "w",
    encoding = "utf-8") as f:
    json.dump(metadata, f, indent = 4)

history_df = pd.DataFrame(dnn_model_history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/replica_{replica_number}_history.csv", index = False)

dnn_evaluation_statistics = dnn_model.evaluate(x_testing, y_testing, verbose = 0, return_dict = True)
print(f"[INFO]: Test Loss for Replica {replica_number}: {dnn_evaluation_statistics}")
pd.DataFrame([dnn_evaluation_statistics]).to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/replica_{replica_number}_test_metrics.csv", 
    index = False)
# https://stackoverflow.com/a/17840195 -> for why we need to cast it into a list!

y_predictions = dnn_model.predict(x_testing)
prediction_results = x_testing.copy()
prediction_results['true_y'] = y_testing
prediction_results['predicted_y'] = y_predictions.flatten()
prediction_results.to_csv(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/replica_{replica_number}_test_predictions.csv", index = False)

# cleanup
del dnn_model
del training_df
del validation_df
del testing_df

tf.keras.backend.clear_session()
gc.collect()

with open(
    file = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/learning_curves/log_v{MAJOR_MINOR_NUMBER}.txt",
    mode = "w",
    buffering = 1,
    encoding = 'utf-8') as logfile:
    logfile.write(f"[INFO]: #{kinematic_set_number}: Bin k = {FIXED_BEAM_ENERGY}, Q^2 = {FIXED_Q_SQUARED}, xb = {FIXED_X_BJORKEN}, t = {FIXED_T_VALUE}\n")
    logfile.write(f"[INFO]: Re[H] = {CFF_REAL_H_KM15}, Im[H] = {CFF_IMAG_H_KM15}, Re[E] = {CFF_REAL_E_KM15}, Im[H] = {CFF_IMAG_E_KM15}\n")
    logfile.write(f"[INFO]: Re[Ht] = {CFF_REAL_HT_KM15}, Im[Ht] = {CFF_IMAG_HT_KM15}, Re[Et] = {CFF_REAL_ET_KM15}, Im[Ht] = {CFF_IMAG_ET_KM15}\n")
    logfile.write(f"[INFO]: Total replicas: {NUMBER_OF_REPLICAS}\n")
    logfile.write(f"[INFO]: Batch size: {_BATCH_SIZE}\n")
    logfile.write(f"[INFO]: Maximum number of epochs: {_NUMBER_OF_EPOCHS}\n")
    logfile.write(f"[INFO]: Out of {TOTAL_DATA_SIZE}, we picked {number_of_dnn_training_points} training, {number_of_dnn_validation_points} validation, and {number_of_dnn_testing_points} testing.\n")

print("[INFO]: End of script reached!")
