##########################################
# FILE INFORMATION:
# Purpose: generate replica pseudodata
# Created: 20260325
# Last changed: 20260427
##########################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import gc
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import corner
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split

#################################################################################
# Version numbers!
#################################################################################

print(f"[INFO]: numpy version: {np.__version__}")
print(f"[INFO]: pandas version: {pd.__version__}")
print(f"[INFO]: tensorflow version: {tf.__version__}")
print(f"[INFO]: corner version: {corner.__version__}")
print(f"[INFO]: seaborn version: {sns.__version__}")

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Begin main program flow!
#################################################################################

test_dataframe = pd.read_csv(
    f'./local/version_{MAJOR_MINOR_NUMBER}/data/combined_xsec_bsa_experimental_data_v{MAJOR_MINOR_NUMBER}.csv')

x_data = test_dataframe[["k", "t", "x_b", "q_squared", "phi"]]
y_data = test_dataframe[["unp_beam_unp_target_xsec", "unp_target_bsa"]]

TOTAL_DATA_SIZE = len(x_data)
print(f"[NOTE]: Total data size is: {TOTAL_DATA_SIZE}")

_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE = 0.1 # 90% temporary, 10% testing
_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE = 0.1 # of the above 90% temporary, 90% training, 10% validation

number_of_dnn_temporary_points = int(np.ceil(TOTAL_DATA_SIZE * _DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE))
number_of_dnn_testing_points = TOTAL_DATA_SIZE - number_of_dnn_temporary_points
number_of_dnn_training_points = int(np.ceil(number_of_dnn_temporary_points * _DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE))
number_of_dnn_validation_points = TOTAL_DATA_SIZE - number_of_dnn_training_points - number_of_dnn_testing_points

print(f"[NOTE]: Testing/Temporary Split is {_DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_testing_points} testing points (with ceiling).")
print(f"[NOTE]: Training/Validation Split is {_DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_validation_points} validation points (with ceiling).")
print(f"[NOTE]: Remaining training data points are: {number_of_dnn_training_points}")

x_remaining, x_testing, y_remaining, y_testing = train_test_split(
    x_data, y_data,
    test_size = _DNN_TESTING_TEMPORARY_SPLIT_PERCENTAGE, shuffle = True)

x_training, x_validation, y_training, y_validation = train_test_split(
    x_remaining, y_remaining,
    test_size = _DNN_TRAINING_VALIDATION_SPLIT_PERCENTAGE, shuffle = True)

print(f"[NOTE]: x-training size is: {len(x_training)}")
print(f"[NOTE]: x-validation size is: {len(x_validation)}")
print(f"[NOTE]: x-testing size is: {len(x_testing)}")

#################################################################################
# Plotting stuff!
#################################################################################

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['savefig.dpi'] = 300

all_uuxsec_experiments_figure = plt.figure(figsize = (9, 7))
all_uuxsec_experiments_axis = all_uuxsec_experiments_figure.add_subplot(1, 1, 1, projection = "3d")

axis_elevation = all_uuxsec_experiments_axis.elev # extract eleveation param
axis_azimuthal = all_uuxsec_experiments_axis.azim # extract azimuth parm

all_uuxsec_experiments_axis.text2D(
    0.01, 0.03,
    fr"elevation = {axis_elevation}, $\phi = {axis_azimuthal}^{{\circ}}$", 
    transform = all_uuxsec_experiments_axis.transAxes)
all_uuxsec_experiments_axis.text2D(
    0.01, 0.00,
    fr"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
    transform = all_uuxsec_experiments_axis.transAxes)

x_train_vals = x_training["x_b"]
y_train_vals = x_training["q_squared"]
z_train_vals = x_training["t"]

x_val_vals = x_validation["x_b"]
y_val_vals = x_validation["q_squared"]
z_val_vals = x_validation["t"]

x_test_vals = x_testing["x_b"]
y_test_vals = x_testing["q_squared"]
z_test_vals = x_testing["t"]

all_uuxsec_experiments_axis.scatter(
    x_train_vals, y_train_vals, z_train_vals,
    c = 'green', label = 'Training',
    s = 20, alpha = 0.3, edgecolors = 'none')

all_uuxsec_experiments_axis.scatter(
    x_val_vals, y_val_vals, z_val_vals,
    c = 'orange', label = 'Validation',
    s = 20, alpha = 0.3, edgecolors = 'none')

all_uuxsec_experiments_axis.scatter(
    x_test_vals, y_test_vals, z_test_vals,
    c = 'red', label = 'Testing',
    s = 25, alpha = 0.3, edgecolors = 'none')

all_uuxsec_experiments_axis.grid(True)
all_uuxsec_experiments_axis.set_title(
    r"Experimental Kinematic Settings Space for $d^{{4}}\sigma^{{UU}}$", fontsize = 18)
all_uuxsec_experiments_axis.set_xlabel(r"$x_{\textrm{B}}$", fontsize = 16.)
all_uuxsec_experiments_axis.set_ylabel(r"$Q^{2}$", fontsize = 16.)
all_uuxsec_experiments_axis.set_zlabel(r"$t$", fontsize = 16.)
all_uuxsec_experiments_axis.legend(fontsize = 16.)

for extension in ['png', 'eps']:
    all_uuxsec_experiments_figure.savefig(
        f"./local/version_{MAJOR_MINOR_NUMBER}/plots/surrogate_dnn_data_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = 'white',
            transparent = False)

plt.close(all_uuxsec_experiments_figure)

#################################################################################
# Training!
#################################################################################

def cff_h_model():
    initializer = tf.keras.initializers.RandomUniform(
        minval = -0.1, maxval = 0.1, seed = None)
    
    model_inputs = tf.keras.Input(shape = (5,), name = "input_values")
    hidden = tf.keras.layers.Dense(10, kernel_initializer = initializer, activation = "relu")(model_inputs)
    hidden = tf.keras.layers.Dense(10, kernel_initializer = initializer, activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(10, kernel_initializer = initializer, activation = "relu")(hidden)
    model_output = tf.keras.layers.Dense(2, kernel_initializer = initializer, activation = "relu")(hidden)

    model = tf.keras.Model(
        inputs = model_inputs,
        outputs = model_output)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.MeanSquaredError(),
        )
    return model

NUMBER_OF_EPOCHS = 750
number_of_replicas = 1

for replica in range(number_of_replicas):

    replica_number = replica + 1

    tf.keras.backend.clear_session()
    dnn_model = cff_h_model()

    dnn_model_history = dnn_model.fit(
        x_training, y_training,
        validation_data = (x_validation, y_validation),
        epochs = NUMBER_OF_EPOCHS, batch_size = 1,
        verbose = 1)

    number_of_epochs_run = len(dnn_model_history.epoch)
    print(f"[INFO]: The model ran for {number_of_epochs_run} epochs before early stopping.")

    dnn_model.save(f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras")

    training_loss_data = dnn_model_history.history["loss"]
    validation_loss_data = dnn_model_history.history["val_loss"]

    dnn_evaluation_statistics = dnn_model.evaluate(x_testing, y_testing, verbose = 1)
    print(f"[INFO]: Test Loss for Replica {replica_number}: {dnn_evaluation_statistics}")

    pd.DataFrame({
        'testing_loss': [dnn_evaluation_statistics], # https://stackoverflow.com/a/17840195 -> for why we need to cast it into a list!
    }).to_csv(
        f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_loss_data.csv", 
        index = False)

    # make DF with DNN training information
    pd.DataFrame(dnn_model_history.history).to_csv(
        f"./local/version_{MAJOR_MINOR_NUMBER}/replicas/replica_{replica_number}_losses_vs_epochs.csv", 
        index = False)

    gc.collect()

    y_predictions = dnn_model.predict(x_training)

    true_xsec = y_training.iloc[:, 0].values
    pred_xsec = y_predictions[:, 0]
    true_bsa = y_training.iloc[:, 1].values
    pred_bsa = y_predictions[:, 1]

    residuals = np.abs(y_training - y_predictions)
    _EPSILON = 1e-8
    residuals_normalized = (residuals - residuals.min()) / (residuals.max() - residuals.min() + _EPSILON)

    separation_figure = plt.figure(figsize = (8, 6))
    separation_axis = separation_figure.add_subplot(1, 1, 1)

    min_val = min(true_xsec.min(), pred_xsec.min())
    max_val = max(true_xsec.max(), pred_xsec.max())
    separation_axis.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth = 2, label = "Perfect Fit")

    separation_scatterplot = separation_axis.scatter(
        true_xsec, pred_xsec,
        c = residuals_normalized, cmap = "RdYlGn_r", alpha = 0.7)

    colorbar = separation_figure.colorbar(separation_scatterplot, ax = separation_axis)
    colorbar.set_label("Normalized Residual", fontsize = 16)

    separation_axis.set_xlabel("True Cross Section", rotation = 0, fontsize = 18)
    separation_axis.set_ylabel("Predicted Cross Section", fontsize = 18)
    separation_axis.set_title("Model Fit: Prediction vs. Ground Truth", rotation = 0, fontsize = 20)
    separation_axis.grid(True)

    for extension in ['png', 'eps']:
        separation_figure.savefig(
            fname = f"./local/version_{MAJOR_MINOR_NUMBER}/plots/surrogate_dnn_performance_v{MAJOR_MINOR_NUMBER}.{extension}",
            facecolor = 'white',
            transparent = False)
        
    plt.close(separation_figure)
    del dnn_model
