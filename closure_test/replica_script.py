##########################################
# FILE INFORMATION:
# Purpose: runs *a* replica based
# Created: 20260107
# Last changed: 20260108
##########################################

print(f"[INFO]: Script began running!")

##########################################
# [IMPORTANT]: Static quantities parametrizing
# the program. Change these if you need!
##########################################

SCRATCH_PATH = 'placeholder'

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

TEST_LEPTON_HELICITY = 0.0
TEST_TARGET_POLARIZATION = 0.0

print(f"[INFO]: Detected lepton helicity to be: {'unpolarized' if TEST_LEPTON_HELICITY == 0.0 else 'polarized'}")

_DNN_TRAINING_TEMPORARY_SPLIT_PERCENTAGE = 0.1
_DNN_VALIDATION_TESTING_SPLIT_PERCENTAGE = 0.1

print(f"[INFO]: Detected training/temporary data split percentage to be {100. * _DNN_TRAINING_TEMPORARY_SPLIT_PERCENTAGE}%")
print(f"[INFO]: Detected validation/testing data split percentage to be {100. * _DNN_VALIDATION_TESTING_SPLIT_PERCENTAGE}%")

_INITIALIZER_MINIMUM_VALUE = -0.1
_INITIALIZER_MAXMIMUM_VALUE = 0.1

_NUMBER_NODES_HIDDEN_1 = 10
_NUMBER_NODES_HIDDEN_2 = 10
_NUMBER_NODES_HIDDEN_3 = 10
_NUMBER_NODES_HIDDEN_4 = 10

_NUMBER_OF_EPOCHS = 750
_BATCH_SIZE = 25

print(f"[INFO]: Each replica has a batch size of {_BATCH_SIZE}")
print(f"[INFO]: Each replica will run for {_NUMBER_OF_EPOCHS} if assuming no EarlyStopping.")

print(f"[INFO]: Replica DNN initializer minimum value set to {_INITIALIZER_MINIMUM_VALUE}")
print(f"[INFO]: Replica DNN initializer maximum value set to {_INITIALIZER_MAXMIMUM_VALUE}")

print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_1} nodes in 1st hidden layer")
print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_2} nodes in 2nd hidden layer")
print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_3} nodes in 3rd hidden layer")
print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_4} nodes in 4th hidden layer")

##########################################
# Importing Python Libraries
##########################################

import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

print(f"[INFO]: Libraries imported!")

replica_index = sys.argv[1]
replica_number = int(replica_index) + 1

print(f"[INFO]: This replica number is: Replica #{replica_number}")

##########################################
# Matplotlib Plotting Customizability
##########################################

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

##########################################
# Tensorflow Configuration
##########################################

_FLOATX = tf.float32

print(f"[INFO]: Physical devices available to TF are: {tf.config.list_physical_devices()}")
print(f"[INFO]: Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"[INFO]: Number of CPUs Available: {len(tf.config.list_physical_devices('CPU'))}")
print(f"[INFO]: Checking the GPU name: {tf.test.gpu_device_name()}")

##########################################
# Reading the Datafile:
##########################################

test_dataframe = pd.read_csv(
    filepath_or_buffer = f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/data/main_pseudodata_file_v{MAJOR_MINOR_NUMBER}.csv"
)

x_data = test_dataframe[["t", "x_b", "q_squared", "phi"]]
y_data = test_dataframe[["cross_section","beam_spin_asymmetry"]]

TOTAL_DATA_SIZE = len(x_data)
print(f"[INFO]: Total data size is: {TOTAL_DATA_SIZE}")

FIXED_BEAM_ENERGY = test_dataframe["k"].iloc[0]
FIXED_Q_SQUARED = test_dataframe["q_squared"].iloc[0]
FIXED_X_BJORKEN = test_dataframe["x_b"].iloc[0]
FIXED_T_VALUE = test_dataframe["t"].iloc[0]

print(f"[INFO]: Selected beam energy k =  {FIXED_BEAM_ENERGY}")
print(f"[INFO]: Selected Q^2 = {FIXED_Q_SQUARED}")
print(f"[INFO]: Selected xB = {FIXED_X_BJORKEN}")
print(f"[INFO]: Selected t = {FIXED_T_VALUE}")

CFF_REAL_H_KM15 = test_dataframe["Re[H]"].iloc[0]
CFF_IMAG_H_KM15 = test_dataframe["Im[H]"].iloc[0]
CFF_REAL_E_KM15 = test_dataframe["Re[E]"].iloc[0]
CFF_IMAG_E_KM15 = test_dataframe["Im[E]"].iloc[0]
CFF_REAL_HT_KM15 = test_dataframe["Re[Ht]"].iloc[0]
CFF_IMAG_HT_KM15 = test_dataframe["Im[Ht]"].iloc[0]
CFF_REAL_ET_KM15 = test_dataframe["Re[Et]"].iloc[0]
CFF_IMAG_ET_KM15 = test_dataframe["Im[Et]"].iloc[0]

CFF_H_KM15 = complex(CFF_REAL_H_KM15, CFF_IMAG_H_KM15)
CFF_H_TILDE_KM15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
CFF_E_KM15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
CFF_E_TILDE_KM15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

print(f"[INFO]: Selected CFF H = {CFF_H_KM15}")
print(f"[INFO]: Selected CFF E = {CFF_E_KM15}")
print(f"[INFO]: Selected CFF Ht = {CFF_H_TILDE_KM15}")
print(f"[INFO]: Selected CFF Et = {CFF_E_TILDE_KM15}")

number_of_dnn_temporary_points = int(np.ceil(TOTAL_DATA_SIZE * _DNN_TRAINING_TEMPORARY_SPLIT_PERCENTAGE))
number_of_dnn_testing_points = int(np.ceil(number_of_dnn_temporary_points * _DNN_VALIDATION_TESTING_SPLIT_PERCENTAGE))
number_of_dnn_validation_points = number_of_dnn_temporary_points - number_of_dnn_testing_points
number_of_dnn_training_points = TOTAL_DATA_SIZE - number_of_dnn_validation_points - number_of_dnn_testing_points

print(f"[INFO]: Train/Validation Split is {_DNN_TRAINING_TEMPORARY_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_validation_points} validation points (with ceiling).")
print(f"[INFO]: Validation/Test Split is {_DNN_VALIDATION_TESTING_SPLIT_PERCENTAGE * 100}%, giving {number_of_dnn_testing_points} testing points (with ceiling).")
print(f"[INFO]: Remaining training data points are: {number_of_dnn_training_points}")

x_training, x_remaining, y_training, y_remaining = train_test_split(
    x_data, y_data,
    test_size = _DNN_TRAINING_TEMPORARY_SPLIT_PERCENTAGE,
    shuffle = True)

x_validation, x_testing, y_validation, y_testing = train_test_split(
    x_remaining, y_remaining,
    test_size = _DNN_VALIDATION_TESTING_SPLIT_PERCENTAGE,
    shuffle = True)

print(f"[INFO]: Detected size of x training: {len(x_training)}")
assert len(x_training) == number_of_dnn_training_points, "[ASSERT]: Mismatch between expected x-training size and computed size."

print(f"[INFO]: Detected size of y training: {len(y_training)}")
assert len(y_training) == number_of_dnn_training_points, "[ASSERT]: Mismatch between expected y-training size and computed size."

print(f"[INFO]: Detected size of x validation: {len(x_validation)}")
assert len(x_validation) == number_of_dnn_validation_points, "[ASSERT]: Mismatch between expected x-validation size and computed size."

print(f"[INFO]: Detected size of y validation: {len(y_validation)}")
assert len(y_validation) == number_of_dnn_validation_points, "[ASSERT]: Mismatch between expected y-validation size and computed size."

print(f"[INFO]: Detected size of x testing: {len(x_testing)}")
assert len(x_testing) == number_of_dnn_testing_points, "[ASSERT]: Mismatch between expected x-testing size and computed size."

print(f"[INFO]: Detected size of x testing: {len(y_testing)}")
assert len(y_testing) == number_of_dnn_testing_points, "[ASSERT]: Mismatch between expected y-testing size and computed size."

##########################################
# Finding Observable Values Labeled by
# Training/Validation/Testing
##########################################

# a numpy array of all the corresponding phi-points
# if you do NOT UNDERSTAND why the code below selects the phi-values that are used in training, please run each piece interactively
x_training_phi_points = np.array(x_training["phi"])
x_validation_phi_points = np.array(x_validation["phi"])
x_testing_phi_points = np.array(x_testing["phi"])

xsecs = DifferentialCrossSection(
    configuration = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k = FIXED_BEAM_ENERGY,
            squared_Q_momentum_transfer = FIXED_Q_SQUARED,
            x_Bjorken = FIXED_X_BJORKEN,
            squared_hadronic_momentum_transfer_t = FIXED_T_VALUE),
        "cff_inputs": CFFInputs(
            compton_form_factor_h = CFF_H_KM15,
            compton_form_factor_h_tilde = CFF_H_TILDE_KM15,
            compton_form_factor_e = CFF_E_KM15,
            compton_form_factor_e_tilde = CFF_E_TILDE_KM15),
        "target_polarization": 0.0,
        "lepton_beam_polarization": 0.0,
        "using_ww": True
    },
    verbose = False,
    debugging = False)

x_training_bkm10_xsec = xsecs.compute_cross_section(x_training_phi_points).real
x_validation_bkm10_xsec = xsecs.compute_cross_section(x_validation_phi_points).real
x_testing_bkm10_xsec = xsecs.compute_cross_section(x_testing_phi_points).real

x_training_bkm10_bsa = xsecs.compute_bsa(x_training_phi_points).real
x_validation_bkm10_bsa = xsecs.compute_bsa(x_validation_phi_points).real
x_testing_bkm10_bsa = xsecs.compute_bsa(x_testing_phi_points).real

title_string = (
    rf"$Q^2 = {FIXED_Q_SQUARED:.2f}$ GeV$^2$, "
    rf"$x_B = {FIXED_X_BJORKEN:.2f}$, "
    rf"$t = {FIXED_T_VALUE:.2f}$ ,"
    rf"$k = {FIXED_BEAM_ENERGY:.2f}$ GeV"
)
km15_cff_string = (
    rf"$\mathcal{{H}} = {CFF_H_KM15:.3f}$, "
    rf"$\mathcal{{E}} = {CFF_E_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{H}}}} = {CFF_H_TILDE_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{E}}}} = {CFF_E_TILDE_KM15:.3f}$ "
)

data_vis_figure, data_vis_axis = plt.subplots(1, 1, figsize = (8, 7))
data_vis_axis.scatter(x_training_phi_points, x_training_bkm10_xsec, color = 'green', s = 4., label = rf"(KM15) Training Points $N = {len(x_training_phi_points)}$")
data_vis_axis.scatter(x_validation_phi_points, x_validation_bkm10_xsec, color = 'orange', s = 4., label = rf"(KM15) Validation Points $N = {len(x_validation_phi_points)}$")
data_vis_axis.scatter(x_testing_phi_points, x_testing_bkm10_xsec, color = 'red', s = 4., label = rf"(KM15) Testing Points $N = {len(x_testing_phi_points)}$")
data_vis_axis.legend(fontsize = 14.)
data_vis_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16.)
data_vis_axis.set_ylabel(r"$d^{4}\sigma$", fontsize = 16.)
data_vis_axis.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
data_vis_axis.grid(visible = True)
data_vis_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/training_set_xsec_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png")
data_vis_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/training_set_xsec_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(data_vis_figure)

bsa_train_test_split_figure, bsa_train_test_split_axis = plt.subplots(1, 1, figsize = (8, 7))
bsa_train_test_split_axis.scatter(x_training_phi_points, x_training_bkm10_bsa, color = 'green', s = 4., label = rf"(KM15) Training Points $N = {len(x_training_phi_points)}$")
bsa_train_test_split_axis.scatter(x_validation_phi_points, x_validation_bkm10_bsa, color = 'orange', s = 4., label = rf"(KM15) Validation Points $N = {len(x_validation_phi_points)}$")
bsa_train_test_split_axis.scatter(x_testing_phi_points, x_testing_bkm10_bsa, color = 'red', s = 4., label = rf"(KM15) Testing Points $N = {len(x_testing_phi_points)}$")
bsa_train_test_split_axis.legend(fontsize = 14.)
bsa_train_test_split_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16.)
bsa_train_test_split_axis.set_ylabel(r"BSA", fontsize = 16.)
bsa_train_test_split_axis.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
bsa_train_test_split_axis.grid(visible = True)
bsa_train_test_split_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/training_set_bsa_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png")
bsa_train_test_split_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/training_set_bsa_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(bsa_train_test_split_figure)

##########################################
# DNN Model Setup
##########################################

_CONVERSION_GEV6_GEV4NB = tf.constant(.389379 * 1000000., dtype = _FLOATX)
_MASS_OF_PROTON_IN_GEV = tf.constant(0.93827208816, dtype = _FLOATX)
_QED_FINE_STRUCTURE = tf.constant(1./137.035999177, dtype = _FLOATX)
_ELECTRIC_FORM_FACTOR_CONSTANT = tf.constant(0.710649, dtype = _FLOATX)
_PROTON_MAGNETIC_MOMENT = tf.constant(2.79284734463, dtype = _FLOATX)

_LAB_K_BEAM = tf.constant(test_dataframe["k"].iloc[0], dtype = _FLOATX)
print(f"[INFO]: Chosen fixed k to be {_LAB_K_BEAM}")

assert test_dataframe["q_squared"].iloc[0] == test_dataframe["q_squared"].iloc[5], "[ASSERT]: iloc revealed kinematic sub-dataframe not invariant under index."
assert test_dataframe["k"].iloc[0] == test_dataframe["k"].iloc[10], "[ASSERT]: iloc revealed kinematic sub-dataframe not invariant under index."

def compute_fe(t):
    return tf.divide(1., tf.square(1. - tf.divide(t, _ELECTRIC_FORM_FACTOR_CONSTANT)))

def compute_fg(fe):
    return _PROTON_MAGNETIC_MOMENT * fe

def compute_f2(t, fe, fg):
    tau = tf.divide(-1. * t, 4. * tf.square(_MASS_OF_PROTON_IN_GEV))
    numerator = fg - fe
    denominator = 1. + tau
    return tf.divide(numerator, denominator)

def compute_f1(fg, f2):
    return fg - f2

def compute_epsilon(xb, q_squared):
    return tf.divide(2. * xb * _MASS_OF_PROTON_IN_GEV, tf.sqrt(q_squared))

def compute_y(k_beam, q_squared, ep):
    return tf.sqrt(q_squared) / (ep * k_beam)

def compute_skewness(xb, t, q_squared):
    return xb * (1. + (t / (2. * q_squared))) / (2. - xb + (xb * t / q_squared))

def compute_t_min(xb, q_squared, ep):
    return -1. * q_squared * ((2. * (1. - xb) * (1. - tf.sqrt(1. + ep**2))) + ep**2) / ((4. * xb * (1. - xb)) + ep**2)

def compute_t_prime(t, tmin):
    return (t-tmin)

def compute_k_tilde(xb, q_squared, t, tmin, ep):
    return tf.sqrt(tmin - t) * tf.sqrt(((1. - xb) * tf.sqrt((1. + ep**2))) + (((tmin - t) * (ep**2 + (4. * (1. - xb) * xb))) / (4. * q_squared)))

def compute_k(q_squared, y_lep, ep, k_tilde):
    return tf.sqrt(((1. - y_lep + (ep**2 * y_lep**2 / 4.)) / q_squared)) * k_tilde

def compute_k_dot_delta(q_squared, xb, t, phi_azi, ep, y_lep, k):
    return (-1.*q_squared / (2.*y_lep*(1.+ep**2))) * (1. + ((2.*k*tf.cos(tf.constant(np.pi)- phi_azi)) - ((t / q_squared)*(1.-(xb * (2. - y_lep)) + (y_lep * ep**2 / 2.))) + (y_lep * ep**2 / 2.)))

def prop_1(q_squared, kdd):
    return (1. + (2. * (kdd / q_squared)))

def prop_2(q_squared, t, kdd):
    return ((-2. * (kdd / q_squared)) + (t / q_squared))

def bh_unp_c0(
    q_sq: float, xb: float, t: float, ep: float,
    y: float, k: float, f1: float, f2: float):
    first_line = 8. * k**2 * (((2. + 3. * ep**2) * (f1**2 - (t * f2**2 / (4. * _MASS_OF_PROTON_IN_GEV**2))) / (t / q_sq)) + (2. * xb**2 * (f1 + f2)**2))
    second_line_first_part = (2. + ep**2) * ((4. * xb**2 * _MASS_OF_PROTON_IN_GEV**2 / t) * (1. + (t / q_sq))**2 + 4. * (1 - xb) * (1. + (xb * (t / q_sq)))) * (f1**2 - (t * f2**2 / (4. * _MASS_OF_PROTON_IN_GEV**2)))
    second_line_second_part = 4. * xb**2 * (xb + (1. - xb + (ep**2 / 2.)) * (1 - (t / q_sq))**2 - xb * (1. - 2. * xb) * (t / q_sq)**2) * (f1 + f2)**2
    second_line = (2. - y)**2 * (second_line_first_part + second_line_second_part)
    third_line = 8. * (1. + ep**2) * (1. - y - (ep**2 * y**2 / 4.)) * (2. * ep**2 * (1 - (t / (4. * _MASS_OF_PROTON_IN_GEV**2))) * (f1**2 - (t * f2**2 / (4. * _MASS_OF_PROTON_IN_GEV**2))) - xb**2 * (1 - (t / q_sq))**2 * (f1 + f2)**2)
    c0_unpolarized_bh = first_line + second_line + third_line
    return c0_unpolarized_bh

def bh_unp_c1(
    q_sq: float, xb: float, t: float, ep: float,
    y: float, k: float, f1: float, f2: float) -> float:
    addition_of_form_factors_squared = (f1 + f2)**2
    weighted_combination_of_form_factors = f1**2 - ((t / (4. * _MASS_OF_PROTON_IN_GEV**2)) * f2**2)
    first_line_first_part = ((4. * xb**2 * _MASS_OF_PROTON_IN_GEV**2 / t) - 2. * xb - ep**2) * weighted_combination_of_form_factors
    first_line_second_part = 2. * xb**2 * (1. - (1. - 2. * xb) * (t / q_sq)) * addition_of_form_factors_squared
    c1_unpolarized_bh = 8. * k * (2. - y) * (first_line_first_part + first_line_second_part)
    return c1_unpolarized_bh

def bh_unp_c2( 
    xb: float, t: float, k: float, f1: float, f2: float) -> float:
    addition_of_form_factors_squared = (f1 + f2)**2
    weighted_combination_of_form_factors = f1**2 - ((t/ (4. * _MASS_OF_PROTON_IN_GEV**2)) * f2**2)
    first_part_of_contribution = (4. * _MASS_OF_PROTON_IN_GEV**2 / t) * weighted_combination_of_form_factors
    c2_unpolarized_bh = 8. * xb**2 * k**2 * (first_part_of_contribution + 2. * addition_of_form_factors_squared)
    return c2_unpolarized_bh

def bh_squared(lep_helicity, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2):

    if lep_helicity == 0.0 and target_polar == 0.0:
        bh_c0 = 0.5 * (bh_unp_c0(q_sq, xb, t, ep, y, k, f1, f2) + bh_unp_c0(q_sq, xb, t, ep, y, k, f1, f2))
        bh_c1 = 0.5 * (bh_unp_c1(q_sq, xb, t, ep, y, k, f1, f2) + bh_unp_c1(q_sq, xb, t, ep, y, k, f1, f2))
        bh_c2 = 0.5 * (bh_unp_c2(xb, t, k, f1, f2) + bh_unp_c2(xb, t, k, f1, f2))
    elif lep_helicity == 1.0 and target_polar == 0.0:
        bh_c0 = bh_unp_c0(q_sq, xb, t, ep, y, k, f1, f2)
        bh_c1 = bh_unp_c1(q_sq, xb, t, ep, y, k, f1, f2)
        bh_c2 = bh_unp_c2(xb, t, k, f1, f2)
    elif lep_helicity == -1.0 and target_polar == 0.0: 
        bh_c0 = bh_unp_c0(q_sq, xb, t, ep, y, k, f1, f2)
        bh_c1 = bh_unp_c1(q_sq, xb, t, ep, y, k, f1, f2)
        bh_c2 = bh_unp_c2(xb, t, k, f1, f2)
    elif lep_helicity == 1.0 and target_polar == 1.0:
        raise NotImplementedError("[ERROR]: NO POLARIZED TARGET YET!")
    
    return ((
        bh_c0 + 
        bh_c1 * tf.cos(1.* (tf.constant(np.pi) - phi)) + 
        bh_c2 * tf.cos(2.* (tf.constant(np.pi) - phi))) / (xb * xb * y * y * (1.+ep**2)**2 * t * p1 * p2))

def f_eff(xi: float, cff: complex, use_ww: bool = True):
    if use_ww:
        cff_effective = 2. * cff / (1. + xi)
    else:
        cff_effective = -2. * xi * cff / (1. + xi)
    return cff_effective

def curly_c_real(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):
    
    first_line = (4.*(1.-xb)*(cff_re_h*cff_re_h_star - cff_im_h*cff_im_h_star)) + (4.*(1.-xb + 0.25*((2.*q_sq + t)*ep**2)/(q_sq + xb*t))*(cff_re_ht * cff_re_ht_star - cff_im_ht * cff_im_ht_star))
    next_line = -xb**2*(q_sq+t)**2*(cff_re_h*cff_re_e_star - cff_im_e*cff_im_h_star + cff_re_e*cff_re_h_star - cff_im_h*cff_im_e_star)/(q_sq*(q_sq+xb*t)) - (xb**2*q_sq*(cff_re_ht*cff_re_et_star - cff_im_et *cff_im_ht_star + cff_re_et*cff_re_ht_star - cff_im_ht*cff_im_et_star)/(q_sq+xb*t))
    final_line = -1.*(xb**2*(q_sq+t)**2/(q_sq*(q_sq+xb*t)) + 0.25*((2.-xb)*q_sq+xb*t)**2*t/(q_sq*_MASS_OF_PROTON_IN_GEV**2*(q_sq+xb*t)))*(cff_re_e*cff_re_e_star - cff_im_e*cff_im_e_star) -0.25*xb**2*q_sq*t*(cff_re_et*cff_re_et_star - cff_im_et*cff_im_et_star)/((q_sq+xb*t)*_MASS_OF_PROTON_IN_GEV**2)

    return ((first_line + next_line + final_line)*q_sq*(q_sq+xb*t)/((2.-xb)*q_sq+xb*t)**2)

def curly_c_imag(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):
    
    first_line = (4.*(1.-xb)*(cff_im_h*cff_re_h_star + cff_re_h*cff_im_h_star)) + (4.*(1.-xb + 0.25*(2.*q_sq + t)*ep**2/(q_sq + xb*t))*(cff_im_ht * cff_re_ht_star + cff_re_ht * cff_im_ht_star))
    next_line = -xb**2*(q_sq+t)**2*(cff_im_h*cff_re_e_star + cff_re_e*cff_im_h_star + cff_im_e*cff_re_h_star + cff_re_h*cff_im_e_star)/(q_sq*(q_sq+xb*t)) - (xb**2*q_sq*(cff_im_ht*cff_re_et_star + cff_re_et*cff_im_ht_star + cff_im_et*cff_re_ht_star + cff_re_ht*cff_im_et_star)/(q_sq+xb*t))
    final_line = -1.*(xb**2*(q_sq+t)**2/(q_sq*(q_sq+xb*t)) + 0.25*((2.-xb)*q_sq+xb*t)**2*t/(q_sq*_MASS_OF_PROTON_IN_GEV**2*(q_sq+xb*t)))*(cff_im_e*cff_re_e_star + cff_re_e*cff_im_e_star) -0.25*xb**2*q_sq*t*(cff_im_et*cff_re_et_star + cff_re_et*cff_im_et_star)/((q_sq+xb*t)*_MASS_OF_PROTON_IN_GEV**2)

    return ((first_line + next_line + final_line)*q_sq*(q_sq+xb*t)/((2.-xb)*q_sq+xb*t)**2)

def dvcs_unp_c0(
    q_sq: float, xb: float, t: float,ep: float,y: float, xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:
    
    first_term_prefactor = 2. * ( 2. - 2. * y + y**2 + (ep**2 * y**2 / 2.)) / (1. + ep**2)
    second_term_prefactor = 16. * k**2 / ((2. - xb)**2 * (1. + ep**2))
    first_term_curlyc = curly_c_real(
        q_sq, xb, t, ep,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        cff_im_h, cff_im_ht, cff_im_e, cff_im_et,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    second_term_curlyc = curly_c_real(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_e, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, -1.*cff_im_h, use_ww), f_eff(xi, -1.*cff_im_ht, use_ww), f_eff(xi, -1.*cff_im_e, use_ww), f_eff(xi, -1.*cff_im_et, use_ww))
    c0_dvcs_unpolarized_coefficient = first_term_prefactor * first_term_curlyc + second_term_prefactor * second_term_curlyc
    return c0_dvcs_unpolarized_coefficient

def dvcs_unp_c1(
    q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = 8. * k * (2. - y) / ((2. - xb) * (1. + ep**2))
    curlyC_unp_DVCS = curly_c_real(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_unp_s1(
    lepton_helicity: float,q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:
    prefactor = -8. * k * lepton_helicity * y * tf.sqrt(1. + ep**2) / ((2. - xb) * (1. + ep**2))
    curlyC_unp_DVCS = curly_c_imag(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_squared(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, phi,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    if target_polar == 1.0:
        raise NotImplementedError("[ERROR]: NO POLARIZED TARGET YET!")
    
    if lep_helicity == 0.0:
        dvcs_c0 = 0.5 * (
            dvcs_unp_c0(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww) +
            dvcs_unp_c0(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
        )

        dvcs_c1 = 0.5 * (
            dvcs_unp_c1(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww) +
            dvcs_unp_c1(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
        )
        dvcs_s1 = 0.5 * (
            dvcs_unp_s1(1.0, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww) +
            dvcs_unp_s1(-1.0, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
        )
    else:
        dvcs_c0 = dvcs_unp_c0(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
        dvcs_c1 = dvcs_unp_c1(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
        dvcs_s1 = dvcs_unp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    return (
        (dvcs_c0 + 
         dvcs_c1 * tf.cos(1.* (tf.constant(np.pi) - phi)) + 
         dvcs_s1 * tf.sin(1.* (tf.constant(np.pi) - phi))) / (y * y * q_sq))

def i_c_unp_pp_0(
    q_sq: float,xb: float,t: float,ep: float,y: float,k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    two_minus_xb = 2. - xb
    two_minus_y = 2. - y
    first_term_in_brackets = k_tilde**2 * two_minus_y**2 / (q_sq * root_one_plus_epsilon_squared)
    second_term_in_brackets_first_part = t_over_Q_squared * two_minus_xb * (1. - y - (ep**2 * y**2 / 4.))
    second_term_in_brackets_second_part_numerator = 2. * xb * t_over_Q_squared * (two_minus_xb + 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep**2 / xb) + ep**2
    second_term_in_brackets_second_part =  1. + second_term_in_brackets_second_part_numerator / (two_minus_xb * one_plus_root_epsilon_stuff)
    prefactor = -4. * two_minus_y * one_plus_root_epsilon_stuff / tf.pow(root_one_plus_epsilon_squared, 4)
    c_0_plus_plus_unp = prefactor * (first_term_in_brackets + second_term_in_brackets_first_part * second_term_in_brackets_second_part)
    return c_0_plus_plus_unp

def i_c_unp_v_pp_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_term_in_brackets = (2. - y)**2 * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)
    second_term_first_multiplicative_term = 1. - y - (ep**2 * y**2 / 4.)
    second_term_second_multiplicative_term = one_plus_root_epsilon_stuff / 2.
    second_term_third_multiplicative_term = 1. + t_over_Q_squared
    second_term_fourth_multiplicative_term = 1. + (root_one_plus_epsilon_squared - 1. + (2. * xb)) * t_over_Q_squared / one_plus_root_epsilon_stuff
    second_term_in_brackets = second_term_first_multiplicative_term * second_term_second_multiplicative_term * second_term_third_multiplicative_term * second_term_fourth_multiplicative_term
    coefficient_prefactor = 8. * (2. - y) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_0_plus_plus_V_unp = coefficient_prefactor * (first_term_in_brackets + second_term_in_brackets)
    return c_0_plus_plus_V_unp

def i_c_unp_a_pp_0(
    q_sq: float,xb: float,t: float,ep: float,y: float,k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    two_minus_y = 2. - y
    ktilde_over_Q_squared = k_tilde**2 / q_sq
    curly_bracket_first_term = two_minus_y**2 * ktilde_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb) / (2. * root_one_plus_epsilon_squared)
    deepest_parentheses_term = (xb * (2. + one_plus_root_epsilon_stuff - 2. * xb) / one_plus_root_epsilon_stuff + (one_plus_root_epsilon_stuff - 2.)) * t_over_Q_squared
    square_bracket_term = one_plus_root_epsilon_stuff * (one_plus_root_epsilon_stuff - xb + deepest_parentheses_term) / 2. - (2. * ktilde_over_Q_squared)
    curly_bracket_second_term = (1. - y - ep**2 * y**2 / 4.) * square_bracket_term
    coefficient_prefactor = 8. * two_minus_y * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_0_plus_plus_A_unp = coefficient_prefactor * (curly_bracket_first_term + curly_bracket_second_term)
    return c_0_plus_plus_A_unp

def i_c_unp_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    bracket_quantity = ep**2 + t * (2. - 6.* xb - ep**2) / (3. * q_sq)
    prefactor = 12. * tf.sqrt(2.) * k * (2. - y) * tf.sqrt(1. - y - (ep**2 * y**2 / 4)) / tf.pow(1. + ep**2, 2.5)
    c_0_zero_plus_unp = prefactor * bracket_quantity
    return c_0_zero_plus_unp

def i_c_unp_v_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    t_over_Q_squared = t / q_sq
    main_part = xb * t_over_Q_squared * (1. - (1. - 2. * xb) * t_over_Q_squared)
    prefactor = 24. * tf.sqrt(2.) * k * (2. - y) * tf.sqrt(1. - y - (y**2 * ep**2 / 4.)) / (1. + ep**2)**2.5
    c_0_zero_plus_V_unp = prefactor * main_part
    return c_0_zero_plus_V_unp

def i_c_unp_a_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    t_over_Q_squared = t / q_sq
    fancy_xb_epsilon_term = 8. - 6. * xb + 5. * ep**2
    brackets_term = 1. - t_over_Q_squared * (2. - 12. * xb * (1. - xb) - ep**2) / fancy_xb_epsilon_term
    prefactor = 4. * tf.sqrt(2.) * k * (2. - y) * tf.sqrt(1. - y - (y**2 * ep**2 / 4.)) / tf.pow(1. + ep**2, 2.5)
    c_0_zero_plus_A_unp = prefactor * t_over_Q_squared * fancy_xb_epsilon_term * brackets_term
    return c_0_zero_plus_A_unp

def i_c_unp_pp_1(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_first_term = (1. + (1. - xb) * (root_one_plus_epsilon_squared - 1.) / (2. * xb) + ep**2 / (4. * xb)) * xb * t_over_Q_squared
    first_bracket_term = first_bracket_first_term - 3. * ep**2 / 4.
    second_bracket_term = 1. - (1. - 3. * xb) * t_over_Q_squared + (1. - root_one_plus_epsilon_squared + 3. * ep**2) * xb * t_over_Q_squared / (one_plus_root_epsilon_stuff - ep**2)
    fancy_y_coefficient = 2. - 2. * y + y**2 + ep**2 * y**2 / 2.
    second_term = -4. * k * fancy_y_coefficient * (one_plus_root_epsilon_stuff - ep**2) * second_bracket_term / root_one_plus_epsilon_squared**5
    first_term = -16. * k * (1. - y - ep**2 * y**2 / 4.) * first_bracket_term / root_one_plus_epsilon_squared**5
    c_1_plus_plus_unp = first_term + second_term
    return c_1_plus_plus_unp

def i_c_unp_v_pp_1(
    q_sq: float, xb: float, t: float,ep: float,y: float,tp: float,k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    first_bracket_term = (2. - y)**2 * (1. - (1. - 2. * xb) * t_over_Q_squared)
    second_bracket_term_first_part = 1. - y - ep**2 * y**2 / 4.
    second_bracket_term_second_part = 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb) * tp / q_sq
    coefficient_prefactor = 16. * k * xb * t_over_Q_squared / tf.pow(root_one_plus_epsilon_squared, 5)
    c_1_plus_plus_V_unp = coefficient_prefactor * (first_bracket_term + second_bracket_term_first_part * second_bracket_term_second_part)
    return c_1_plus_plus_V_unp

def i_c_unp_a_pp_1(
    q_sq: float, xb: float, t: float,ep: float,y: float,  tp: float,k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = tp / q_sq
    one_minus_xb = 1. - xb
    one_minus_2xb = 1. - 2. * xb
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    first_bracket_term_second_part = 1. - one_minus_2xb * t_over_Q_squared + (4. * xb * one_minus_xb + ep**2) * t_prime_over_Q_squared / (4. * root_one_plus_epsilon_squared)
    second_bracket_term = 1. - 0.5 * xb + 0.25 * (one_minus_2xb + root_one_plus_epsilon_squared) * (1. - t_over_Q_squared) + (4. * xb * one_minus_xb + ep**2) * t_prime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -16. * k * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_1_plus_plus_A_unp = prefactor * (fancy_y_stuff * first_bracket_term_second_part - (2. - y)**2 * second_bracket_term)
    return c_1_plus_plus_A_unp

def i_c_unp_0p_1(
    q_sq: float, xb: float, t: float,ep: float,y: float, tp: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = tp / q_sq
    one_minus_xb = 1. - xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    first_bracket_term = (2. - y)**2 * t_prime_over_Q_squared * (one_minus_xb + (one_minus_xb * xb + (ep**2 / 4.)) * t_prime_over_Q_squared / root_one_plus_epsilon_squared)
    second_bracket_term = y_quantity * (1. - (1. - 2. * xb) * t_over_Q_squared) * (ep**2 - 2. * (1. + (ep**2 / (2. * xb))) * xb * t_over_Q_squared) / root_one_plus_epsilon_squared
    prefactor = 8. * tf.sqrt(2. * y_quantity) / root_one_plus_epsilon_squared**4
    c_1_zero_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_1_zero_plus_unp

def i_c_unp_v_0p_1(
    q_sq: float, xb: float, t: float,ep: float,y: float, k_tilde: float):
    t_over_Q_squared = t / q_sq
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    major_part = (2 - y)**2 * k_tilde**2 / q_sq + (1. - (1. - 2. * xb) * t_over_Q_squared)**2 * y_quantity
    prefactor = 16. * tf.sqrt(2. * y_quantity) * xb * t_over_Q_squared / (1. + ep**2)**2.5
    c_1_zero_plus_V_unp = prefactor * major_part
    return c_1_zero_plus_V_unp

def i_c_unp_a_0p_1(
    q_sq: float, xb: float, t: float,ep: float,y: float, k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_minus_2xb = 1. - 2. * xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    second_term_first_part = (1. - one_minus_2xb * t_over_Q_squared) * y_quantity
    second_term_second_part = 4. - 2. * xb + 3. * ep**2 + t_over_Q_squared * (4. * xb * (1. - xb) + ep**2)
    first_term = k_tilde**2 * one_minus_2xb * (2. - y)**2 / q_sq
    prefactor = 8. * tf.sqrt(2. * y_quantity) * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_1_zero_plus_unp_A = prefactor * (first_term + second_term_first_part * second_term_second_part)
    return c_1_zero_plus_unp_A

def i_s_unp_pp_1(lepton_helicity: float,q_sq: float, xb: float, ep: float,y: float,tp: float,k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    tPrime_over_Q_squared = tp / q_sq
    bracket_term = 1. + ((1. - xb + 0.5 * (root_one_plus_epsilon_squared - 1.)) / root_one_plus_epsilon_squared**2) * tPrime_over_Q_squared
    prefactor = 8. * lepton_helicity * k * y * (2. - y) / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp = prefactor * bracket_term
    return s_1_plus_plus_unp

def i_s_unp_v_pp_1(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    bracket_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb) * t_over_Q_squared
    prefactor = -8. * lepton_helicity * k * y * (2. - y) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_1_plus_plus_unp_V = prefactor * bracket_term
    return s_1_plus_plus_unp_V

def i_s_unp_a_pp_1(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float, tp: float,k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    tPrime_over_Q_squared = tp / q_sq
    one_minus_2xb = 1. - 2. * xb
    bracket_term = 1. - one_minus_2xb * (one_minus_2xb + root_one_plus_epsilon_squared) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = 8. * lepton_helicity * k * y * (2. - y) * t_over_Q_squared / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp_A = prefactor * bracket_term
    return s_1_plus_plus_unp_A

def i_s_unp_0p_1(
    lepton_helicity: float,q_sq: float, ep: float,y: float,k_tilde: float):
    root_one_plus_epsilon_squared = (1. + ep**2)**2
    y_quantity = tf.sqrt(1. - y - (ep**2 * y**2 / 4.))
    s_1_zero_plus_unp = 8. * tf.sqrt(2.) * lepton_helicity * (2. - y) * y * y_quantity * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)
    return s_1_zero_plus_unp

def i_s_unp_v_0p_1(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float):
    one_plus_epsilon_squared_squared = (1. + ep**2)**2
    t_over_Q_squared = t / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    bracket_term = 4. * (1. - 2. * xb) * t_over_Q_squared * (1. + xb * t_over_Q_squared) + ep**2 * (1. + t_over_Q_squared)**2
    prefactor = 4. * tf.sqrt(2. * fancy_y_stuff) * lepton_helicity * y * (2. - y) * xb * t_over_Q_squared / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_V = prefactor * bracket_term
    return s_1_zero_plus_unp_V

def i_s_unp_a_0p_1(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    one_plus_epsilon_squared_squared = (1. + ep**2)**2
    fancy_y_stuff = tf.sqrt(1. - y - ep**2 * y**2 / 4.)
    prefactor = -8. * tf.sqrt(2.) * lepton_helicity * y * (2. - y) * (1. - 2. * xb) / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_A = prefactor * fancy_y_stuff * t * k**2 / q_sq
    return s_1_zero_plus_unp_A

def i_c_unp_pp_2(
    q_sq: float, xb: float, t: float,ep: float,y: float,tp: float,k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    first_bracket_term = 2. * ep**2 * k_tilde**2 / (root_one_plus_epsilon_squared * (1. + root_one_plus_epsilon_squared) * q_sq)
    second_bracket_term = xb * tp * t_over_Q_squared * (1. - xb - 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep**2 / xb) / q_sq
    prefactor = 8. * (2. - y) * (1. - y - ep**2 * y**2 / 4.) / root_one_plus_epsilon_squared**4
    c_2_plus_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_2_plus_plus_unp

def i_c_unp_v_pp_2(
    q_sq: float, xb: float, t: float,ep: float,y: float, tp: float,k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = tp / q_sq
    major_term = (4. * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)) + 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb) * (1. + t_over_Q_squared) * t_prime_over_Q_squared
    prefactor = 8. * (2. - y) * (1. - y - ep**2 * y**2 / 4.) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_V_unp = prefactor * major_term
    return c_2_plus_plus_V_unp

def i_c_unp_a_pp_2(
    q_sq: float, xb: float, t: float,ep: float,y: float, tp: float,k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = tp / q_sq
    first_bracket_term = 4. * (1. - 2. * xb) * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)
    second_bracket_term = (3.  - root_one_plus_epsilon_squared - 2. * xb + ep**2 / xb ) * xb * t_prime_over_Q_squared
    prefactor = 4. * (2. - y) * (1. - y - ep**2 * y**2 / 4.) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_A_unp = prefactor * (first_bracket_term - second_bracket_term)
    return c_2_plus_plus_A_unp

def i_c_unp_0p_2(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    epsilon_squared_over_2 = ep**2 / 2.
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb) / (1. + epsilon_squared_over_2)) * xb * t / q_sq
    prefactor = -8. * tf.sqrt(2. * y_quantity) * k * (2. - y) / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return c_2_zero_plus_unp

def i_c_unp_v_0p_2(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    y_quantity = tf.sqrt(1. - y - (ep**2 * y**2 / 4.))
    prefactor = 8. * tf.sqrt(2.) * y_quantity * k * (2. - y) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * xb) * t_over_Q_squared)
    return c_2_zero_plus_unp_V

def i_c_unp_a_0p_2(
    q_sq: float, xb: float, t: float,ep: float,y: float, tp: float,k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = tp / q_sq
    one_minus_xb = 1. - xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    bracket_term = one_minus_xb + 0.5 * t_prime_over_Q_squared * (4. * xb * one_minus_xb + ep**2) / root_one_plus_epsilon_squared
    prefactor = 8. * tf.sqrt(2. * y_quantity) * k * (2. - y) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * bracket_term
    return c_2_zero_plus_unp_A

def i_s_unp_pp_2(
    lepton_helicity: float,q_sq: float, xb: float, ep: float,y: float,tp: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    tPrime_over_Q_squared = tp / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    first_bracket_term = (ep**2 - xb * (root_one_plus_epsilon_squared - 1.)) / (1. + root_one_plus_epsilon_squared - 2. * xb)
    second_bracket_term = (2. * xb + ep**2) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -4. * lepton_helicity * fancy_y_stuff * y * (1. + root_one_plus_epsilon_squared - 2. * xb) * tPrime_over_Q_squared / root_one_plus_epsilon_squared**3
    s_2_plus_plus_unp = prefactor * (first_bracket_term - second_bracket_term)
    return s_2_plus_plus_unp

def i_s_unp_v_pp_2(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    one_minus_2xb = 1. - 2. * xb
    bracket_term = root_one_plus_epsilon_squared - 1. + (one_minus_2xb + root_one_plus_epsilon_squared) * t_over_Q_squared
    parentheses_term = 1. - one_minus_2xb * t_over_Q_squared
    prefactor = -4. * lepton_helicity * fancy_y_stuff * y * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_V = prefactor * parentheses_term * bracket_term
    return s_2_plus_plus_unp_V

def i_s_unp_a_pp_2(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float, tp: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    tPrime_over_Q_squared = tp / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    last_term = 1. + (4. * (1. - xb) * xb + ep**2) * t_over_Q_squared / (4. - 2. * xb + 3. * ep**2)
    middle_term = 1. + root_one_plus_epsilon_squared - 2. * xb
    prefactor = -8. * lepton_helicity * fancy_y_stuff * y * t_over_Q_squared * tPrime_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_A = prefactor * middle_term * last_term
    return s_2_plus_plus_unp_A

def i_s_unp_0p_2(
    lepton_helicity: float, q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    epsilon_squared_over_2 = ep**2 / 2.
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb) / (1. + epsilon_squared_over_2)) * xb * t / q_sq
    prefactor = 8. * lepton_helicity * tf.sqrt(2. * y_quantity) * k * y / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return s_2_zero_plus_unp

def i_s_unp_v_0p_2(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    y_quantity = tf.sqrt(1. - y - (ep**2 * y**2 / 4.))
    prefactor = -8. * tf.sqrt(2.) * lepton_helicity * y_quantity * k * y * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * xb) * t_over_Q_squared)
    return s_2_zero_plus_unp_V

def i_s_unp_a_0p_2(
    lepton_helicity: float,q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_minus_xb = 1. - xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    main_term = 4. * one_minus_xb + 2. * ep**2 + 4. * t_over_Q_squared * (4. * xb * one_minus_xb + ep**2)
    prefactor = -2. * tf.sqrt(2. * y_quantity) * lepton_helicity * k * y * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * main_term
    return c_2_zero_plus_unp_A

def i_c_unp_pp_3(
    q_sq: float, xb: float, t: float,ep: float,y: float,k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    major_term = (1. - xb) * t_over_Q_squared + 0.5 * (root_one_plus_epsilon_squared - 1.) * (1. + t_over_Q_squared)
    intermediate_term = (root_one_plus_epsilon_squared - 1.) / root_one_plus_epsilon_squared**5
    prefactor = -8. * k * (1. - y - ep**2 * y**2 / 4.)
    c_3_plus_plus_unp = prefactor * intermediate_term * major_term
    return c_3_plus_plus_unp

def i_c_unp_v_pp_3(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    major_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb) * t_over_Q_squared
    prefactor = -8. * k * (1. - y - ep**2 * y**2 / 4.) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_3_plus_plus_V_unp = prefactor * major_term
    return c_3_plus_plus_V_unp

def i_c_unp_a_pp_3(
    q_sq: float, xb: float, t: float,ep: float,y: float, tp: float,k: float):
    main_term = t * tp * (xb * (1. - xb) + ep**2 / 4.) / q_sq**2
    prefactor = 16. * k * (1. - y - ep**2 * y**2 / 4.) / tf.pow(1. + ep**2, 2.5)
    c_3_plus_plus_A_unp = prefactor * main_term
    return c_3_plus_plus_A_unp

def i_curly_c_unp(
    q_sq: float,xb: float,t: float,f1: float,f2: float,cff_h: float,cff_h_tilde: float,cff_e: float) -> float:
    weighted_cffs = (f1 * cff_h) - (t * f2 * cff_e / (4. * _MASS_OF_PROTON_IN_GEV**2))
    second_term = xb * (f1 + f2) * cff_h_tilde / (2. - xb + (xb * t / q_sq))
    curly_C_unpolarized_interference = weighted_cffs + second_term
    return curly_C_unpolarized_interference

def i_curly_c_v_unp(
    q_sq: float, xb: float,t: float,f1: float,f2: float,cff_h: float,cff_e: float) -> float:
    cff_term = cff_h + cff_e
    second_term = xb * (f1 + f2) / (2. - xb + (xb * t / q_sq))
    curly_C_unpolarized_interference_V = cff_term * second_term
    return curly_C_unpolarized_interference_V

def i_curly_c_a_unp(
    q_sq: float, xb: float,t: float,f1: float,f2: float,cff_h: float) -> float:
    xb_modulation = xb * (f1 + f2) / (2. - xb + (xb * t / q_sq))
    curly_C_unpolarized_interference_A = cff_h * xb_modulation
    return curly_C_unpolarized_interference_A

def i_unp_c0(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_0 = i_c_unp_pp_0(q_sq, xb, t, ep, y, ktilde)
    i_c_pp_v_0 = i_c_unp_v_pp_0(q_sq, xb, t, ep, y, ktilde)
    i_c_pp_a_0 = i_c_unp_a_pp_0(q_sq, xb, t, ep, y, ktilde)

    i_c_0p_0 = i_c_unp_0p_0(q_sq, xb, t, ep, y, k)
    i_c_0p_v_0 = i_c_unp_v_0p_0(q_sq, xb, t, ep, y, k)
    i_c_0p_a_0 = i_c_unp_a_0p_0(q_sq, xb, t, ep, y, k)

    return (i_c_pp_0*i_curly_c + i_c_pp_v_0*i_curly_c_v + i_c_pp_a_0*i_curly_c_a + i_c_0p_0*i_curly_c_eff + i_c_0p_v_0*i_curly_c_eff_v + i_c_0p_a_0*i_curly_c_eff_a)

def i_unp_c1(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_1 = i_c_unp_pp_1(q_sq, xb, t, ep, y, k)
    i_c_pp_v_1 = i_c_unp_v_pp_1(q_sq, xb, t, ep, y, tprime, k)
    i_c_pp_a_1 = i_c_unp_a_pp_1(q_sq, xb, t, ep, y, tprime, k)

    i_c_0p_1 = i_c_unp_0p_1(q_sq, xb, t, ep, y, tprime)
    i_c_0p_v_1 = i_c_unp_v_0p_1(q_sq, xb, t, ep, y, ktilde)
    i_c_0p_a_1 = i_c_unp_a_0p_1(q_sq, xb, t, ep, y, ktilde)

    return (i_c_pp_1*i_curly_c + i_c_pp_v_1*i_curly_c_v + i_c_pp_a_1*i_curly_c_a + i_c_0p_1*i_curly_c_eff + i_c_0p_v_1*i_curly_c_eff_v + i_c_0p_a_1*i_curly_c_eff_a)

def i_unp_c2(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_2 = i_c_unp_pp_2(q_sq, xb, t, ep, y, tprime, ktilde)
    i_c_pp_v_2 = i_c_unp_v_pp_2(q_sq, xb, t, ep, y, tprime, ktilde)
    i_c_pp_a_2 = i_c_unp_a_pp_2(q_sq, xb, t, ep, y, tprime, ktilde)

    i_c_0p_2 = i_c_unp_0p_2(q_sq, xb, t, ep, y, k)
    i_c_0p_v_2 = i_c_unp_v_0p_2(q_sq, xb, t, ep, y, k)
    i_c_0p_a_2 = i_c_unp_a_0p_2(q_sq, xb, t, ep, y, tprime, k)

    return (i_c_pp_2*i_curly_c + i_c_pp_v_2*i_curly_c_v + i_c_pp_a_2*i_curly_c_a + i_c_0p_2*i_curly_c_eff + i_c_0p_v_2*i_curly_c_eff_v + i_c_0p_a_2*i_curly_c_eff_a)

def i_unp_c3(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_3 = i_c_unp_pp_3(q_sq, xb, t, ep, y, k)
    i_c_pp_v_3 = i_c_unp_v_pp_3(q_sq, xb, t, ep, y, k)
    i_c_pp_a_3 = i_c_unp_a_pp_3(q_sq, xb, t, ep, y, tprime, k)

    i_c_0p_3 = 0.
    i_c_0p_v_3 = 0.
    i_c_0p_a_3 = 0.

    return (i_c_pp_3*i_curly_c + i_c_pp_v_3*i_curly_c_v + i_c_pp_a_3*i_curly_c_a + i_c_0p_3*i_curly_c_eff + i_c_0p_v_3*i_curly_c_eff_v + i_c_0p_a_3*i_curly_c_eff_a)

def i_unp_s1(
    lepton_helicity: float, q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_im_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_s_pp_1 = i_s_unp_pp_1(lepton_helicity, q_sq, xb, ep, y, tprime, k)
    i_s_pp_v_1 = i_s_unp_v_pp_1(lepton_helicity, q_sq, xb, t, ep, y, k)
    i_s_pp_a_1 = i_s_unp_a_pp_1(lepton_helicity, q_sq, xb, t, ep, y, tprime, k)

    i_s_0p_1 = i_s_unp_0p_1(lepton_helicity, q_sq, ep, y, ktilde)
    i_s_0p_v_1 = i_s_unp_v_0p_1(lepton_helicity, q_sq, xb, t, ep, y)
    i_s_0p_a_1 = i_s_unp_a_0p_1(lepton_helicity, q_sq, xb, t, ep, y, k)

    return (i_s_pp_1*i_curly_c + i_s_pp_v_1*i_curly_c_v + i_s_pp_a_1*i_curly_c_a + i_s_0p_1*i_curly_c_eff + i_s_0p_v_1*i_curly_c_eff_v + i_s_0p_a_1*i_curly_c_eff_a)

def i_unp_s2(
    lepton_helicity: float, q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_im_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_s_pp_2 = i_s_unp_pp_2(lepton_helicity, q_sq, xb, ep, y, tprime)
    i_s_pp_v_2 = i_s_unp_v_pp_2(lepton_helicity, q_sq, xb, t, ep, y)
    i_s_pp_a_2 = i_s_unp_a_pp_2(lepton_helicity, q_sq, xb, t, ep, y, tprime)

    i_s_0p_2 = i_s_unp_0p_2(lepton_helicity, q_sq, xb, t, ep, y, k)
    i_s_0p_v_2 = i_s_unp_v_0p_2(lepton_helicity, q_sq, xb, t, ep, y, k)
    i_s_0p_a_2 = i_s_unp_a_0p_2(lepton_helicity, q_sq, xb, t, ep, y, k)

    return (i_s_pp_2*i_curly_c + i_s_pp_v_2*i_curly_c_v + i_s_pp_a_2*i_curly_c_a + i_s_0p_2*i_curly_c_eff + i_s_0p_v_2*i_curly_c_eff_v + i_s_0p_a_2*i_curly_c_eff_a)

def interference_amplitude(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, use_ww: bool = True):

    if target_polar == 1.0:
        # [TODO]: Code polarized target coefficients. @Woofmagic
        raise NotImplementedError("NO POLARIZED TARGET YET!")
    
    if lep_helicity == 0.0:
        i_c0 = 0.5 * (
            i_unp_c0(q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww) +
            i_unp_c0(q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
        )

        i_c1 = 0.5 * (
            i_unp_c1(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww) +
            i_unp_c1(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
        )

        i_c2 = 0.5 * (
            i_unp_c2(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww) +
            i_unp_c2(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
        )

        i_c3 = 0.5 * (
            i_unp_c3(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww) +
            i_unp_c3(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
        )

        i_s1 = 0.5 * (
            i_unp_s1(1.0, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww) + 
            i_unp_s1(-1.0, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
        )

        i_s2 = 0.5 * (
            i_unp_s2(1.0, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww) +
            i_unp_s2(-1.0, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
        )
    else:
        i_c0 = i_unp_c0(q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
        i_c1 = i_unp_c1(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
        i_c2 = i_unp_c2(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
        i_c3 = i_unp_c3(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)

        i_s1 = i_unp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
        i_s2 = i_unp_s2(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)

    return ((i_c0 + 
            i_c1 * tf.cos(1. * (tf.constant(np.pi) - phi)) + 
            i_c2 * tf.cos(2. * (tf.constant(np.pi) - phi)) + 
            i_c3 * tf.cos(3. * (tf.constant(np.pi) - phi)) + 
            i_s1 * tf.sin(1. * (tf.constant(np.pi) - phi)) + 
            i_s2 * tf.sin(2. * (tf.constant(np.pi) - phi)))/(xb * y * y * y * t * p1 * p2))
    
def bkm10_cross_section(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    if target_polar == 1.0:
        # [TODO]: Code polarized target coefficients. @Woofmagic
        raise NotImplementedError("NO POLARIZED TARGET YET!")
    
    bh_km15_plus_beam = bh_squared(
        +1.0, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)

    bh_km15_minus_beam = bh_squared(
        -1.0, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)

    dvcs_km15_plus_beam = dvcs_squared(
        +1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    
    dvcs_km15_minus_beam = dvcs_squared(
        -1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    interference_km15_plus_beam = interference_amplitude(
        +1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    
    interference_km15_minus_beam = interference_amplitude(
        -1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    
    tf_cross_section_km15 = 0.0
    
    if lep_helicity == 0.0:
        tf_cross_section_km15 = 0.5 * (
            _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*(
                bh_km15_plus_beam + bh_km15_minus_beam +
                dvcs_km15_plus_beam + dvcs_km15_minus_beam +
                interference_km15_plus_beam + interference_km15_minus_beam) / (16. * tf.square(tf.constant(np.pi)) * q_sq * tf.sqrt(1. + ep**2)))
        
    elif lep_helicity == 1.0:
        tf_cross_section_km15 = (
            _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*(
                bh_km15_plus_beam + 0.0 +
                dvcs_km15_plus_beam + 0.0 +
                interference_km15_plus_beam + 0.0) / (16. * tf.square(tf.constant(np.pi)) * q_sq * tf.sqrt(1. + ep**2)))
        
    elif lep_helicity == -1.0:
        tf_cross_section_km15 = (
            _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*(
                0.0 + bh_km15_minus_beam +
                0.0 + dvcs_km15_minus_beam +
                0.0 + interference_km15_minus_beam) / (16. * tf.square(tf.constant(np.pi)) * q_sq * tf.sqrt(1. + ep**2)))
        
    return tf_cross_section_km15

def bkm10_bsa(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    if target_polar == 1.0:
        # [TODO]: Code polarized target coefficients. @Woofmagic
        raise NotImplementedError("[ERROR]: NO POLARIZED TARGET YET!")
    
    bh_km15_plus_beam = bh_squared(
        +1.0, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)

    bh_km15_minus_beam = bh_squared(
        -1.0, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)

    dvcs_km15_plus_beam = dvcs_squared(
        +1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    
    dvcs_km15_minus_beam = dvcs_squared(
        -1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    interference_km15_plus_beam = interference_amplitude(
        +1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    
    interference_km15_minus_beam = interference_amplitude(
        -1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    
    cross_section_plus_beam = (_CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*(
                bh_km15_plus_beam + dvcs_km15_plus_beam + interference_km15_plus_beam) / (16. * tf.square(tf.constant(np.pi)) * q_sq * tf.sqrt(1. + ep**2))
                )
    
    cross_section_minus_beam = (_CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*(
                bh_km15_minus_beam + dvcs_km15_minus_beam + interference_km15_minus_beam) / (16. * tf.square(tf.constant(np.pi)) * q_sq * tf.sqrt(1. + ep**2))
                )

    tf_bsa = ((cross_section_plus_beam - cross_section_minus_beam) / (cross_section_plus_beam + cross_section_minus_beam))
        
    return tf_bsa

class SimultaneousObservablesLoss(tf.keras.losses.Loss):
    def __init__(self, name = "simultaneous_observables_loss"):
        super().__init__(name = name)

        self._OBSERVABLE_WEIGHT_1 = 0.5
        self._OBSERVABLE_WEIGHT_2 = 0.5
        
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
            TEST_LEPTON_HELICITY, TEST_TARGET_POLARIZATION,
            q_squared_tf, xb_tf, t_tf, epsilon_tf, y_lep_tf, xi_tf, k_tf, f1_tf, f2_tf, ktilde_tf, tprime_tf, phi_values, p1_tf, p2_tf,
            cff_h_real_tf, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag_tf, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)
        
        # compute BSA residuals:
        residuals_bsa = true_bsa - predicted_bsa

        # compute the MSE:
        mean_squared_error = (
            self._OBSERVABLE_WEIGHT_1 * tf.reduce_mean(tf.square(residuals_cross_section)) + 
            self._OBSERVABLE_WEIGHT_2 * tf.reduce_mean(tf.square(residuals_bsa)))

        return mean_squared_error
    
def cff_h_model():
    initializer = tf.keras.initializers.RandomUniform(
        minval = _INITIALIZER_MINIMUM_VALUE, maxval = _INITIALIZER_MAXMIMUM_VALUE, seed = None)
    
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
        loss = SimultaneousObservablesLoss(),
        metrics = ["accuracy"])
    return model

dnn_model = cff_h_model()

dnn_model_history = dnn_model.fit(
    x_training,
    y_training,
    validation_data = (x_validation, y_validation),
    epochs = _NUMBER_OF_EPOCHS,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 25, # stop if no improvement for 25 epochs
            restore_best_weights = True
        )
    ],
    batch_size = _BATCH_SIZE,
    verbose = 0
    )

number_of_epochs_run = len(dnn_model_history.epoch)
print(f"[NOTE]: The model ran for {number_of_epochs_run} epochs before early stopping.")

dnn_model.save(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/replicas/replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.keras")

training_loss_data = dnn_model_history.history["loss"]
validation_loss_data = dnn_model_history.history["val_loss"]

loss, accuracy = dnn_model.evaluate(x_testing, y_testing, verbose = 1)
print(f"[NOTE]: Test Loss for Replica {replica_number}: {loss}")
print(f"[NOTE]: Test Accuracy for Replica {replica_number}: {accuracy}")

curves_fig, curves_ax = plt.subplots(1, figsize = (8, 8))
log_curves_fig, log_curves_ax = plt.subplots(1, figsize = (8, 8))

curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.array([np.max(training_loss_data) for number in training_loss_data]), color = "red", label = "Initial Loss Value")
curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.zeros(shape = (number_of_epochs_run)), color = "green", label = r"Loss $= 0$")
curves_ax.plot(np.arange(0, number_of_epochs_run, 1), training_loss_data, color = "blue", label = "Training Loss")
curves_ax.plot(np.arange(0, number_of_epochs_run, 1), validation_loss_data, color = "purple", label = "Validation Loss")

log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(np.array([np.max(training_loss_data) for number in training_loss_data])), color = "red", label = "Initial Loss Value")
log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(np.zeros(shape = (number_of_epochs_run)) + 1e-20), color = "green", label = r"Loss $= 0$")
log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(training_loss_data), color = "blue", label = "Log Training Loss")
log_curves_ax.plot(np.arange(0, number_of_epochs_run, 1), np.log(validation_loss_data), color = "purple", label = "Log Validation Loss")

curves_ax.legend(fontsize = 15)
log_curves_ax.legend(fontsize = 15)

curves_ax.set_xlabel("Epoch", fontsize = 15)
curves_ax.set_ylabel("MSE", fontsize = 15)
curves_ax.set_title(f"Replica {replica_number} Learning Curves\n(Eval. Loss $= {loss:.3f}$, Eval. Accuracy $= {accuracy:.3f})$", fontsize = 15)

log_curves_ax.set_xlabel("Epoch", fontsize = 15)
log_curves_ax.set_ylabel("Log MSE Loss", fontsize = 15)
log_curves_ax.set_title(f"Replica {replica_number} Learning Curves\n(Eval. Loss $= {loss:.3f}$, Eval. Accuracy $= {accuracy:.3f})$", fontsize = 15)

curves_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/learning_curves/lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png")
curves_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/learning_curves/lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps")

log_curves_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/learning_curves/log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png")
log_curves_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/learning_curves/log_lc_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps")

plt.close(curves_fig)
plt.close(log_curves_fig)

tf.keras.backend.clear_session()

print(f"[INFO]: End of script reached!")
