##########################################
# FILE INFORMATION:
# Purpose: generates a .csv file containing
# pseudodata for training DNN models down
# the line.
# Created: 20260107
# Last changed: 20260308
##########################################

print(f"[INFO]: Script began running!")

##########################################
# Importing Python Libraries
##########################################

import os
import sys

import pandas as pd
import numpy as np

import gepard as g
from gepard.fits import th_KM15

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

print(f"[INFO]: Libraries imported!")

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

##########################################
# Kinematic Grid:
##########################################

# kinematic boundaries
BEAM_K_LOWER = 5.0
BEAM_K_UPPER = 11.0
Q_SQUARED_LOWER = 1.0
Q_SQUARED_UPPER = 1.0
X_B_LOWER = 0.1
X_B_UPPER = 0.9
T_LOWER = -1.0
T_UPPER = -.1

# number of points for each variable
NUMBER_OF_BEAM_K = 6
NUMBER_OF_Q_SQUARED = 1
NUMBER_OF_X_B = 8
NUMBER_OF_T = 9

# iterable ranges:
K_RANGE = np.linspace(BEAM_K_LOWER, BEAM_K_UPPER, NUMBER_OF_BEAM_K)
Q2_RANGE = np.linspace(Q_SQUARED_LOWER, Q_SQUARED_UPPER, NUMBER_OF_Q_SQUARED)
X_B_RANGE = np.linspace(X_B_LOWER, X_B_UPPER, NUMBER_OF_X_B)
T_RANGE = np.linspace(T_LOWER, T_UPPER, NUMBER_OF_T)

NUMBER_OF_PHI_POINTS = 360
STARTING_PHI_VALUE_IN_DEGREES = 0
ENDING_PHI_VALUE_IN_DEGREES = 360

##########################################
# Initializing the Phi Array
##########################################

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = np.array([np.radians(degree_value) for degree_value in phi_array_in_degrees])
print(f"We have constructed a Python list of length {len(phi_array_in_radians)} of azimuthal angles from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees.")

##########################################
# Tons of TensorFlow Functions
##########################################

_MASS_OF_PROTON_IN_GEV = 0.93827208816
_ELECTRIC_FORM_FACTOR_CONSTANT = 0.710649
_PROTON_MAGNETIC_MOMENT = 2.79284734463

def compute_fe(t):
    return 1./ (1. - (t / _ELECTRIC_FORM_FACTOR_CONSTANT))**2

def compute_fg(fe):
    return _PROTON_MAGNETIC_MOMENT * fe

def compute_f2(t, fe, fg):
    tau = -1. * t/ (4. * _MASS_OF_PROTON_IN_GEV**2)
    numerator = fg - fe
    denominator = 1. + tau
    return numerator / denominator

def compute_f1(fg, f2):
    return fg - f2

def compute_epsilon(xb, q_squared):
    return 2. * xb * _MASS_OF_PROTON_IN_GEV / np.sqrt(q_squared)

def compute_y(k_beam, q_squared, ep):
    return np.sqrt(q_squared) / (ep * k_beam)

def compute_skewness(xb, t, q_squared):
    return xb * (1. + (t / (2. * q_squared))) / (2. - xb + (xb * t / q_squared))

def compute_t_min(xb, q_squared, ep):
    return -1. * q_squared * ((2. * (1. - xb) * (1. - np.sqrt(1. + ep**2))) + ep**2) / ((4. * xb * (1. - xb)) + ep**2)

def compute_t_prime(t, tmin):
    return (t-tmin)

def compute_k_tilde(xb, q_squared, t, tmin, ep):
    return np.sqrt(tmin - t) * np.sqrt(((1. - xb) * np.sqrt((1. + ep**2))) + (((tmin - t) * (ep**2 + (4. * (1. - xb) * xb))) / (4. * q_squared)))

def compute_k(q_squared, y_lep, ep, k_tilde):
    return np.sqrt(((1. - y_lep + (ep**2 * y_lep**2 / 4.)) / q_squared)) * k_tilde

##########################################
# THE MAIN LOOP!
##########################################

rows = []
total_kinematic_settings = NUMBER_OF_BEAM_K * NUMBER_OF_Q_SQUARED * NUMBER_OF_X_B * NUMBER_OF_T
total_settings = NUMBER_OF_BEAM_K * NUMBER_OF_Q_SQUARED * NUMBER_OF_X_B * NUMBER_OF_T * NUMBER_OF_PHI_POINTS

print(f"[INFO]: Ready to iterate over {total_kinematic_settings} kinematic settings.")
print(f"[INFO]: A total of {total_settings} combinations points.")

# Define your ranges (ensure these are lists or arrays)
lengths = [len(K_RANGE), len(Q2_RANGE), len(X_B_RANGE), len(T_RANGE)]

kinematic_set_number = int(sys.argv[1])

idx_t = kinematic_set_number % lengths[3]
idx_x = (kinematic_set_number // lengths[3]) % lengths[2]
idx_q2 = (kinematic_set_number // (lengths[3] * lengths[2])) % lengths[1]
idx_k = (kinematic_set_number // (lengths[3] * lengths[2] * lengths[1])) % lengths[0]

fixed_k  = K_RANGE[idx_k]
fixed_q_squared = Q2_RANGE[idx_q2]
fixed_x_bjorken = X_B_RANGE[idx_x]
fixed_t = T_RANGE[idx_t]

print(f"[INFO]: Task {kinematic_set_number} corresponds to: k = {fixed_k}, q2 = {fixed_q_squared}, x = {fixed_x_bjorken}, t = {fixed_t}\n")
    
# using gepard's DataPoint with phi varying:
current_kinematic_setting = [g.DataPoint(
    xB = fixed_x_bjorken, t = fixed_t, Q2 = fixed_q_squared, phi = fixed_phi,
    process = "ep2epgamma", exptype = 'fixed target',
    in1energy = fixed_k, in1charge = -1, in1polarization = +1,
    observable = 'XS',
    fname = 'Trento') for fixed_phi in phi_array_in_radians]

assert len(current_kinematic_setting) == len(phi_array_in_radians), "[ASSERT]: Datapoint array has wrong size."

##########################################
# Predicting CFFs at datapoint:
##########################################
# cff H
real_h_values = [th_KM15.ReH(datapoint) for datapoint in current_kinematic_setting]
imag_h_values = [th_KM15.ImH(datapoint) for datapoint in current_kinematic_setting]

# cff E
real_e_values = [th_KM15.ReE(datapoint) for datapoint in current_kinematic_setting]
imag_e_values = [th_KM15.ImE(datapoint) for datapoint in current_kinematic_setting]

# cff Ht
real_ht_values = [th_KM15.ReHt(datapoint) for datapoint in current_kinematic_setting]
imag_ht_values = [th_KM15.ImHt(datapoint) for datapoint in current_kinematic_setting]

# cff Et
real_et_values = [th_KM15.ReEt(datapoint) for datapoint in current_kinematic_setting]
imag_et_values = [th_KM15.ImEt(datapoint) for datapoint in current_kinematic_setting]

assert len(real_h_values) == len(current_kinematic_setting), "[ASSERT]: Re[H] array has wrong size."
assert len(imag_h_values) == len(current_kinematic_setting), "[ASSERT]: Im[H] array has wrong size."

# we slice at [0] because, as we verified earlier, these arrays are all the same value; equivalence class defined on them
CFF_H_KM15 = complex(real_h_values[0], imag_h_values[0])
CFF_H_TILDE_KM15 = complex(real_ht_values[0], imag_ht_values[0])
CFF_E_KM15 = complex(real_e_values[0], imag_e_values[0])
CFF_E_TILDE_KM15 = complex(real_et_values[0], imag_et_values[0])

##########################################
# Obtaining Observable Values:
##########################################

km15_cross_section = DifferentialCrossSection(
    configuration = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k = fixed_k,
            squared_Q_momentum_transfer = fixed_q_squared,
            x_Bjorken = fixed_x_bjorken,
            squared_hadronic_momentum_transfer_t = fixed_t),
        "cff_inputs": CFFInputs(
            compton_form_factor_h = CFF_H_KM15,
            compton_form_factor_h_tilde = CFF_H_TILDE_KM15,
            compton_form_factor_e = CFF_E_KM15,
            compton_form_factor_e_tilde = CFF_E_TILDE_KM15),
        "using_ww": True
    },
    verbose = False,
    debugging = False)

##########################################
# d^{4}\sigma(lambda = 0, Lambda = 0)
##########################################

bkm10_unp_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = 0.0,
    target_polarization = 0.0).real

##########################################
# d^{4}\sigma(lambda = +1, Lambda = 0)
##########################################
    
bkm10_plus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = +1.0,
    target_polarization = 0.0).real
    
##########################################
# d^{4}\sigma(lambda = -1, Lambda = 0)
##########################################

bkm10_minus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = -1.0,
    target_polarization = 0.0).real
    
##########################################
# d^{4}\sigma(lambda = 0, Lambda = 1/2)
##########################################

bkm10_unp_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = 0.0,
    target_polarization = +0.5).real
    
##########################################
# d^{4}\sigma(lambda = +1, Lambda = 1/2)
##########################################
    
bkm10_plus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = +1.0,
    target_polarization = +0.5).real
    
##########################################
# d^{4}\sigma(lambda = -1, Lambda = 1/2)
##########################################

bkm10_minus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
    phi_array_in_radians,
    lepton_helicity = -1.0,
    target_polarization = +0.5).real

##########################################
# BSA(Lambda = 0)
##########################################
    
bkm10_bsa_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians, 
    target_polarization = 0.0).real

##########################################
# BSA(Lambda = 1/2)
##########################################
    
bkm10_plus_lp_bsa_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = +0.5).real

##########################################
# BSA(Lambda = -1/2)
##########################################
    
bkm10_minus_lp_bsa_km15 = km15_cross_section.compute_bsa(
    phi_array_in_radians,
    target_polarization = -0.5).real

# ##########################################
# # TSA(lambda = 0.0)
# ##########################################

bkm10_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = 0.0).real

# ##########################################
# # TSA(lambda = +1.0)
# ##########################################

bkm10_plus_beam_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = +1.0).real

# ##########################################
# # TSA(lambda = -1.0)
# ##########################################

bkm10_minus_beam_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = -1.0).real

# ##########################################
# # DSA
# ##########################################

bkm10_dsa_km15 = km15_cross_section.compute_dsa(
    phi_array_in_radians).real

##########################################
# Making required directories
##########################################

os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/learning_curves", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas", exist_ok = True)

this_kinematic_set_dataframe = []

for phi_index, phi_value in enumerate(phi_array_in_radians):
    new_entry = {
        "set": kinematic_set_number,
        "k": fixed_k,
        "q_squared": fixed_q_squared,
        "x_b": fixed_x_bjorken,
        "t": fixed_t,
        "phi": phi_value,
        ####### IMPORTANT! Using KM15 VALUES! #######
        "unp_beam_unp_target_xsec": bkm10_unp_beam_unp_target_km15[phi_index],
        "plus_beam_unp_target_xsec": bkm10_plus_beam_unp_target_km15[phi_index],
        "plus_minus_unp_target_xsec": bkm10_minus_beam_unp_target_km15[phi_index],
        "unp_beam_unp_lp_xsec": bkm10_unp_beam_lp_target_km15[phi_index],
        "plus_beam_unp_lp_xsec": bkm10_plus_beam_lp_target_km15[phi_index],
        "plus_minus_unp_lp_xsec": bkm10_minus_beam_lp_target_km15[phi_index],
        "unp_target_bsa": bkm10_bsa_km15[phi_index],
        "plus_target_bsa": bkm10_plus_lp_bsa_km15[phi_index],
        "minus_target_bsa": bkm10_minus_lp_bsa_km15[phi_index],
        "unp_beam_tsa": bkm10_tsa_km15[phi_index],
        "plus_beam_tsa": bkm10_plus_beam_tsa_km15[phi_index],
        "minus_beam_tsa": bkm10_minus_beam_tsa_km15[phi_index],
        "dsa": bkm10_dsa_km15[phi_index],
        ####### IMPORTANT! Using KM15 VALUES! #######
        "Re[H]": real_h_values[0], # KM15 value for Re[H]
        "Im[H]": imag_h_values[0], # KM15 value for Im[H]
        "Re[E]": real_e_values[0],
        "Im[E]": imag_e_values[0],
        "Re[Ht]": real_ht_values[0],
        "Im[Ht]": imag_ht_values[0],
        "Re[Et]": real_et_values[0],
        "Im[Et]": imag_et_values[0]
    }

    rows.append(new_entry)
    this_kinematic_set_dataframe.append(new_entry)

pd.DataFrame(this_kinematic_set_dataframe).to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/main_pseudodata_file_set_{kinematic_set_number}_v{MAJOR_MINOR_NUMBER}.csv"
)

# clean up:
del this_kinematic_set_dataframe
del km15_cross_section
del current_kinematic_setting

with open(
    file = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/log_set_{kinematic_set_number}_v{MAJOR_MINOR_NUMBER}.txt",
    mode = "w",
    buffering = 1) as logfile:

    logfile.write(f"Kinematic Set Number: {kinematic_set_number}\n")
    logfile.write(f"k = {fixed_k}\n")
    logfile.write(f"Q^2 = {fixed_q_squared}\n")
    logfile.write(f"xb = {fixed_x_bjorken}\n")
    logfile.write(f"t = {fixed_t}\n")

    fixed_ep = compute_epsilon(fixed_x_bjorken, fixed_q_squared)
    fixed_y = compute_y(fixed_k, fixed_q_squared, fixed_ep)
    fixed_xi = compute_skewness(fixed_x_bjorken, fixed_t, fixed_q_squared)
    fixed_t_min = compute_t_min(fixed_x_bjorken, fixed_q_squared, fixed_ep)
    fixed_k_tilde = compute_k_tilde(fixed_x_bjorken, fixed_q_squared, fixed_t, fixed_t_min, fixed_ep)
    fixed_big_k = compute_k(fixed_q_squared, fixed_y, fixed_ep, fixed_k_tilde)
    fixed_t_prime = compute_t_prime(fixed_t, fixed_t_min)
    fixed_fe = compute_fe(fixed_t)
    fixed_fg = compute_fg(fixed_fe)
    fixed_f2 = compute_f2(fixed_t, fixed_fe, fixed_fg)
    fixed_f1 = compute_f1(fixed_fg, fixed_f2)

    logfile.write(f"ep: {fixed_ep}\n")
    logfile.write(f"y = {fixed_y}\n")
    logfile.write(f"xi = {fixed_xi}\n")
    logfile.write(f"tmin = {fixed_t_min}\n")
    logfile.write(f"Ktilde = {fixed_k_tilde}\n")
    logfile.write(f"K: {fixed_big_k}\n")
    logfile.write(f"tprime = {fixed_t_prime}\n")
    logfile.write(f"FE = {fixed_fe}\n")
    logfile.write(f"FG = {fixed_fg}\n")
    logfile.write(f"F2 = {fixed_f2}\n")
    logfile.write(f"F1 = {fixed_f1}\n")

    logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data\n")
    logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots\n")
    logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas\n")
    logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/learning_curves\n")

    logfile.write(f"Total number of proposed (initial) grid points: {total_kinematic_settings}\n")
    logfile.write(f"Each point has {NUMBER_OF_PHI_POINTS} phi values, which makes the grid's total size {total_settings}\n")
    logfile.close()
                
print(f"[INFO]: Has iteration successfully executed: {kinematic_set_number == total_kinematic_settings}")
print(f"[INFO]: New dataframe has {len(rows)} total rows.")
print(f"[INFO]: Attempted {total_settings} total kinematic settings.")
print(f"[INFO]: Angles per setting: {NUMBER_OF_PHI_POINTS}")

print(f"[INFO]: End of script reached!")
