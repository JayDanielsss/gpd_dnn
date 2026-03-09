##########################################
# FILE INFORMATION:
# Purpose: generates a .csv file containing
# pseudodata for training DNN models down
# the line.
# Created: 20260107
# Last changed: 20260309
##########################################

print(f"[INFO]: Script began running!")

##########################################
# Importing Python Libraries
##########################################

import os

import pandas as pd
import matplotlib.pyplot as plt
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

STARTING_PHI_VALUE_IN_DEGREES = 0
ENDING_PHI_VALUE_IN_DEGREES = 360
NUMBER_OF_PHI_POINTS = 360 # this number is important!

TEST_LEPTON_HELICITY = 0.0
TEST_TARGET_POLARIZATION = 0.0

print(f"[INFO]: Detected lepton helicity to be: {'unpolarized' if TEST_LEPTON_HELICITY == 0.0 else 'polarized'}")

##########################################
# Making required directories
##########################################

os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/data", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/learning_curves", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/replicas", exist_ok = True)

##########################################
# Initializing the Phi Array
##########################################

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = np.array([np.radians(degree_value) for degree_value in phi_array_in_degrees])

##########################################
# Defining the Kinematic Point
##########################################

TEST_BEAM_ENERGY = 5.75
TEST_QSQ = 1.82
TEST_XB = 0.34
TEST_T = -.17

# these values come later *and* are derived from the values above:
# see [this Mathematica notebook](https://github.com/Woofmagic/BKM10_Spin_Polarized/blob/main/mathematica/bkm10_test.nb)
TEST_EP = 0.47293561004973345
TEST_Y = 0.49609612355928445
TEST_XI = 0.19906188837146524
TEST_T_MIN = -0.13551824472915253
TEST_T_PRIME = -0.034481755270847486
TEST_K_TILDE = 0.1592415651944438
TEST_K = 0.08492693191323883
TEST_FE = 0.6511853266961825
TEST_FG = 1.8186612105254523
TEST_F2 = 1.1137103937669304
TEST_F1 = 0.7049508167585219
TEST_KDD = (
    (-1.*TEST_QSQ / (2.*TEST_Y*(1.+TEST_EP**2))) * 
    (1. + ((2.*TEST_K*np.cos(np.pi - phi_array_in_radians)) - ((TEST_T / TEST_QSQ)*(1.-(TEST_XB * (2. - TEST_Y)) + (TEST_Y * TEST_EP**2 / 2.))) + (TEST_Y * TEST_EP**2 / 2.))))

TEST_P1 = (1. + (2. * (TEST_KDD / TEST_QSQ)))
TEST_P2 = ((-2. * (TEST_KDD / TEST_QSQ)) + (TEST_T / TEST_QSQ))

print(f"[INFO]: We have constructed a Python list of length {len(phi_array_in_radians)} of azimuthal angles from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees.")

##########################################
# Defining the Kinematic Point
##########################################

test_datapoints = [g.DataPoint(
    xB = TEST_XB, t = TEST_T, Q2 = TEST_QSQ, phi = fixed_phi,
    process = "ep2epgamma", exptype = 'fixed target',
    in1energy = TEST_BEAM_ENERGY, in1charge = -1, in1polarization = +1,
    observable = 'XS',
    fname = 'Trento') for fixed_phi in phi_array_in_radians]

assert len(test_datapoints) == len(phi_array_in_radians), "[ASSERT]: Datapoint array has wrong size."

##########################################
# Compton Form Factor Values
##########################################

# cff H
real_h_values = [th_KM15.ReH(datapoint) for datapoint in test_datapoints]
imag_h_values = [th_KM15.ImH(datapoint) for datapoint in test_datapoints]

# cff E
real_e_values = [th_KM15.ReE(datapoint) for datapoint in test_datapoints]
imag_e_values = [th_KM15.ImE(datapoint) for datapoint in test_datapoints]

# cff Ht
real_ht_values = [th_KM15.ReHt(datapoint) for datapoint in test_datapoints]
imag_ht_values = [th_KM15.ImHt(datapoint) for datapoint in test_datapoints]

# cff Et
real_et_values = [th_KM15.ReEt(datapoint) for datapoint in test_datapoints]
imag_et_values = [th_KM15.ImEt(datapoint) for datapoint in test_datapoints]

assert len(real_h_values) == len(test_datapoints), "[ASSERT]: Re[H] array has wrong size."
assert len(imag_h_values) == len(test_datapoints), "[ASSERT]: Im[H] array has wrong size."

# we slice at [0] because, as we verified earlier, these arrays are all the same value; equivalence class defined on them

print(f"[INFO]: Now setting Re[H] = {real_h_values[0]}")
print(f"[INFO]: Now setting Im[H] = {imag_h_values[0]}")

print(f"[INFO]: Now setting Re[H] = {real_e_values[0]}")
print(f"[INFO]: Now setting Im[E] = {imag_e_values[0]}")

print(f"[INFO]: Now setting Re[Ht] = {real_ht_values[0]}")
print(f"[INFO]: Now setting Im[Ht] = {imag_ht_values[0]}")

print(f"[INFO]: Now setting Re[Et] = {real_et_values[0]}")
print(f"[INFO]: Now setting Im[Et] = {imag_et_values[0]}")

CFF_REAL_H_KM15 = real_h_values[0]
CFF_IMAG_H_KM15 = imag_h_values[0]
CFF_REAL_E_KM15 = real_e_values[0]
CFF_IMAG_E_KM15 = imag_e_values[0]
CFF_REAL_HT_KM15 = real_ht_values[0]
CFF_IMAG_HT_KM15 = imag_ht_values[0]
CFF_REAL_ET_KM15 = real_et_values[0]
CFF_IMAG_ET_KM15 = imag_et_values[0]

CFF_H_KM15 = complex(CFF_REAL_H_KM15, CFF_IMAG_H_KM15)
CFF_H_TILDE_KM15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
CFF_E_KM15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
CFF_E_TILDE_KM15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

##########################################
# Obtaining Observable Values:
##########################################

km15_cross_section = DifferentialCrossSection(
    configuration = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k = TEST_BEAM_ENERGY,
            squared_Q_momentum_transfer = TEST_QSQ,
            x_Bjorken = TEST_XB,
            squared_hadronic_momentum_transfer_t = TEST_T),
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

##########################################
# TSA(lambda = 0.0)
##########################################

bkm10_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = 0.0).real

##########################################
# TSA(lambda = +1.0)
##########################################

bkm10_plus_beam_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = +1.0).real

##########################################
# TSA(lambda = -1.0)
##########################################

bkm10_minus_beam_tsa_km15 = km15_cross_section.compute_tsa(
    phi_array_in_radians,
    lepton_polarization = -1.0).real

##########################################
# DSA
##########################################

bkm10_dsa_km15 = km15_cross_section.compute_dsa(
    phi_array_in_radians).real

##########################################
# Making the Training Data:
##########################################

test_dataframe = pd.DataFrame({
    "set": 1,
    "k": TEST_BEAM_ENERGY,
    "q_squared": TEST_QSQ,
    "x_b": TEST_XB,
    "t": TEST_T,
    "phi": phi_array_in_radians,
    ####### IMPORTANT! Using KM15 VALUES! #######
    "unp_beam_unp_target_xsec": bkm10_unp_beam_unp_target_km15,
    "plus_beam_unp_target_xsec": bkm10_plus_beam_unp_target_km15,
    "minus_beam_unp_target_xsec": bkm10_minus_beam_unp_target_km15,
    "unp_beam_lp_target_xsec": bkm10_unp_beam_lp_target_km15,
    "plus_beam_lp_target_xsec": bkm10_plus_beam_lp_target_km15,
    "minus_beam_lp_target_xsec": bkm10_minus_beam_lp_target_km15,
    "unp_beam_unp_target_xsec": bkm10_unp_beam_unp_target_km15,
    "unp_target_bsa": bkm10_bsa_km15,
    "plus_lp_target_bsa": bkm10_plus_lp_bsa_km15,
    "minus_lp_target_bsa": bkm10_minus_lp_bsa_km15,
    "unp_beam_tsa": bkm10_tsa_km15,
    "plus_beam_tsa": bkm10_plus_beam_tsa_km15,
    "minus_beam_tsa": bkm10_minus_beam_tsa_km15,
    "dsa": bkm10_dsa_km15,
    ####### IMPORTANT! Using KM15 VALUES! #######
    "Re[H]": real_h_values, # KM15 value for Re[H]
    "Im[H]": imag_h_values, # KM15 value for Im[H],
    "Re[E]": real_e_values,
    "Im[E]": imag_e_values,
    "Re[Ht]": real_ht_values,
    "Im[Ht]": imag_ht_values,
    "Re[Et]": real_et_values,
    "Im[Et]": imag_et_values,
})

test_dataframe.to_csv(
    path_or_buf = f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/data/main_pseudodata_file_v{MAJOR_MINOR_NUMBER}.csv"
)

print(f"[INFO]: End of script reached!")
