##########################################
# FILE INFORMATION:
# Purpose: generates a .csv file containing
# pseudodata for training DNN models down
# the line.
# Created: 20260107
# Last changed: 20260111
##########################################

print(f"[INFO]: Script began running!")

##########################################
# [IMPORTANT]: Static quantities parametrizing
# the program. Change these if you need!
##########################################

# verify this is what you want
SCRATCH_PATH = 'placeholder'

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

STARTING_PHI_VALUE_IN_DEGREES = 0
ENDING_PHI_VALUE_IN_DEGREES = 360

# this number is important!
NUMBER_OF_PHI_POINTS = 360

TEST_LEPTON_HELICITY = 0.0
TEST_TARGET_POLARIZATION = 0.0

print(f"[INFO]: Detected lepton helicity to be: {'unpolarized' if TEST_LEPTON_HELICITY == 0.0 else 'polarized'}")

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
# Making required directories
##########################################

os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/data", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/learning_curves", exist_ok = True)
os.makedirs(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/replicas", exist_ok = True)

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
# Initializing the Phi Array
##########################################

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = [np.radians(degree_value) for degree_value in phi_array_in_degrees]

##########################################
# Defining the Kinematic Point
##########################################

TEST_BEAM_ENERGY = 5.75
TEST_Q_SQUARED = 1.82
TEST_X_BJORKEN = 0.34
TEST_T_VALUE = -.17

print(f"[INFO]: We have constructed a Python list of length {len(phi_array_in_radians)} of azimuthal angles from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees.")

##########################################
# Defining the Kinematic Point
##########################################

test_datapoints = [g.DataPoint(
    xB = TEST_X_BJORKEN, t = TEST_T_VALUE, Q2 = TEST_Q_SQUARED, phi = fixed_phi,
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

cross_section_km15 = DifferentialCrossSection(
    configuration = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k = TEST_BEAM_ENERGY,
            squared_Q_momentum_transfer = TEST_Q_SQUARED,
            x_Bjorken = TEST_X_BJORKEN,
            squared_hadronic_momentum_transfer_t = TEST_T_VALUE),
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

bkm10_cross_sections_km15 = cross_section_km15.compute_cross_section(phi_array_in_radians).real
bkm10_bsa_km15 = cross_section_km15.compute_bsa(phi_array_in_radians).real

##########################################
# Plotting Observable Values
##########################################

cross_section_fig, cross_section_axes = plt.subplots(1, figsize = (8, 7))
cross_section_axes.scatter(
    phi_array_in_radians, bkm10_cross_sections_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")
title_string = (
    rf"$Q^2 = {TEST_Q_SQUARED:.2f}$ GeV$^2$, "
    rf"$x_B = {TEST_X_BJORKEN:.2f}$, "
    rf"$t = {TEST_T_VALUE:.2f}$ ,"
    rf"$k = {TEST_BEAM_ENERGY:.2f}$ GeV"
)
km15_cff_string = (
    rf"$\mathcal{{H}} = {CFF_H_KM15:.3f}$, "
    rf"$\mathcal{{E}} = {CFF_E_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{H}}}} = {CFF_H_TILDE_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{E}}}} = {CFF_E_TILDE_KM15:.3f}$ "
)
cross_section_axes.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
cross_section_axes.set_ylabel(r"$d^{4}\sigma$", fontsize = 16)
cross_section_axes.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
cross_section_axes.legend(fontsize = 14.3)
cross_section_axes.grid(visible = True)
cross_section_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/xsec_vs_phi_v{MAJOR_MINOR_NUMBER}.png")
cross_section_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/xsec_vs_phi_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(cross_section_fig)

bsa_fig, bsa_axes = plt.subplots(1, figsize = (8, 7))
bsa_axes.scatter(
    phi_array_in_radians, bkm10_bsa_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")
title_string = (
    rf"$Q^2 = {TEST_Q_SQUARED:.2f}$ GeV$^2$, "
    rf"$x_B = {TEST_X_BJORKEN:.2f}$, "
    rf"$t = {TEST_T_VALUE:.2f}$ ,"
    rf"$k = {TEST_BEAM_ENERGY:.2f}$ GeV"
)
km15_cff_string = (
    rf"$\mathcal{{H}} = {CFF_H_KM15:.3f}$, "
    rf"$\mathcal{{E}} = {CFF_E_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{H}}}} = {CFF_H_TILDE_KM15:.3f}$, "
    rf"$\widetilde{{\mathcal{{E}}}} = {CFF_E_TILDE_KM15:.3f}$ "
)
bsa_axes.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
bsa_axes.set_ylabel(r"BSA", fontsize = 16)
bsa_axes.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
bsa_axes.legend(fontsize = 14)
bsa_axes.grid(visible = True)
bsa_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/bsa_vs_phi_v{MAJOR_MINOR_NUMBER}.png")
bsa_fig.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/bsa_vs_phi_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(bsa_fig)

##########################################
# Making the Training Data:
##########################################

test_dataframe = pd.DataFrame({
    "set": 1,
    "k": TEST_BEAM_ENERGY,
    "q_squared": TEST_Q_SQUARED,
    "x_b": TEST_X_BJORKEN,
    "t": TEST_T_VALUE,
    "phi": phi_array_in_radians,
    "cross_section": bkm10_cross_sections_km15, # IMPORTANT! We are using the KM15-predicted x-section here!
    "beam_spin_asymmetry": bkm10_bsa_km15,
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
