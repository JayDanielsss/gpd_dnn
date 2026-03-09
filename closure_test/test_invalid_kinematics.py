
import sys
import numpy as np
import gepard as g
from gepard.fits import th_KM15
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

print(f"[INFO]: Libraries imported!")

# kinematic boundaries
BEAM_K_LOWER = 5.5
BEAM_K_UPPER = 11.0
Q_SQUARED_LOWER = 1.0
Q_SQUARED_UPPER = 5.0
X_B_LOWER = 0.1
X_B_UPPER = 0.6
T_LOWER = -1.0
T_UPPER = -.1

# number of points for each variable
NUMBER_OF_BEAM_K = 6
NUMBER_OF_Q_SQUARED = 10
NUMBER_OF_X_B = 10
NUMBER_OF_T = 6

# iterable ranges:
K_RANGE = np.linspace(BEAM_K_LOWER, BEAM_K_UPPER, NUMBER_OF_BEAM_K)
Q2_RANGE = np.linspace(Q_SQUARED_LOWER, Q_SQUARED_UPPER, NUMBER_OF_Q_SQUARED)
X_B_RANGE = np.linspace(X_B_LOWER, X_B_UPPER, NUMBER_OF_X_B)
T_RANGE = np.linspace(T_LOWER, T_UPPER, NUMBER_OF_T)

NUMBER_OF_PHI_POINTS = 360
STARTING_PHI_VALUE_IN_DEGREES = 0
ENDING_PHI_VALUE_IN_DEGREES = 360

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = np.array([np.radians(degree_value) for degree_value in phi_array_in_degrees])
print(f"We have constructed a Python list of length {len(phi_array_in_radians)} of azimuthal angles from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees.")

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

bkm10_plus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
        phi_array_in_radians,
        lepton_helicity = +1.0,
        target_polarization = 0.0).real

bkm10_minus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
        phi_array_in_radians,
        lepton_helicity = -1.0,
        target_polarization = 0.0).real

bkm10_unp_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
        phi_array_in_radians,
        lepton_helicity = 0.0,
        target_polarization = +0.5).real

bkm10_plus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
        phi_array_in_radians,
        lepton_helicity = +1.0,
        target_polarization = +0.5).real

bkm10_minus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
        phi_array_in_radians,
        lepton_helicity = -1.0,
        target_polarization = +0.5).real

array_of_cross_sections = [
        bkm10_unp_beam_unp_target_km15,
        bkm10_plus_beam_unp_target_km15,
        bkm10_minus_beam_unp_target_km15,
        bkm10_unp_beam_lp_target_km15,
        bkm10_plus_beam_lp_target_km15,
        bkm10_minus_beam_lp_target_km15,
    ]

if np.any(np.isnan(array_of_cross_sections)):
    print('[WARNING]: NaNs detected!')

# note that NaNs are *not finite*, i.e. np.isfinite(np.nan) is False
if not np.all(np.isfinite(array_of_cross_sections)):
    print('[WARNING]: plus or minus infinity detected!')

if np.any(np.concatenate(array_of_cross_sections).flatten() < 0.):
    print('[WARNING]: Negative cross-sections detected!')

print("BREAK BREAK BREAK")

test_array_of_nan_cross_sections = [
    np.array([np.nan, 2.0, 3.0, 4.0, 5.0]),
    np.array([1.0, np.nan, 3.0, 4.0, 5.0]),
    np.array([1.0, 2.0, 3.0, 4.0, np.nan]),
    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
]

test_array_of_negative_cross_sections = [
    np.array([-1.0, 2.0, 3.0, 4.0, 5.0]),
    np.array([1.0, -2.0, 3.0, 4.0, 5.0]),
    np.array([1.0, 2.0, 3.0, 4.0, -5.0]),
    np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    np.array([-1.0, -2.0, -3.0, -4.0, -5.0]),
]

if np.any(np.isnan(test_array_of_nan_cross_sections)):
    print('[WARNING]: NaNs in NAN cross section array!')

if np.any(np.isnan(test_array_of_negative_cross_sections)):
    print('[WARNING]: NaNs in NEGATIVE cross section array!')

# note that NaNs are *not finite*, i.e. np.isfinite(np.nan) is False
if not np.all(np.isfinite(test_array_of_nan_cross_sections)):
    print('[WARNING]: plus or minus infinity in NAN cross section array!')

# note that NaNs are *not finite*, i.e. np.isfinite(np.nan) is False
if not np.all(np.isfinite(test_array_of_negative_cross_sections)):
    print('[WARNING]: plus or minus infinity in NEGATIVE cross section array!')

if np.any(np.concatenate(test_array_of_nan_cross_sections).flatten() < 0.):
    print('[WARNING]: Negative cross-sections detected in NAN cross section array!')

if np.any(np.concatenate(test_array_of_negative_cross_sections).flatten() < 0.):
    print('[WARNING]: Negative cross-sections detected in NEGATIVE cross section array!')