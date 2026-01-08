##########################################
# FILE INFORMATION:
# Purpose: averages over all the replicas
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

##########################################
# Importing Python Libraries
##########################################

import sys
import glob

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import norm

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

print(f"[INFO]: Libraries imported!")

##########################################
# Matplotlib Plotting Customizability
##########################################

plt.rcParams.update({
    "text.usetex": False, "font.family": "serif",
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

# this is the entire dataset! (i.e. training+validation)
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
CFF_H_TILDE_KM15 = complex(CFF_REAL_E_KM15, CFF_IMAG_E_KM15)
CFF_E_KM15 = complex(CFF_REAL_HT_KM15, CFF_IMAG_HT_KM15)
CFF_E_TILDE_KM15 = complex(CFF_REAL_ET_KM15, CFF_IMAG_ET_KM15)

print(f"[INFO]: Selected CFF H = {CFF_H_KM15}")
print(f"[INFO]: Selected CFF E = {CFF_E_KM15}")
print(f"[INFO]: Selected CFF Ht = {CFF_H_TILDE_KM15}")
print(f"[INFO]: Selected CFF Et = {CFF_E_TILDE_KM15}")

title_string = (
    rf"$Q^2 = {FIXED_BEAM_ENERGY:.2f}$ GeV$^2$, "
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

##########################################
# Initializing the Phi Array
##########################################

STARTING_PHI_VALUE_IN_DEGREES = test_dataframe['phi'].min()
ENDING_PHI_VALUE_IN_DEGREES = test_dataframe['phi'].max()
NUMBER_OF_PHI_POINTS = test_dataframe[test_dataframe['phi'].notna()]

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)

phi_array_in_radians = [np.radians(degree_value) for degree_value in phi_array_in_degrees]

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

##########################################
# Making individual replica predictions
##########################################

replica_paths = sorted(glob.glob(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/replicas/replica_*_v{MAJOR_MINOR_NUMBER}.keras"))
replicas = [tf.keras.models.load_model(
    path,
    compile = False,
    safe_mode = False) for path in replica_paths]

print(f"[INFO]: Loaded {len(replicas)} replica models.")

all_predictions = []

for replica in replicas:
    predicted_outputs = replica.predict(x_data) # predicting using x_data
    all_predictions.append(predicted_outputs)

all_predictions = np.array(all_predictions)

print(f"[INFO]: The total number of predictions is {len(all_predictions)}.")

assert len(all_predictions) == len(replicas), "[ASSERT]: Number of total predictions doesn't match expected number."

replicas_cross_predictions = []
replicas_bsa_predictions = []

for index, _ in enumerate(all_predictions):

    cff_h_real = all_predictions[index][:, 0] # getting Re[H]
    cff_h_imag = all_predictions[index][:, 1] # getting Im[H]
    t = all_predictions[index][:, 2] # getting t 
    xb = all_predictions[index][:, 3] # getting xb
    q_squared = all_predictions[index][:, 4] # getting Q^{2}
    phi = all_predictions[index][:, 5]

    fe = compute_fe(t)
    fg = compute_fg(fe) 
    f2 = compute_f2(t, fe, fg)
    f1 = compute_f1(fg, f2)
    
    epsilon = compute_epsilon(xb, q_squared)
    y_lep = compute_y(FIXED_BEAM_ENERGY, q_squared, epsilon)
    xi = compute_skewness(xb, t, q_squared)
    tmin = compute_t_min(xb, q_squared, epsilon)
    tprime = compute_t_prime(t, tmin) # used in interference only
    ktilde = compute_k_tilde(xb, q_squared, t, tmin, epsilon)
    k = compute_k(q_squared, y_lep, epsilon, ktilde)
    kdd = compute_k_dot_delta(q_squared, xb, t, phi, epsilon, y_lep, k)
    p1 = prop_1(q_squared, kdd)
    p2 = prop_2(q_squared, t, kdd)

    cross_section = bkm10_cross_section(
        TEST_LEPTON_HELICITY, TEST_TARGET_POLARIZATION,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    predicted_bsa = bkm10_bsa(
        TEST_LEPTON_HELICITY, TEST_TARGET_POLARIZATION,
        q_squared, xb, t, epsilon, y_lep, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_h_real, CFF_REAL_HT_KM15, CFF_REAL_E_KM15, CFF_REAL_ET_KM15, cff_h_imag, CFF_IMAG_HT_KM15, CFF_IMAG_E_KM15, CFF_IMAG_ET_KM15)

    replicas_cross_predictions.append(cross_section)
    replicas_bsa_predictions.append(predicted_bsa)

replicas_cross_predictions = np.array(replicas_cross_predictions)
replicas_bsa_predictions = np.array(replicas_bsa_predictions)

##########################################
# Making replica ensemble predictions for
# cross section
##########################################

mean_xs = np.mean(replicas_cross_predictions, axis = 0)
std_dev_xs = np.std(replicas_cross_predictions, axis = 0)

xs_mean = np.mean(replicas_cross_predictions, axis = 0)
xs_min = np.min(replicas_cross_predictions, axis = 0)
xs_max = np.max(replicas_cross_predictions, axis = 0)
xs_q1 = np.percentile(replicas_cross_predictions, 25, axis = 0)
xs_q3 = np.percentile(replicas_cross_predictions, 75, axis = 0)

xs_percentile_10 = np.percentile(replicas_cross_predictions, 10, axis = 0)
xs_percentile_20 = np.percentile(replicas_cross_predictions, 20, axis = 0)
xs_percentile_30 = np.percentile(replicas_cross_predictions, 30, axis = 0)
xs_percentile_40 = np.percentile(replicas_cross_predictions, 40, axis = 0)
xs_median = np.percentile(replicas_cross_predictions, 50, axis = 0)
xs_percentile_60 = np.percentile(replicas_cross_predictions, 60, axis = 0)
xs_percentile_70 = np.percentile(replicas_cross_predictions, 70, axis = 0)
xs_percentile_80 = np.percentile(replicas_cross_predictions, 80, axis = 0)
xs_percentile_90 = np.percentile(replicas_cross_predictions, 90, axis = 0)

mean_bsa = np.mean(replicas_bsa_predictions, axis = 0)
std_dev_bsa = np.std(replicas_bsa_predictions, axis = 0)

bsa_mean = np.mean(replicas_bsa_predictions, axis = 0)
bsa_min = np.min(replicas_bsa_predictions, axis = 0)
bsa_max = np.max(replicas_bsa_predictions, axis = 0)
bsa_q1 = np.percentile(replicas_bsa_predictions, 25, axis = 0)
bsa_q3 = np.percentile(replicas_bsa_predictions, 75, axis = 0)

bsa_percentile_10 = np.percentile(replicas_bsa_predictions, 10, axis = 0)
bsa_percentile_20 = np.percentile(replicas_bsa_predictions, 20, axis = 0)
bsa_percentile_30 = np.percentile(replicas_bsa_predictions, 30, axis = 0)
bsa_percentile_40 = np.percentile(replicas_bsa_predictions, 40, axis = 0)
bsa_median = np.percentile(replicas_bsa_predictions, 50, axis = 0)
bsa_percentile_60 = np.percentile(replicas_bsa_predictions, 60, axis = 0)
bsa_percentile_70 = np.percentile(replicas_bsa_predictions, 70, axis = 0)
bsa_percentile_80 = np.percentile(replicas_bsa_predictions, 80, axis = 0)
bsa_percentile_90 = np.percentile(replicas_bsa_predictions, 90, axis = 0)

cross_section_km15 = DifferentialCrossSection(
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

bkm10_cross_sections_km15 = cross_section_km15.compute_cross_section(phi_array_in_radians).real
bkm10_bsa_km15 = cross_section_km15.compute_bsa(phi_array_in_radians).real

##########################################
# Figures for cross-section predictions:
##########################################
fig2, ax2 = plt.subplots(1, figsize = (7, 7))

ax2.scatter(
    phi_array_in_radians, bkm10_cross_sections_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax2.plot(
    phi_array_in_radians,
    mean_xs,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_max,
    y2 = xs_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_90,
    y2 = xs_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_80,
    y2 = xs_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_70,
    y2 = xs_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax2.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_60,
    y2 = xs_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

ax2.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
ax2.set_ylabel(r"$d^{4}\sigma$ [nb / GeV$^{4}$]", fontsize = 16)
ax2.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
plt.legend()
fig2.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/dnn_xsec_vs_phi_v{MAJOR_MINOR_NUMBER}.png")
fig2.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/dnn_xsec_vs_phi_v{MAJOR_MINOR_NUMBER}.eps")
plt.show()
plt.close(fig2)

##########################################
# Figures for BSA predictions:
##########################################
fig3, ax3 = plt.subplots(1, figsize = (7, 7))

ax3.scatter(
    phi_array_in_radians, bkm10_bsa_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")

ax3.plot(
    phi_array_in_radians,
    mean_bsa,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_max,
    y2 = bsa_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_90,
    y2 = bsa_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_80,
    y2 = bsa_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_70,
    y2 = bsa_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)

ax3.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_60,
    y2 = bsa_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)

ax3.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
ax3.set_ylabel(r"BSA", fontsize = 16)
ax3.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
plt.legend()
fig3.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/dnn_bsa_vs_phi_v{MAJOR_MINOR_NUMBER}.png")
fig3.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/dnn_bsa_vs_phi_v{MAJOR_MINOR_NUMBER}.eps")
plt.show()
plt.close(fig3)

##########################################
# Making replica ensemble predictions for
# BSA
##########################################

mean_bsa = np.mean(replicas_bsa_predictions, axis = 0)
std_dev_bsa = np.std(replicas_bsa_predictions, axis = 0)

bsa_mean = np.mean(replicas_bsa_predictions, axis = 0)
bsa_min = np.min(replicas_bsa_predictions, axis = 0)
bsa_max = np.max(replicas_bsa_predictions, axis = 0)
bsa_q1 = np.percentile(replicas_bsa_predictions, 25, axis = 0)
bsa_q3 = np.percentile(replicas_bsa_predictions, 75, axis = 0)

bsa_percentile_10 = np.percentile(replicas_bsa_predictions, 10, axis = 0)
bsa_percentile_20 = np.percentile(replicas_bsa_predictions, 20, axis = 0)
bsa_percentile_30 = np.percentile(replicas_bsa_predictions, 30, axis = 0)
bsa_percentile_40 = np.percentile(replicas_bsa_predictions, 40, axis = 0)
bsa_median = np.percentile(replicas_bsa_predictions, 50, axis = 0)
bsa_percentile_60 = np.percentile(replicas_bsa_predictions, 60, axis = 0)
bsa_percentile_70 = np.percentile(replicas_bsa_predictions, 70, axis = 0)
bsa_percentile_80 = np.percentile(replicas_bsa_predictions, 80, axis = 0)
bsa_percentile_90 = np.percentile(replicas_bsa_predictions, 90, axis = 0)

##########################################
# Figures for CFF Predictions
##########################################

cff_h_real_pred_per_replica = np.mean(all_predictions[:, :, 0], axis = 1)
cff_h_imag_pred_per_replica = np.mean(all_predictions[:, :, 1], axis = 1)
cff_h_real_mean, cff_h_real_stddev = norm.fit(cff_h_real_pred_per_replica)
cff_h_imag_mean, cff_h_imag_stddev = norm.fit(cff_h_imag_pred_per_replica)

print(f"[NOTE]: Re[H] mean of {cff_h_real_mean} and stdddev of {cff_h_real_stddev}")
print(f"[NOTE]: Im[H] mean of {cff_h_imag_mean} and stdddev of {cff_h_imag_stddev}")

burner_x_values_cff_h_real = np.linspace(
    cff_h_real_mean - 4.*cff_h_real_stddev,
    cff_h_real_mean + 4.*cff_h_real_stddev,
    200)

fig4, ax4 = plt.subplots(1, 1, figsize = (9, 7))
ax4.hist(cff_h_real_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax4.plot(
    burner_x_values_cff_h_real, norm.pdf(burner_x_values_cff_h_real, cff_h_real_mean, cff_h_real_stddev), 
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_h_real_mean:.3f}$, $\sigma = {cff_h_real_stddev:.3f}$")
ax4.axvline(CFF_REAL_E_KM15, color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {CFF_REAL_E_KM15:.3f}")
ax4.set_ylabel("Frequency", rotation = 90.)
ax4.set_xlabel(r"Re$[\mathcal{H}]$")
ax4.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
ax4.legend()
fig4.savefig(f"./version_{VERSION_NUMBER}/plots/cff_h_real_fits_v{MAJOR_MINOR_NUMBER}.png")
fig4.savefig(f"./version_{VERSION_NUMBER}/plots/cff_h_real_fits_v{MAJOR_MINOR_NUMBER}.eps")
plt.show()
plt.close(fig4)

burner_x_values_cff_h_imag = np.linspace(
    cff_h_imag_mean - 4.*cff_h_imag_stddev,
    cff_h_imag_mean + 4.*cff_h_imag_stddev,
    200)

fig5, ax5 = plt.subplots(1, 1, figsize = (9, 7))
ax5.hist(cff_h_imag_pred_per_replica, bins = 30, alpha = 0.6, color = 'skyblue', edgecolor = 'black')
ax5.plot(
    burner_x_values_cff_h_imag, norm.pdf(burner_x_values_cff_h_imag, cff_h_imag_mean, cff_h_imag_stddev), 
    color = "red", linestyle = "--", label = fr"Gaussian Fit: $\mu = {cff_h_imag_mean:.3f}$, $\sigma = {cff_h_imag_stddev:.3f}$")
ax5.axvline(CFF_IMAG_E_KM15, color = "green", linestyle = "-", linewidth = 2., label = f"KM15: {CFF_IMAG_E_KM15:.3f}")
ax5.set_ylabel("Frequency", rotation = 90.)
ax5.set_xlabel(r"Im$[\mathcal{H}]$")
ax5.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
ax5.legend()
fig5.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/cff_h_imag_fits_v{MAJOR_MINOR_NUMBER}.png")
fig5.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/cff_h_imag_fits_v{MAJOR_MINOR_NUMBER}.eps")
plt.show()
plt.close(fig5)

tf.keras.backend.clear_session()

predicted_h_bkm10 = DifferentialCrossSection(
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

bkm10_cross_sections_dnn = predicted_h_bkm10.compute_cross_section(phi_array_in_radians).real
bkm10_bsa_dnn = predicted_h_bkm10.compute_bsa(phi_array_in_radians).real

post_cff_fit_xsec_figure, post_cff_fit_xsec_axis = plt.subplots(1, figsize = (7, 7))
post_cff_fit_xsec_axis.scatter(
    phi_array_in_radians, bkm10_cross_sections_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")
post_cff_fit_xsec_axis.scatter(
    phi_array_in_radians, bkm10_cross_sections_dnn,
    s = 4., label = "BKM10 Prediction with DNN CFFs", color = "red")
post_cff_fit_xsec_axis.plot(
    phi_array_in_radians,
    mean_xs,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')
post_cff_fit_xsec_axis.fill_between(
    x = phi_array_in_radians,
    y1 = xs_max,
    y2 = xs_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)
post_cff_fit_xsec_axis.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_90,
    y2 = xs_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)
post_cff_fit_xsec_axis.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_80,
    y2 = xs_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)
post_cff_fit_xsec_axis.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_70,
    y2 = xs_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)
post_cff_fit_xsec_axis.fill_between(
    x = phi_array_in_radians,
    y1 = xs_percentile_60,
    y2 = xs_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)
post_cff_fit_xsec_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
post_cff_fit_xsec_axis.set_ylabel(r"$d^{4}\sigma$ [nb / GeV$^{4}$]", fontsize = 16)
post_cff_fit_xsec_axis.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
plt.legend()
post_cff_fit_xsec_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/cross_section_comparison_v{MAJOR_MINOR_NUMBER}.png")
post_cff_fit_xsec_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/cross_section_comparison_v{MAJOR_MINOR_NUMBER}.eps")
plt.show()
plt.close(post_cff_fit_xsec_figure)

post_cff_fit_bsa_figure, post_cff_fit_bsa_axis = plt.subplots(1, figsize = (7, 7))  
post_cff_fit_bsa_axis.scatter(
    phi_array_in_radians, bkm10_bsa_km15,
    s = 4., label = "BKM10 Prediction with KM15 CFFs", color = "blue")
post_cff_fit_bsa_axis.scatter(
    phi_array_in_radians, bkm10_bsa_dnn,
    s = 4., label = "BKM10 Prediction with XDJ CFFs", color = "red")
post_cff_fit_bsa_axis.plot(
    phi_array_in_radians,
    mean_bsa,
    label = r'Replica Average',
    color = "blue",
    linewidth = 0.5,
    linestyle = 'dashed')
post_cff_fit_bsa_axis.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_max,
    y2 = bsa_min,
    label = r'Min/Max Bound',
    color = "lightgray",
    alpha = 0.2)
post_cff_fit_bsa_axis.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_90,
    y2 = bsa_percentile_10,
    label = r'10/90 \% Bound',
    color = "gray",
    alpha = 0.25)
post_cff_fit_bsa_axis.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_80,
    y2 = bsa_percentile_20,
    label = r'20/80 \% Bound',
    color = "gray",
    alpha = 0.3)
post_cff_fit_bsa_axis.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_70,
    y2 = bsa_percentile_30,
    label = r'30/70 \% Bound',
    color = "gray",
    alpha = 0.35)
post_cff_fit_bsa_axis.fill_between(
    x = phi_array_in_radians,
    y1 = bsa_percentile_60,
    y2 = bsa_percentile_40,
    label = r'40/60 \% Bound',
    color = "gray",
    alpha = 0.4)
post_cff_fit_bsa_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16)
post_cff_fit_bsa_axis.set_ylabel(r"BSA", fontsize = 16)
post_cff_fit_bsa_axis.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
plt.legend()
post_cff_fit_bsa_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/bsa_comparison_v{MAJOR_MINOR_NUMBER}.png")
post_cff_fit_bsa_figure.savefig(f"{SCRATCH_PATH}/version_{VERSION_NUMBER}/plots/bsa_comparison_v{MAJOR_MINOR_NUMBER}.eps")
plt.show()
plt.close(post_cff_fit_bsa_figure)