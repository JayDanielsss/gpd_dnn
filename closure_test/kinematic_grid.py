##########################################
# FILE INFORMATION:
# Purpose: generates a .csv file containing
# pseudodata for training DNN models down
# the line.
# Created: 20260107
# Last changed: 20260216
##########################################

print(f"[INFO]: Script began running!")

##########################################
# Importing Python Libraries
##########################################

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
BEAM_K_LOWER = 5.5
BEAM_K_UPPER = 11.0
Q_SQUARED_LOWER = 1.0
Q_SQUARED_UPPER = 5.0
X_B_LOWER = 0.1
X_B_UPPER = 0.6
T_LOWER = -1.0
T_UPPER = -.1

# number of points for each variable
NUMBER_OF_BEAM_K = 1
NUMBER_OF_Q_SQUARED = 1
NUMBER_OF_X_B = 1
NUMBER_OF_T = 1

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
# Tons of TensorFlow Functions
##########################################

# BE EXTREMELY CAREFUL ABOUT THIS SETTING!
# It can change numerical precision!
#_FLOATX = tf.float32

# below did NOT work for fixing the stupid tf type mismatch
# tf.keras.mixed_precision.set_global_policy('float32')

# _CONVERSION_GEV6_GEV4NB = tf.constant(.389379 * 1000000., dtype = _FLOATX)
# _MASS_OF_PROTON_IN_GEV = tf.constant(0.93827208816, dtype = _FLOATX)
# _QED_FINE_STRUCTURE = tf.constant(1./137.035999177, dtype = _FLOATX)
# _ELECTRIC_FORM_FACTOR_CONSTANT = tf.constant(0.710649, dtype = _FLOATX)
# _PROTON_MAGNETIC_MOMENT = tf.constant(2.79284734463, dtype = _FLOATX)

_CONVERSION_GEV6_GEV4NB = .389379 * 1000000.
_MASS_OF_PROTON_IN_GEV = 0.93827208816
_QED_FINE_STRUCTURE = 1./137.035999177
_ELECTRIC_FORM_FACTOR_CONSTANT = 0.710649
_PROTON_MAGNETIC_MOMENT = 2.79284734463

print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print(tf.test.gpu_device_name())

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

def compute_k_dot_delta(q_squared, xb, t, phi_azi, ep, y_lep, k):
    return ((-1.*q_squared / (2.*y_lep*(1.+ep**2))) * (1. + ((2.*k*np.cos(tf.constant(np.pi)- phi_azi)) - ((t / q_squared)*(1.-(xb * (2. - y_lep)) + (y_lep * ep**2 / 2.))) + (y_lep * ep**2 / 2.))))

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

def bh_lp_c0(
    lep_helicity: float,target_polar: float,
    q_sq: float, xb: float, t: float,ep: float,y: float, f1: float, f2: float) -> float:
    sum_of_form_factors = (f1 + f2)
    t_over_four_mp_squared = t / (4. * _MASS_OF_PROTON_IN_GEV**2)
    weighted_sum_of_form_factors = f1 + t_over_four_mp_squared * f2
    one_minus_xb = 1. - xb
    t_over_Q_squared = t / q_sq
    one_minus_t_over_Q_squared = 1. - t_over_Q_squared
    first_term_first_bracket = 0.5 * xb * (one_minus_t_over_Q_squared) - t_over_four_mp_squared
    first_term_second_bracket = 2. - xb - (2. * (one_minus_xb)**2 * t_over_Q_squared) + (ep**2 * one_minus_t_over_Q_squared) - (xb * (1. - 2. * xb) * t_over_Q_squared**2)
    first_term = 0.5 * sum_of_form_factors * first_term_first_bracket * first_term_second_bracket
    second_term_first_bracket = xb**2 * (1. + t_over_Q_squared)**2 / (4. * t_over_four_mp_squared) + ((1. - xb) * (1. + xb * t_over_Q_squared))
    second_term = (1. - (1. - xb) * t_over_Q_squared) * weighted_sum_of_form_factors * second_term_first_bracket
    prefactor = 8. * float(lep_helicity) * float(target_polar) * xb * (2. - y) * y * np.sqrt(1. + ep**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
    c0LP_BH = prefactor * (first_term + second_term)
    return c0LP_BH

def bh_lp_c1(
    lep_helicity: float,target_polar: float,
    q_sq: float, xb: float, t: float,ep: float,y: float,shorthand_k: float,f1: float, f2: float) -> float:
    sum_of_form_factors = (f1 + f2)
    t_over_four_mp_squared = t / (4. * _MASS_OF_PROTON_IN_GEV**2)
    weighted_sum_of_form_factors = f1 + t_over_four_mp_squared * f2
    t_over_Q_squared = t / q_sq
    first_term = ((2. * t_over_four_mp_squared) - (xb * (1. - t_over_Q_squared))) * ((1. - xb + (xb * t_over_Q_squared))) * sum_of_form_factors
    second_term_bracket_term = 1. + xb - ((3. - 2. * xb) * (1. + xb * t_over_Q_squared)) - (xb**2 * (1. + t_over_Q_squared**2) / t_over_four_mp_squared)
    second_term = weighted_sum_of_form_factors * second_term_bracket_term
    prefactor = -8. * lep_helicity * target_polar * xb * y * shorthand_k * np.sqrt(1. + ep**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
    c1LP_BH = prefactor * (first_term + second_term)
    return c1LP_BH

def bh_squared(lep_helicity, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2):

    bh_c0 = bh_unp_c0(q_sq, xb, t, ep, y, k, f1, f2)
    bh_c1 = bh_unp_c1(q_sq, xb, t, ep, y, k, f1, f2)
    bh_c2 = bh_unp_c2(xb, t, k, f1, f2)

    bh_c0_lp = bh_lp_c0(lep_helicity, target_polar, q_sq, xb, t, ep, y, f1, f2)
    bh_c1_lp = bh_lp_c1(lep_helicity, target_polar, q_sq, xb, t, ep, y, k, f1, f2)

    return ((
        (bh_c0 + bh_c0_lp) * np.cos(0.* (tf.constant(np.pi) - phi)) +
        (bh_c1 + bh_c1_lp) * np.cos(1.* (tf.constant(np.pi) - phi)) + 
        (bh_c2 + 0.0) * np.cos(2.* (tf.constant(np.pi) - phi))) / (xb * xb * y * y * (1.+ep**2)**2 * t * p1 * p2))

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

def curly_c_real_lp(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):

    first_line=(4.*(1.-xb+(((3.-2.*xb)*q_sq + t))*ep*ep/(4.*(q_sq+xb*t)))*(cff_re_h*cff_re_ht_star-cff_im_ht*cff_im_h_star+cff_re_ht*cff_re_h_star-cff_im_h*cff_im_ht_star))
    second_line=-xb*xb*(q_sq-xb*t*(1.-2.*xb))*(cff_re_h*cff_re_et_star-cff_im_et*cff_im_h_star+cff_re_et*cff_re_h_star-cff_im_h*cff_im_et_star+cff_re_ht*cff_re_e_star-cff_im_e*cff_im_ht_star+cff_re_e*cff_re_ht_star-cff_im_ht*cff_im_e_star)/(q_sq+xb*t)
    third_line=-xb*((4.*(1.-xb)*(q_sq+xb*t)*t+ep*ep*(q_sq+t)**2)*(cff_re_ht*cff_re_e_star-cff_im_e*cff_im_ht_star+cff_re_e*cff_re_ht_star-cff_im_ht*cff_im_e_star))/(2.*q_sq*(q_sq+xb*t))
    fourth_line=-xb*((q_sq*(2.-xb)+xb*t)/(q_sq+xb*t))*((xb*xb*(q_sq+t)**2)/(2*q_sq*(q_sq*(2.-xb)+xb*t)) + t/(4.*_MASS_OF_PROTON_IN_GEV**2))*(cff_re_e*cff_re_et_star-cff_im_e*cff_im_et_star+cff_re_et*cff_re_e_star-cff_im_et*cff_im_e_star)

    return ((first_line+second_line+third_line+fourth_line)*q_sq*(q_sq+xb*t)/(np.sqrt(1.+ep*ep)*((2.-xb)*q_sq+xb*t)**2))

def curly_c_imag_lp(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):

    first_line=(4.*(1.-xb+(((3.-2.*xb)*q_sq + t))*ep*ep/(4.*(q_sq+xb*t)))*(cff_im_h*cff_re_ht_star+cff_re_ht*cff_im_h_star+cff_im_ht*cff_re_h_star+cff_re_h*cff_im_ht_star))
    second_line=-xb*xb*(q_sq-xb*t*(1.-2.*xb))*(cff_im_h*cff_re_et_star+cff_re_et*cff_im_h_star+cff_im_et*cff_re_h_star+cff_re_h*cff_im_et_star+cff_im_ht*cff_re_e_star+cff_re_e*cff_im_ht_star+cff_im_e*cff_re_ht_star+cff_re_ht*cff_im_e_star)/(q_sq+xb*t)
    third_line=-xb*((4.*(1.-xb)*(q_sq+xb*t)*t+ep*ep*(q_sq+t)**2)*(cff_im_h*cff_re_et_star+cff_re_et*cff_im_h_star+cff_im_et*cff_re_h_star+cff_re_h*cff_im_et_star)/(2.*q_sq*(q_sq+xb*t)))
    fourth_line=-xb*((q_sq*(2.-xb)+xb*t)/(q_sq+xb*t))*((xb*xb*(q_sq+t)**2)/(2*q_sq*(q_sq*(2.-xb)+xb*t)) + t/(4.*_MASS_OF_PROTON_IN_GEV**2))*(cff_im_e*cff_re_et_star+cff_re_e*cff_im_et_star+cff_im_et*cff_re_e_star+cff_re_et*cff_im_e_star)

    return ((first_line+second_line+third_line+fourth_line)*q_sq*(q_sq+xb*t)/(np.sqrt(1.+ep*ep)*((2.-xb)*q_sq+xb*t)**2))

def dvcs_unp_c0(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,
    cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
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
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, -1.*cff_im_h, use_ww), f_eff(xi, -1.*cff_im_ht, use_ww), f_eff(xi, -1.*cff_im_e, use_ww), f_eff(xi, -1.*cff_im_et, use_ww))
    c0_dvcs_unpolarized_coefficient = first_term_prefactor * first_term_curlyc + second_term_prefactor * second_term_curlyc
    return c0_dvcs_unpolarized_coefficient

def dvcs_unp_c1(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
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
    lep_helicity: float,q_sq: float,xb: float,t: float,ep: float,y: float,xi: float, k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,
    cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:
    prefactor = -8. * k * lep_helicity * y * np.sqrt(1. + ep**2) / ((2. - xb) * (1. + ep**2))
    curlyC_unp_DVCS = curly_c_imag(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_lp_c0(
    lep_helicity: float, target_polar: float,
    q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = 2.*lep_helicity*target_polar*y*(2.-y)/np.sqrt(1.+ep*ep)
    first_term_curlyc = curly_c_real_lp(
        q_sq, xb, t, ep,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        cff_im_h, cff_im_ht, cff_im_e, cff_im_et,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * first_term_curlyc)

def dvcs_lp_c1(
    lep_helicity: float, target_polar: float,
    q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = 8.*target_polar*k*lep_helicity*y*np.sqrt(1+ep*ep)/((2.-xb)*(1.+ep*ep))
    curlyC_unp_DVCS = curly_c_real_lp(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_lp_s1(
    lep_helicity: float, target_polar: float,
    q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = -8.*target_polar*k*(2.-y)/((2.-xb)*(1.+ep*ep))
    curlyC_unp_DVCS = curly_c_imag_lp(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_squared(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, phi,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    dvcs_c0 = dvcs_unp_c0(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_c1 = dvcs_unp_c1(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_s1 = dvcs_unp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    
    dvcs_c0_lp = dvcs_lp_c0(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_c1_lp = dvcs_lp_c1(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_s1_lp = dvcs_lp_s1(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    return (
        (
            (dvcs_c0 + dvcs_c0_lp) * np.cos(0.* (tf.constant(np.pi) - phi)) +
            (dvcs_c1 + dvcs_c1_lp) * np.cos(1.* (tf.constant(np.pi) - phi)) + 
            (dvcs_s1 + dvcs_s1_lp) * np.sin(1.* (tf.constant(np.pi) - phi))) / (y * y * q_sq))

def i_c_unp_pp_0(
    q_sq: float,xb: float,t: float,ep: float,y: float,k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    two_minus_xb = 2. - xb
    two_minus_y = 2. - y
    first_term_in_brackets = k_tilde**2 * two_minus_y**2 / (q_sq * root_one_plus_epsilon_squared)
    second_term_in_brackets_first_part = t_over_Q_squared * two_minus_xb * (1. - y - (ep**2 * y**2 / 4.))
    second_term_in_brackets_second_part_numerator = 2. * xb * t_over_Q_squared * (two_minus_xb + 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep**2 / xb) + ep**2
    second_term_in_brackets_second_part =  1. + second_term_in_brackets_second_part_numerator / (two_minus_xb * one_plus_root_epsilon_stuff)
    prefactor = -4. * two_minus_y * one_plus_root_epsilon_stuff / np.power(root_one_plus_epsilon_squared, 4)
    c_0_plus_plus_unp = prefactor * (first_term_in_brackets + second_term_in_brackets_first_part * second_term_in_brackets_second_part)
    return c_0_plus_plus_unp

def i_c_unp_v_pp_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
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
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
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
    prefactor = 12. * np.sqrt(2.) * k * (2. - y) * np.sqrt(1. - y - (ep**2 * y**2 / 4)) / np.power(1. + ep**2, 2.5)
    c_0_zero_plus_unp = prefactor * bracket_quantity
    return c_0_zero_plus_unp

def i_c_unp_v_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    t_over_Q_squared = t / q_sq
    main_part = xb * t_over_Q_squared * (1. - (1. - 2. * xb) * t_over_Q_squared)
    prefactor = 24. * np.sqrt(2.) * k * (2. - y) * np.sqrt(1. - y - (y**2 * ep**2 / 4.)) / (1. + ep**2)**2.5
    c_0_zero_plus_V_unp = prefactor * main_part
    return c_0_zero_plus_V_unp

def i_c_unp_a_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    t_over_Q_squared = t / q_sq
    fancy_xb_epsilon_term = 8. - 6. * xb + 5. * ep**2
    brackets_term = 1. - t_over_Q_squared * (2. - 12. * xb * (1. - xb) - ep**2) / fancy_xb_epsilon_term
    prefactor = 4. * np.sqrt(2.) * k * (2. - y) * np.sqrt(1. - y - (y**2 * ep**2 / 4.)) / np.power(1. + ep**2, 2.5)
    c_0_zero_plus_A_unp = prefactor * t_over_Q_squared * fancy_xb_epsilon_term * brackets_term
    return c_0_zero_plus_A_unp

def i_c_unp_pp_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_first_term = (1. + (1. - x_Bjorken) * (root_one_plus_epsilon_squared - 1.) / (2. * x_Bjorken) + epsilon**2 / (4. * x_Bjorken)) * x_Bjorken * t_over_Q_squared
    first_bracket_term = first_bracket_first_term - 3. * epsilon**2 / 4.
    second_bracket_term = 1. - (1. - 3. * x_Bjorken) * t_over_Q_squared + (1. - root_one_plus_epsilon_squared + 3. * epsilon**2) * x_Bjorken * t_over_Q_squared / (one_plus_root_epsilon_stuff - epsilon**2)
    fancy_y_coefficient = 2. - 2. * lepton_energy_fraction_y + lepton_energy_fraction_y**2 + epsilon**2 * lepton_energy_fraction_y**2 / 2.
    second_term = -4. * shorthand_k * fancy_y_coefficient * (one_plus_root_epsilon_stuff - epsilon**2) * second_bracket_term / root_one_plus_epsilon_squared**5
    first_term = -16. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * first_bracket_term / root_one_plus_epsilon_squared**5
    c_1_plus_plus_unp = first_term + second_term
    return c_1_plus_plus_unp

def i_c_unp_v_pp_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    first_bracket_term = (2. - lepton_energy_fraction_y)**2 * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)
    second_bracket_term_first_part = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    second_bracket_term_second_part = 0.5 * (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * t_prime / squared_Q_momentum_transfer
    coefficient_prefactor = 16. * shorthand_k * x_Bjorken * t_over_Q_squared / np.power(root_one_plus_epsilon_squared, 5)
    c_1_plus_plus_V_unp = coefficient_prefactor * (first_bracket_term + second_bracket_term_first_part * second_bracket_term_second_part)
    return c_1_plus_plus_V_unp

def i_c_unp_a_pp_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    one_minus_2xb = 1. - 2. * x_Bjorken
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    first_bracket_term_second_part = 1. - one_minus_2xb * t_over_Q_squared + (4. * x_Bjorken * one_minus_xb + epsilon**2) * t_prime_over_Q_squared / (4. * root_one_plus_epsilon_squared)
    second_bracket_term = 1. - 0.5 * x_Bjorken + 0.25 * (one_minus_2xb + root_one_plus_epsilon_squared) * (1. - t_over_Q_squared) + (4. * x_Bjorken * one_minus_xb + epsilon**2) * t_prime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -16. * shorthand_k * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_1_plus_plus_A_unp = prefactor * (fancy_y_stuff * first_bracket_term_second_part - (2. - lepton_energy_fraction_y)**2 * second_bracket_term)
    return c_1_plus_plus_A_unp

def i_c_unp_0p_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    first_bracket_term = (2. - lepton_energy_fraction_y)**2 * t_prime_over_Q_squared * (one_minus_xb + (one_minus_xb * x_Bjorken + (epsilon**2 / 4.)) * t_prime_over_Q_squared / root_one_plus_epsilon_squared)
    second_bracket_term = y_quantity * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared) * (epsilon**2 - 2. * (1. + (epsilon**2 / (2. * x_Bjorken))) * x_Bjorken * t_over_Q_squared) / root_one_plus_epsilon_squared
    prefactor = 8. * np.sqrt(2. * y_quantity) / root_one_plus_epsilon_squared**4
    c_1_zero_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_1_zero_plus_unp

def i_c_unp_v_0p_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    k_tilde: float):
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    major_part = (2 - lepton_energy_fraction_y)**2 * k_tilde**2 / squared_Q_momentum_transfer + (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)**2 * y_quantity
    prefactor = 16. * np.sqrt(2. * y_quantity) * x_Bjorken * t_over_Q_squared / (1. + epsilon**2)**2.5
    c_1_zero_plus_V_unp = prefactor * major_part
    return c_1_zero_plus_V_unp

def i_c_unp_a_0p_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    one_minus_2xb = 1. - 2. * x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    second_term_first_part = (1. - one_minus_2xb * t_over_Q_squared) * y_quantity
    second_term_second_part = 4. - 2. * x_Bjorken + 3. * epsilon**2 + t_over_Q_squared * (4. * x_Bjorken * (1. - x_Bjorken) + epsilon**2)
    first_term = k_tilde**2 * one_minus_2xb * (2. - lepton_energy_fraction_y)**2 / squared_Q_momentum_transfer
    prefactor = 8. * np.sqrt(2. * y_quantity) * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_1_zero_plus_unp_A = prefactor * (first_term + second_term_first_part * second_term_second_part)
    return c_1_zero_plus_unp_A

def i_c_unp_pp_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    first_bracket_term = 2. * epsilon**2 * k_tilde**2 / (root_one_plus_epsilon_squared * (1. + root_one_plus_epsilon_squared) * squared_Q_momentum_transfer)
    second_bracket_term = x_Bjorken * t_prime * t_over_Q_squared * (1. - x_Bjorken - 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * epsilon**2 / x_Bjorken) / squared_Q_momentum_transfer
    prefactor = 8. * (2. - lepton_energy_fraction_y) * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) / root_one_plus_epsilon_squared**4
    c_2_plus_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_2_plus_plus_unp

def i_c_unp_v_pp_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    major_term = (4. * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)) + 0.5 * (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * (1. + t_over_Q_squared) * t_prime_over_Q_squared
    prefactor = 8. * (2. - lepton_energy_fraction_y) * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_V_unp = prefactor * major_term
    return c_2_plus_plus_V_unp

def i_c_unp_a_pp_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    first_bracket_term = 4. * (1. - 2. * x_Bjorken) * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)
    second_bracket_term = (3.  - root_one_plus_epsilon_squared - 2. * x_Bjorken + epsilon**2 / x_Bjorken ) * x_Bjorken * t_prime_over_Q_squared
    prefactor = 4. * (2. - lepton_energy_fraction_y) * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_A_unp = prefactor * (first_bracket_term - second_bracket_term)
    return c_2_plus_plus_A_unp

def i_c_unp_0p_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    epsilon_squared_over_2 = epsilon**2 / 2.
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / x_Bjorken) / (1. + epsilon_squared_over_2)) * x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    prefactor = -8. * np.sqrt(2. * y_quantity) * shorthand_k * (2. - lepton_energy_fraction_y) / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return c_2_zero_plus_unp

def i_c_unp_v_0p_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    y_quantity = np.sqrt(1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * y_quantity * shorthand_k * (2. - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)
    return c_2_zero_plus_unp_V

def i_c_unp_a_0p_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    bracket_term = one_minus_xb + 0.5 * t_prime_over_Q_squared * (4. * x_Bjorken * one_minus_xb + epsilon**2) / root_one_plus_epsilon_squared
    prefactor = 8. * np.sqrt(2. * y_quantity) * shorthand_k * (2. - lepton_energy_fraction_y) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * bracket_term
    return c_2_zero_plus_unp_A

def i_c_unp_pp_3(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    major_term = (1. - x_Bjorken) * t_over_Q_squared + 0.5 * (root_one_plus_epsilon_squared - 1.) * (1. + t_over_Q_squared)
    intermediate_term = (root_one_plus_epsilon_squared - 1.) / root_one_plus_epsilon_squared**5
    prefactor = -8. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    c_3_plus_plus_unp = prefactor * intermediate_term * major_term
    return c_3_plus_plus_unp

def i_c_unp_v_pp_3(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    major_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * t_over_Q_squared
    prefactor = -8. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_3_plus_plus_V_unp = prefactor * major_term
    return c_3_plus_plus_V_unp

def i_c_unp_a_pp_3(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    main_term = squared_hadronic_momentum_transfer_t * t_prime * (x_Bjorken * (1. - x_Bjorken) + epsilon**2 / 4.) / squared_Q_momentum_transfer**2
    prefactor = 16. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) / (1. + epsilon**2)**2.5
    c_3_plus_plus_A_unp = prefactor * main_term
    return c_3_plus_plus_A_unp

def i_s_unp_pp_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    bracket_term = 1. + ((1. - x_Bjorken + 0.5 * (root_one_plus_epsilon_squared - 1.)) / root_one_plus_epsilon_squared**2) * tPrime_over_Q_squared
    prefactor = 8. * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp = prefactor * bracket_term
    return s_1_plus_plus_unp

def i_s_unp_v_pp_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    bracket_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * t_over_Q_squared
    prefactor = -8. * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_1_plus_plus_unp_V = prefactor * bracket_term
    return s_1_plus_plus_unp_V

def i_s_unp_a_pp_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_2xb = 1. - 2. * x_Bjorken
    bracket_term = 1. - one_minus_2xb * (one_minus_2xb + root_one_plus_epsilon_squared) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = 8. * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * t_over_Q_squared / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp_A = prefactor * bracket_term
    return s_1_plus_plus_unp_A

def i_s_unp_0p_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    epsilon: float,
    lepton_energy_fraction_y: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = (1. + epsilon**2)**2
    y_quantity = np.sqrt(1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.))
    s_1_zero_plus_unp = 8. * np.sqrt(2.) * lepton_helicity * (2. - lepton_energy_fraction_y) * lepton_energy_fraction_y * y_quantity * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)
    return s_1_zero_plus_unp

def i_s_unp_v_0p_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float):
    one_plus_epsilon_squared_squared = (1. + epsilon**2)**2
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    bracket_term = 4. * (1. - 2. * x_Bjorken) * t_over_Q_squared * (1. + x_Bjorken * t_over_Q_squared) + epsilon**2 * (1. + t_over_Q_squared)**2
    prefactor = 4. * np.sqrt(2. * fancy_y_stuff) * lepton_helicity * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_V = prefactor * bracket_term
    return s_1_zero_plus_unp_V

def i_s_unp_a_0p_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    one_plus_epsilon_squared_squared = (1. + epsilon**2)**2
    fancy_y_stuff = np.sqrt(1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    prefactor = -8. * np.sqrt(2.) * lepton_helicity * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * (1. - 2. * x_Bjorken) / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_A = prefactor * fancy_y_stuff * squared_hadronic_momentum_transfer_t * shorthand_k**2 / squared_Q_momentum_transfer
    return s_1_zero_plus_unp_A

def i_s_unp_pp_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    first_bracket_term = (epsilon**2 - x_Bjorken * (root_one_plus_epsilon_squared - 1.)) / (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken)
    second_bracket_term = (2. * x_Bjorken + epsilon**2) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -4. * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * tPrime_over_Q_squared / root_one_plus_epsilon_squared**3
    s_2_plus_plus_unp = prefactor * (first_bracket_term - second_bracket_term)
    return s_2_plus_plus_unp

def i_s_unp_v_pp_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    one_minus_2xb = 1. - 2. * x_Bjorken
    bracket_term = root_one_plus_epsilon_squared - 1. + (one_minus_2xb + root_one_plus_epsilon_squared) * t_over_Q_squared
    parentheses_term = 1. - one_minus_2xb * t_over_Q_squared
    prefactor = -4. * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_V = prefactor * parentheses_term * bracket_term
    return s_2_plus_plus_unp_V

def i_s_unp_a_pp_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    last_term = 1. + (4. * (1. - x_Bjorken) * x_Bjorken + epsilon**2) * t_over_Q_squared / (4. - 2. * x_Bjorken + 3. * epsilon**2)
    middle_term = 1. + root_one_plus_epsilon_squared - 2. * x_Bjorken
    prefactor = -8. * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * t_over_Q_squared * tPrime_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_A = prefactor * middle_term * last_term
    return s_2_plus_plus_unp_A

def i_s_unp_0p_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    epsilon_squared_over_2 = epsilon**2 / 2.
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / x_Bjorken) / (1. + epsilon_squared_over_2)) * x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    prefactor = 8. * lepton_helicity * np.sqrt(2. * y_quantity) * shorthand_k * lepton_energy_fraction_y / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return s_2_zero_plus_unp

def i_s_unp_v_0p_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    y_quantity = np.sqrt(1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.))
    prefactor = -8. * np.sqrt(2.) * lepton_helicity * y_quantity * shorthand_k * lepton_energy_fraction_y * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)
    return s_2_zero_plus_unp_V

def i_s_unp_a_0p_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    main_term = 4. * one_minus_xb + 2. * epsilon**2 + 4. * t_over_Q_squared * (4. * x_Bjorken * one_minus_xb + epsilon**2)
    prefactor = -2. * np.sqrt(2. * y_quantity) * lepton_helicity * shorthand_k * lepton_energy_fraction_y * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * main_term
    return c_2_zero_plus_unp_A

def i_curly_c_unp(
    squared_Q_momentum_transfer: float,
    x_Bjorken: float,
    squared_hadronic_momentum_transfer_t: float,
    Dirac_form_factor_F1: float,
    Pauli_form_factor_F2: float,
    cff_h: float,
    cff_h_tilde: float,
    cff_e: float) -> float:
    weighted_cffs = (Dirac_form_factor_F1 * cff_h) - (squared_hadronic_momentum_transfer_t * Pauli_form_factor_F2 * cff_e / (4. * _MASS_OF_PROTON_IN_GEV**2))
    second_term = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) * cff_h_tilde / (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))
    curly_C_unpolarized_interference = weighted_cffs + second_term
    return curly_C_unpolarized_interference

def i_curly_c_v_unp(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float,
    squared_hadronic_momentum_transfer_t: float,
    Dirac_form_factor_F1: float,
    Pauli_form_factor_F2: float,
    cff_h: float,
    cff_e: float) -> float:
    cff_term = cff_h + cff_e
    second_term = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) / (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))
    curly_C_unpolarized_interference_V = cff_term * second_term
    return curly_C_unpolarized_interference_V

def i_curly_c_a_unp(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float,
    squared_hadronic_momentum_transfer_t: float,
    Dirac_form_factor_F1: float,
    Pauli_form_factor_F2: float,
    cff_ht: float) -> float:
    xb_modulation = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) / (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))
    curly_C_unpolarized_interference_A = cff_ht * xb_modulation
    return curly_C_unpolarized_interference_A

def i_c_lp_pp_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq 
    first_bracket_term = (2. - y)**2 * k_tilde**2 / q_sq
    second_bracket_term_first_part = 1. - y + (ep**2 * y**2 / 4.)
    second_bracket_term_second_part = xb * t_over_Q_squared - (ep**2 * (1. - t_over_Q_squared) / 2.)
    second_bracket_term_third_part = 1. + t_over_Q_squared * ((root_one_plus_epsilon_squared - 1. + 2. * xb) / (1. + root_one_plus_epsilon_squared))
    second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part
    prefactor = -4. * lep_helicity * target_polar * y * (1. + root_one_plus_epsilon_squared) / root_one_plus_epsilon_squared**5
    c_0_plus_plus_LP = prefactor * (first_bracket_term + second_bracket_term)
    return c_0_plus_plus_LP

def i_c_lp_v_pp_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_term = (2. - y)**2 * (one_plus_root_epsilon_stuff - 2. * xb) * k_tilde**2 / (q_sq * one_plus_root_epsilon_stuff)
    second_bracket_term_first_part = 1. - y - (ep**2 * y**2 / 4.)
    second_bracket_term_second_part = 2. - xb + 3. * ep**2 / 2
    second_bracket_term_third_part = 1. + (t_over_Q_squared * (4. * (1. - xb) * xb + ep**2) / (4. - 2. * xb + 3. * ep**2))
    second_bracket_term_fourth_part = 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. + 2. * xb) / one_plus_root_epsilon_stuff)
    second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part * second_bracket_term_fourth_part
    prefactor = 4. * lep_helicity * target_polar * y * one_plus_root_epsilon_stuff * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_0_plus_plus_V_LP = prefactor * (first_bracket_term + second_bracket_term)
    return c_0_plus_plus_V_LP

def i_c_lp_a_pp_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_term = 2. * (2. - y)**2 * k_tilde**2 / q_sq
    second_bracket_term_first_part = 1. - y - (ep**2 * y**2 / 4.)
    second_bracket_term_second_part = 1. - (1. - 2. * xb) * t_over_Q_squared
    second_bracket_term_third_part = 1. + (t_over_Q_squared * (root_one_plus_epsilon_squared - 1. + 2. * xb) / one_plus_root_epsilon_stuff)
    second_bracket_term = second_bracket_term_first_part * one_plus_root_epsilon_stuff * second_bracket_term_second_part * second_bracket_term_third_part
    prefactor = 4. * lep_helicity * target_polar * y * xb * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_0_plus_plus_A_LP = prefactor * (first_bracket_term + second_bracket_term)
    return c_0_plus_plus_A_LP

def i_c_lp_0p_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * lep_helicity * target_polar * shorthand_k * (1. - xb) * y / (1. + ep**2)**2
    c_0_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * t / q_sq
    return c_0_zero_plus_LP

def i_c_lp_v_0p_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    modulating_factor = (xb - (t * (1. - 2. * xb) / q_sq)) / (1. - xb)
    c_0_zero_plus_LP = i_c_lp_0p_0(
        lep_helicity,
        target_polar,
        q_sq, 
        xb, 
        t,
        ep,
        y, 
        shorthand_k)
    c_0_zero_plus_V_LP = c_0_zero_plus_LP * modulating_factor
    return c_0_zero_plus_V_LP

def i_c_lp_a_0p_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * np.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    t_over_Q_squared = t / q_sq
    c_0_zero_plus_A_LP = prefactor * root_combination_of_y_and_epsilon * xb * t_over_Q_squared * (1. + t_over_Q_squared)
    return c_0_zero_plus_A_LP

def i_c_lp_0p_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    ep: float,
    y: float, 
    k_tilde: float,
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * np.sqrt(2.) * lep_helicity * target_polar * shorthand_k * (1. - y) * y / (1. + ep**2)**2
    c_1_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * k_tilde**2 / q_sq
    return c_1_zero_plus_LP

def i_c_lp_v_0p_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * lep_helicity * target_polar  * (2. - y) * y / (1. + ep**2)**2
    c_1_zero_plus_V_LP = prefactor * root_combination_of_y_and_epsilon * t * k_tilde**2 / q_sq**2
    return c_1_zero_plus_V_LP

def i_c_lp_pp_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    one_plus_root_epsilon_minus_epsilon_squared = one_plus_root_epsilon_stuff - ep**2
    major_factor = 1. - ((t / q_sq) * (1. - 2. * xb * (one_plus_root_epsilon_stuff + 1.) / one_plus_root_epsilon_minus_epsilon_squared))
    prefactor = -4. * lep_helicity * target_polar * y * shorthand_k * (2. - y) / root_one_plus_epsilon_squared**5
    c_1_plus_plus_LP = prefactor * one_plus_root_epsilon_minus_epsilon_squared * major_factor
    return c_1_plus_plus_LP

def i_c_lp_v_pp_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    one_minus_xb = 1. - xb
    root_epsilon_and_xb_quantity = root_one_plus_epsilon_squared + 2. * one_minus_xb
    bracket_factor_numerator = 1. + ((1. - ep**2) / root_one_plus_epsilon_squared) - (2. * xb * (1. + (4. * one_minus_xb / root_one_plus_epsilon_squared)))
    bracket_factor_denominator = 2. * root_epsilon_and_xb_quantity
    bracket_factor = 1. - (t_prime * bracket_factor_numerator / (q_sq * bracket_factor_denominator))
    prefactor = 8. * lep_helicity * target_polar * shorthand_k * y * (2. - y) / root_one_plus_epsilon_squared**4
    c_1_plus_plus_V_LP = prefactor * root_epsilon_and_xb_quantity * t * bracket_factor / q_sq
    return c_1_plus_plus_V_LP
    
def i_c_lp_a_pp_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    t_over_Q_squared = t / q_sq
    major_factor = xb * t_over_Q_squared * (1. - (1. - 2. * xb) * t_over_Q_squared)
    prefactor = 16. * lep_helicity * target_polar * shorthand_k * y * (2. - y) / np.sqrt(1. + ep**2)**5
    c_1_plus_plus_A_LP = prefactor * major_factor
    return c_1_plus_plus_A_LP

def i_c_lp_pp_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_multiplicative_factor = (-1. * one_plus_root_epsilon_stuff + 2.) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb)
    second_multiplicative_factor = xb * t_over_Q_squared - (ep**2 * (1. - t_over_Q_squared) / 2.)
    prefactor = -4. * lep_helicity * target_polar * y * (1. - y - (y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**5
    c_2_plus_plus_LP = prefactor * first_multiplicative_factor * second_multiplicative_factor
    return c_2_plus_plus_LP

def i_c_lp_v_pp_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_multiplicative_factor = (one_plus_root_epsilon_stuff - 2.) + t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb)
    second_multiplicative_factor = 1. + (t_over_Q_squared * (4. * (1. - xb) * xb + ep**2 ) / (4. - 2. * xb + 3. * ep**2))
    third_multiplicative_factor = t_over_Q_squared * (4. - 2. * xb + 3. * ep**2)
    prefactor = -2.*lep_helicity*target_polar*y*(1.-y-(y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**5
    c_2_plus_plus_V_LP = prefactor * first_multiplicative_factor * second_multiplicative_factor * third_multiplicative_factor
    return c_2_plus_plus_V_LP

def i_c_lp_a_pp_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_multiplicative_factor = (1. - root_one_plus_epsilon_squared) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb)
    second_multiplicative_factor = xb * t_over_Q_squared * (1. - t_over_Q_squared * (1. - 2. * xb))
    prefactor = 4. * lep_helicity * target_polar * y * (1. - y - (y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**5
    c_2_plus_plus_A_LP = prefactor * first_multiplicative_factor * second_multiplicative_factor
    return c_2_plus_plus_A_LP

def i_c_lp_0p_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * np.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    c_2_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * (1. + (xb * t / q_sq))
    return c_2_zero_plus_LP

def i_c_lp_v_0p_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    c_2_zero_plus_V_LP = prefactor * root_combination_of_y_and_epsilon * (1. - xb ) * t / q_sq
    return c_2_zero_plus_V_LP

def i_c_lp_v_0p_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    c_2_zero_plus_V_LP = prefactor * root_combination_of_y_and_epsilon * (1. - xb ) * t / q_sq
    return c_2_zero_plus_V_LP

def i_c_lp_a_0p_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    t_over_Q_squared = t / q_sq
    c_2_zero_plus_A_LP = prefactor * root_combination_of_y_and_epsilon * xb * t_over_Q_squared * (1. + t / q_sq)
    return c_2_zero_plus_A_LP

def i_s_lp_pp_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    t_over_Q_squared = t / q_sq
    epsilon_y_over_2_squared = (ep * y / 2.) ** 2
    first_bracket_term = 2. * root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb) / one_plus_root_epsilon_stuff)
    second_bracket_term = (3. * ep**2 / 2.) + (t_over_Q_squared * (1. - root_one_plus_epsilon_squared - ep**2 / 2. - xb * (3.  - root_one_plus_epsilon_squared)))
    almost_prefactor = 4. * target_polar * shorthand_k / root_one_plus_epsilon_squared**6
    prefactor_one = almost_prefactor * (2. - 2. * y + y**2 + 2. * epsilon_y_over_2_squared) * one_plus_root_epsilon_stuff
    prefactor_two = 2. * almost_prefactor * (1. - y - epsilon_y_over_2_squared)
    s_1_plus_plus_LP = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term
    return s_1_plus_plus_LP

def i_s_lp_v_pp_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    shorthand_k: float) -> float:
    ep_squared = ep**2
    root_one_plus_epsilon_squared = np.sqrt(1. + ep_squared)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = t_prime / q_sq
    epsilon_y_over_2_squared = ep_squared * y**2 / 4.
    first_bracket_term = 1. - (t_prime_over_Q_squared * ((1. - 2. * xb) * (1. - 2. * xb + root_one_plus_epsilon_squared)) / (2. * root_one_plus_epsilon_squared**2))
    second_term_parentheses_term = t_over_Q_squared * (1. - (xb * ((3. + root_one_plus_epsilon_squared) / 4.)) + (5. * ep_squared / 8.))
    second_bracket_term_numerator = 1. - root_one_plus_epsilon_squared + (ep_squared / 2.) - (2. * xb * (3. * (1. - xb) - root_one_plus_epsilon_squared))
    second_bracket_term_denominator = 4. - (xb * (root_one_plus_epsilon_squared + 3.)) + (5. * ep_squared / 2.)
    second_bracket_term = 1. - (t_over_Q_squared * second_bracket_term_numerator / second_bracket_term_denominator)
    almost_prefactor = 8. * target_polar * shorthand_k / root_one_plus_epsilon_squared**4
    prefactor_one = almost_prefactor * (2. - 2. * y + y**2 + 2. * epsilon_y_over_2_squared) * t_over_Q_squared
    prefactor_two = 4. * almost_prefactor * (1. - y - epsilon_y_over_2_squared) / root_one_plus_epsilon_squared**2
    s_1_plus_plus_V_LP = prefactor_one * first_bracket_term + prefactor_two * second_term_parentheses_term * second_bracket_term
    return s_1_plus_plus_V_LP

def i_s_lp_a_pp_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    xB_t_over_Q_squared = xb * t_over_Q_squared
    three_plus_root_epsilon_stuff = 3 + root_one_plus_epsilon_squared
    epsilon_y_over_2_squared = (ep * y / 2.) ** 2
    almost_prefactor = 8. * target_polar * shorthand_k / root_one_plus_epsilon_squared**6
    first_bracket_term = root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (1. + root_one_plus_epsilon_squared - 2. * xb))
    second_bracket_term = 1. - (t_over_Q_squared * (3.  - root_one_plus_epsilon_squared - 6. * xb) / three_plus_root_epsilon_stuff)
    prefactor_one = -1. * almost_prefactor * (2. - 2. * y + y**2 + 2. * epsilon_y_over_2_squared) * xB_t_over_Q_squared
    prefactor_two = almost_prefactor * (1. - y - epsilon_y_over_2_squared) * three_plus_root_epsilon_stuff * xB_t_over_Q_squared
    s_1_plus_plus_A_LP = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term
    return s_1_plus_plus_A_LP

def i_s_lp_0p_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    combination_of_y_and_epsilon = 1. - y - (y**2 * ep**2 / 4.)
    t_over_Q_squared = t / q_sq
    first_bracket_term = k_tilde**2 * (2. - y)**2 / q_sq
    second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * (2. * xb * t_over_Q_squared - (ep**2 * (1. - t_over_Q_squared)))
    prefactor = 8. * np.sqrt(2.) * target_polar  * np.sqrt(combination_of_y_and_epsilon) / np.sqrt((1. + ep**2)**5)
    s_1_zero_plus_LP = prefactor * (first_bracket_term + second_bracket_term)
    return s_1_zero_plus_LP

def i_s_lp_v_0p_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    combination_of_y_and_epsilon = 1. - y - (y**2 * ep**2 / 4.)
    t_over_Q_squared = t / q_sq
    first_bracket_term = k_tilde**2 * (2. - y)**2 / q_sq
    second_bracket_term_long = 4. - 2. * xb + 3. * ep**2 + t_over_Q_squared * (4. * xb * (1. - xb) + ep**2)
    second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * second_bracket_term_long
    prefactor = -8. * np.sqrt(2.) * target_polar  * np.sqrt(combination_of_y_and_epsilon) * t_over_Q_squared / np.sqrt((1. + ep**2)**5)
    s_1_zero_plus_V_LP = prefactor * (first_bracket_term + second_bracket_term)
    return s_1_zero_plus_V_LP

def i_s_lp_a_0p_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    combination_of_y_and_epsilon_to_3_halves = np.sqrt(1. - y - (y**2 * ep**2 / 4.))**3
    t_over_Q_squared = t / q_sq
    prefactor = -16. * np.sqrt(2.) * target_polar * xb * t_over_Q_squared * (1. + t_over_Q_squared) / np.sqrt((1. + ep**2)**5)
    s_1_zero_plus_A_LP = prefactor * combination_of_y_and_epsilon_to_3_halves * (1. - (1. - 2. * xb) * t_over_Q_squared)
    return s_1_zero_plus_A_LP

def i_s_lp_pp_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    bracket_term = 4. * k_tilde**2 * (one_plus_root_epsilon_stuff - 2. * xb) * (one_plus_root_epsilon_stuff + xb * t / q_sq) * t_prime / (root_one_plus_epsilon_squared * q_sq**2)
    prefactor = -4. * target_polar * (2. - y) * (1. - y - (ep**2 * y**2 / 4.)) / root_one_plus_epsilon_squared**5
    s_2_plus_plus_LP = prefactor * bracket_term
    return s_2_plus_plus_LP

def i_s_lp_v_pp_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    bracket_term_second_term = (3.  - root_one_plus_epsilon_squared - (2. * xb) + (ep**2 / xb)) * xb * t_prime / q_sq
    bracket_term_first_term = 4. * k_tilde**2 * (1. - 2. * xb) / (root_one_plus_epsilon_squared * q_sq)
    bracket_term = t * (bracket_term_first_term - bracket_term_second_term) / q_sq
    prefactor = 4. * target_polar * (2. - y) * (1. - y - ep**2 * y**2 / 4.) / root_one_plus_epsilon_squared**5
    s_2_plus_plus_V_LP = prefactor * bracket_term
    return s_2_plus_plus_V_LP

def i_s_lp_a_pp_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    bracket_term_first_term = (1. + root_one_plus_epsilon_squared - 2. * xb) * (1. - ((1. - 2. * xb) * t / q_sq)) * t_prime / q_sq
    bracket_term_second_term = 4. * k_tilde**2 / q_sq
    bracket_term = xb * t * (bracket_term_second_term - bracket_term_first_term) / q_sq
    prefactor = 4. * target_polar * (2. - y) * (1. - y - ep**2 * y**2 / 4.) / root_one_plus_epsilon_squared**5
    s_2_plus_plus_A_LP = prefactor * bracket_term
    return s_2_plus_plus_A_LP

def i_s_lp_0p_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * target_polar * k * (2. - y )/ np.sqrt((1. + ep**2)**5)
    s_2_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * (1. + (xb * t / q_sq))
    return s_2_zero_plus_LP

def i_s_lp_v_0p_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * np.sqrt(2.) * target_polar * shorthand_k * (2. - y) * t / (np.sqrt((1. + ep**2)**5) * q_sq)
    s_2_zero_plus_V_LP = prefactor * (1. - xb) * root_combination_of_y_and_epsilon
    return s_2_zero_plus_V_LP

def i_s_lp_a_0p_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = np.sqrt(1. - y - (y**2 * ep**2 / 4.))
    t_over_Q_squared = t / q_sq
    prefactor = -8. * np.sqrt(2.) * target_polar  * shorthand_k * (2. - y) * xb * t_over_Q_squared / np.sqrt((1. + ep**2)**5)
    s_2_zero_plus_A_LP = prefactor * root_combination_of_y_and_epsilon * (1. + t_over_Q_squared)
    return s_2_zero_plus_A_LP

def i_s_lp_pp_3(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    prefactor = -4. * target_polar * shorthand_k * (1. - y - y**2 * ep**2 / 4.) / root_one_plus_epsilon_squared**6
    s_3_plus_plus_LP = prefactor * (one_plus_root_epsilon_stuff - 2. * xb) * ep**2 * t_prime / (q_sq * one_plus_root_epsilon_stuff)
    return s_3_plus_plus_LP

def i_s_lp_v_pp_3(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    multiplicative_contribution = t * t_prime * (4. * (1. - xb) * xb + ep**2) / q_sq**2
    prefactor = 4. * target_polar * shorthand_k * (1. - y - y**2 * ep**2 / 4.) / root_one_plus_epsilon_squared**6
    s_3_plus_plus_V_LP = prefactor * multiplicative_contribution
    return s_3_plus_plus_V_LP
    
def i_s_lp_a_pp_3(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    multiplicative_contribution = xb * t * t_prime * (1. + root_one_plus_epsilon_squared - 2. * xb) / q_sq**2
    prefactor = -8. * target_polar * shorthand_k * (1. - y - (y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**6
    s_3_plus_plus_A_LP = prefactor * multiplicative_contribution
    return s_3_plus_plus_A_LP

def i_curly_c_lp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_h: float,
    cff_ht: float,
    cff_e: float,
    cff_et: float) -> float:
    t_over_Q_squared = t / q_sq
    ratio_of_xb_to_more_xb = xb / (2. - xb + xb * t_over_Q_squared)
    x_Bjorken_correction = xb * (1. - t_over_Q_squared) / 2.
    first_cff_contribution = ratio_of_xb_to_more_xb * (f1 + f2) * (cff_h + x_Bjorken_correction * cff_e)
    second_cff_contribution = (1. + (_MASS_OF_PROTON_IN_GEV**2 * xb * ratio_of_xb_to_more_xb * (3. + t_over_Q_squared) / q_sq)) * f1 * cff_ht
    third_cff_contribution = t_over_Q_squared * 2. * (1. - 2. * xb) * ratio_of_xb_to_more_xb * f2 * cff_ht
    fourth_cff_contribution = ratio_of_xb_to_more_xb * (x_Bjorken_correction * f1 + t * f2 / (4. * _MASS_OF_PROTON_IN_GEV**2)) * cff_et
    curly_C_longitudinally_polarized_interference = first_cff_contribution + second_cff_contribution - third_cff_contribution - fourth_cff_contribution
    return curly_C_longitudinally_polarized_interference

def i_curly_c_v_lp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_h: float,
    cff_e: float) -> float:
    t_over_Q_squared = t / q_sq
    ratio_of_xb_to_more_xb = xb / (2. - xb + xb * t_over_Q_squared)
    sum_of_form_factors = f1 + f2
    curly_C_V_longitudinally_polarized_interference = ratio_of_xb_to_more_xb * sum_of_form_factors * (cff_h + (xb * (1. - t_over_Q_squared) * cff_e / 2.))
    return curly_C_V_longitudinally_polarized_interference

def i_curly_c_a_lp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_ht: float,
    cff_et: float) -> float:
    t_over_Q_squared = t / q_sq
    ratio_of_xb_to_more_xb = xb / (2. - xb + xb * t_over_Q_squared)
    sum_of_form_factors = f1 + f2
    cff_appearance = cff_ht * (1. + (2. * xb * _MASS_OF_PROTON_IN_GEV**2 / q_sq)) + (xb * cff_et / 2.)
    curly_C_A_longitudinally_polarized_interference = ratio_of_xb_to_more_xb * sum_of_form_factors * cff_appearance
    return curly_C_A_longitudinally_polarized_interference

def i_unp_c0(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * np.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * np.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * np.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

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

    i_curly_c_eff = ktilde * np.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * np.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * np.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

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

    i_curly_c_eff = ktilde * np.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * np.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * np.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

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

    i_curly_c_eff = ktilde * np.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * np.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * np.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_c_pp_3 = i_c_unp_pp_3(q_sq, xb, t, ep, y, k)
    i_c_pp_v_3 = i_c_unp_v_pp_3(q_sq, xb, t, ep, y, k)
    i_c_pp_a_3 = i_c_unp_a_pp_3(q_sq, xb, t, ep, y, tprime, k)

    i_c_0p_3 = 0.
    i_c_0p_v_3 = 0.
    i_c_0p_a_3 = 0.

    return (i_c_pp_3*i_curly_c + i_c_pp_v_3*i_curly_c_v + i_c_pp_a_3*i_curly_c_a + i_c_0p_3*i_curly_c_eff + i_c_0p_v_3*i_curly_c_eff_v + i_c_0p_a_3*i_curly_c_eff_a)

def i_lp_c0(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e, cff_re_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_re_ht, cff_re_et)

    i_curly_c_eff = ktilde*np.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*np.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*np.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_c_pp_0 = i_c_lp_pp_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, ktilde)
    i_c_pp_v_0 = i_c_lp_v_pp_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, ktilde)
    i_c_pp_a_0 = i_c_lp_a_pp_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, ktilde)

    i_c_0p_0 = i_c_lp_0p_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_v_0 = i_c_lp_v_0p_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_a_0 = i_c_lp_a_0p_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    
    return (i_c_pp_0*i_curly_c + i_c_pp_v_0*i_curly_c_v + i_c_pp_a_0*i_curly_c_a + i_c_0p_0*i_curly_c_eff + i_c_0p_v_0*i_curly_c_eff_v + i_c_0p_a_0*i_curly_c_eff_a)

def i_lp_c1(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e, cff_re_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_re_ht, cff_re_et)

    i_curly_c_eff = ktilde*np.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*np.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*np.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_c_pp_1 = i_c_lp_pp_1(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_pp_v_1 = i_c_lp_v_pp_1(lep_helicity, target_polar, q_sq, xb, t, ep, y, tprime, k)
    i_c_pp_a_1 = i_c_lp_a_pp_1(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)

    i_c_0p_1 = i_c_lp_0p_1(lep_helicity, target_polar, q_sq, ep, y, ktilde, k)
    i_c_0p_v_1 = i_c_lp_v_0p_1(lep_helicity, target_polar, q_sq, t, ep, y, ktilde)
    i_c_0p_a_1 = 0.0
    
    return (i_c_pp_1*i_curly_c + i_c_pp_v_1*i_curly_c_v + i_c_pp_a_1*i_curly_c_a + i_c_0p_1*i_curly_c_eff + i_c_0p_v_1*i_curly_c_eff_v + i_c_0p_a_1*i_curly_c_eff_a)

def i_lp_c2(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e, cff_re_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_re_ht, cff_re_et)

    i_curly_c_eff = ktilde*np.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*np.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*np.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_c_pp_2 = i_c_lp_pp_2(lep_helicity, target_polar, q_sq, xb, t, ep, y)
    i_c_pp_v_2 = i_c_lp_v_pp_2(lep_helicity, target_polar, q_sq, xb, t, ep, y)
    i_c_pp_a_2 = i_c_lp_a_pp_2(lep_helicity, target_polar, q_sq, xb, t, ep, y)

    i_c_0p_2 = i_c_lp_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_v_2 = i_c_lp_v_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_a_2 = i_c_lp_a_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    
    return (i_c_pp_2*i_curly_c + i_c_pp_v_2*i_curly_c_v + i_c_pp_a_2*i_curly_c_a + i_c_0p_2*i_curly_c_eff + i_c_0p_v_2*i_curly_c_eff_v + i_c_0p_a_2*i_curly_c_eff_a)

def i_unp_s1(
    lep_helicity: float, q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_im_ht)

    i_curly_c_eff = ktilde * np.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * np.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * np.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_s_pp_1 = i_s_unp_pp_1(lep_helicity, q_sq, xb, ep, y, tprime, k)
    i_s_pp_v_1 = i_s_unp_v_pp_1(lep_helicity, q_sq, xb, t, ep, y, k)
    i_s_pp_a_1 = i_s_unp_a_pp_1(lep_helicity, q_sq, xb, t, ep, y, tprime, k)

    i_s_0p_1 = i_s_unp_0p_1(lep_helicity, q_sq, ep, y, ktilde)
    i_s_0p_v_1 = i_s_unp_v_0p_1(lep_helicity, q_sq, xb, t, ep, y)
    i_s_0p_a_1 = i_s_unp_a_0p_1(lep_helicity, q_sq, xb, t, ep, y, k)

    return (i_s_pp_1*i_curly_c + i_s_pp_v_1*i_curly_c_v + i_s_pp_a_1*i_curly_c_a + i_s_0p_1*i_curly_c_eff + i_s_0p_v_1*i_curly_c_eff_v + i_s_0p_a_1*i_curly_c_eff_a)

def i_unp_s2(
    lep_helicity: float, q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_im_ht)

    i_curly_c_eff = ktilde * np.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * np.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * np.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_s_pp_2 = i_s_unp_pp_2(lep_helicity, q_sq, xb, ep, y, tprime)
    i_s_pp_v_2 = i_s_unp_v_pp_2(lep_helicity, q_sq, xb, t, ep, y)
    i_s_pp_a_2 = i_s_unp_a_pp_2(lep_helicity, q_sq, xb, t, ep, y, tprime)

    i_s_0p_2 = i_s_unp_0p_2(lep_helicity, q_sq, xb, t, ep, y, k)
    i_s_0p_v_2 = i_s_unp_v_0p_2(lep_helicity, q_sq, xb, t, ep, y, k)
    i_s_0p_a_2 = i_s_unp_a_0p_2(lep_helicity, q_sq, xb, t, ep, y, k)

    return (i_s_pp_2*i_curly_c + i_s_pp_v_2*i_curly_c_v + i_s_pp_a_2*i_curly_c_a + i_s_0p_2*i_curly_c_eff + i_s_0p_v_2*i_curly_c_eff_v + i_s_0p_a_2*i_curly_c_eff_a)

def i_lp_s1(
    target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e, cff_im_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_im_ht, cff_im_et)

    i_curly_c_eff = ktilde*np.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*np.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*np.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_s_pp_1 = i_s_lp_pp_1(target_polar, q_sq, xb, t, ep, y, k)
    i_s_pp_v_1 = i_s_lp_v_pp_1(target_polar, q_sq, xb, t, ep, y, tprime, k)
    i_s_pp_a_1 = i_s_lp_a_pp_1(target_polar, q_sq, xb, t, ep, y, k)

    i_s_0p_1 = i_s_lp_0p_1(target_polar, q_sq, xb, t, ep, y, ktilde)
    i_s_0p_v_1 = i_s_lp_v_0p_1(target_polar, q_sq, xb, t, ep, y, ktilde)
    i_s_0p_a_1 = i_s_lp_a_0p_1(target_polar, q_sq, xb, t, ep, y)
    
    return (i_s_pp_1*i_curly_c + i_s_pp_v_1*i_curly_c_v + i_s_pp_a_1*i_curly_c_a + i_s_0p_1*i_curly_c_eff + i_s_0p_v_1*i_curly_c_eff_v + i_s_0p_a_1*i_curly_c_eff_a)

def i_lp_s2(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e, cff_im_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_im_ht, cff_im_et)

    i_curly_c_eff = ktilde*np.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*np.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*np.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_s_pp_2 = i_s_lp_pp_2(target_polar, q_sq, xb, t, ep, y, tprime, ktilde)
    i_s_pp_v_2 = i_s_lp_v_pp_2(target_polar, q_sq, xb, t, ep, y, tprime, ktilde)
    i_s_pp_a_2 = i_s_lp_a_pp_2(target_polar, q_sq, xb, t, ep, y, tprime, ktilde)

    i_s_0p_2 = i_s_lp_0p_2(target_polar, q_sq, xb, t, ep, y, k)
    i_s_0p_v_2 = i_s_lp_v_0p_2(target_polar, q_sq, xb, t, ep, y, k)
    i_s_0p_a_2 = i_s_lp_a_0p_2(target_polar, q_sq, xb, t, ep, y, k)
    
    return (i_s_pp_2*i_curly_c + i_s_pp_v_2*i_curly_c_v + i_s_pp_a_2*i_curly_c_a + i_s_0p_2*i_curly_c_eff + i_s_0p_v_2*i_curly_c_eff_v + i_s_0p_a_2*i_curly_c_eff_a)

def i_lp_s3(
    target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e, cff_im_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_im_ht, cff_im_et)

    i_curly_c_eff = ktilde*np.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*np.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * np.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*np.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * np.sqrt(q_sq))

    i_s_pp_3 = i_s_lp_pp_3(target_polar, q_sq, xb, ep, y, tprime, k)
    i_s_pp_v_3 = i_s_lp_v_pp_3(target_polar, q_sq, xb, t, ep, y, tprime, k)
    i_s_pp_a_3 = i_s_lp_a_pp_3(target_polar, q_sq, xb, t, ep, y, tprime, k)

    i_s_0p_3 = 0.0
    i_s_0p_v_3 = 0.0
    i_s_0p_a_3 = 0.0
    
    return (i_s_pp_3*i_curly_c + i_s_pp_v_3*i_curly_c_v + i_s_pp_a_3*i_curly_c_a + i_s_0p_3*i_curly_c_eff + i_s_0p_v_3*i_curly_c_eff_v + i_s_0p_a_3*i_curly_c_eff_a)

def interference_amplitude(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    i_c0_unp = i_unp_c0(q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_c1_unp = i_unp_c1(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_c2_unp = i_unp_c2(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_c3_unp = i_unp_c3(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_s1_unp = i_unp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    i_s2_unp = i_unp_s2(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    i_s3_unp = 0.0

    i_c0_lp = i_lp_c0(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
    i_c1_lp = i_lp_c1(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
    i_c2_lp = i_lp_c2(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
    i_c3_lp = 0.5 * 0.0
    i_s1_lp = i_lp_s1(target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    i_s2_lp = i_lp_s2(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    i_s3_lp = i_lp_s3(target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    return (
        (
            (i_c0_unp + i_c0_lp) * np.cos(0. * (tf.constant(np.pi) - phi)) +
            (i_c1_unp + i_c1_lp) * np.cos(1. * (tf.constant(np.pi) - phi)) + 
            (i_c2_unp + i_c2_lp) * np.cos(2. * (tf.constant(np.pi) - phi)) + 
            (i_c3_unp + i_c3_lp) * np.cos(3. * (tf.constant(np.pi) - phi)) + 
            (i_s1_unp + i_s1_lp) * np.sin(1. * (tf.constant(np.pi) - phi)) + 
            (i_s2_unp + i_s2_lp) * np.sin(2. * (tf.constant(np.pi) - phi)) + 
            (i_s3_unp + i_s3_lp) * np.sin(3. * (tf.constant(np.pi) - phi))
        )/(xb * y * y * y * t * p1 * p2))

def bkm10_cross_section(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):
    
    bh_plus_beam_plus_target = bh_squared(+1.0, +0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)
    bh_plus_beam_minus_target = bh_squared(+1.0, -0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)
    bh_minus_beam_plus_target = bh_squared(-1.0, +0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)
    bh_minus_beam_minus_target = bh_squared(-1.0, -0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)

    dvcs_plus_beam_plus_target = dvcs_squared(
        +1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_plus_beam_minus_target = dvcs_squared(
        +1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_minus_beam_plus_target = dvcs_squared(
        -1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_minus_beam_minus_target = dvcs_squared(
        -1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    interference_plus_beam_plus_target = interference_amplitude(
        +1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    interference_plus_beam_minus_target = interference_amplitude(
        +1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    interference_minus_beam_plus_target = interference_amplitude(
        -1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    interference_minus_beam_minus_target = interference_amplitude(
        -1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    tf_cross_section = 0.0
    
    if lep_helicity == 0.0:
        if target_polar == 0.0:
            tf_cross_section = 0.25 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + bh_plus_beam_minus_target + bh_minus_beam_plus_target + bh_minus_beam_minus_target +
                    dvcs_plus_beam_plus_target + dvcs_plus_beam_minus_target + dvcs_minus_beam_plus_target + dvcs_minus_beam_minus_target +
                    interference_plus_beam_plus_target + interference_plus_beam_minus_target + interference_minus_beam_plus_target + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
        
        elif target_polar == +0.5:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + 0.0 + bh_minus_beam_plus_target + 0.0 +
                    dvcs_plus_beam_plus_target + 0.0 + dvcs_minus_beam_plus_target + 0.0 +
                    interference_plus_beam_plus_target + 0.0 + interference_minus_beam_plus_target + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
        
        elif target_polar == -0.5:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + bh_plus_beam_minus_target + 0.0 + bh_minus_beam_minus_target +
                    0.0 + dvcs_plus_beam_minus_target + 0.0 + dvcs_minus_beam_minus_target +
                    0.0 + interference_plus_beam_minus_target + 0.0 + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))

    elif lep_helicity == +1.0:
        if target_polar == 0.0:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + bh_plus_beam_minus_target + 0.0 + 0.0 +
                    dvcs_plus_beam_plus_target + dvcs_plus_beam_minus_target + 0.0 + 0.0 +
                    interference_plus_beam_plus_target + interference_plus_beam_minus_target + 0.0 + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
        
        elif target_polar == +0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + 0.0 + 0.0 + 0.0 +
                    dvcs_plus_beam_plus_target + 0.0 + 0.0 + 0.0 +
                    interference_plus_beam_plus_target + 0.0 + 0.0 + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
        
        elif target_polar == -0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + bh_plus_beam_minus_target + 0.0 + 0.0 +
                    0.0 + dvcs_plus_beam_minus_target + 0.0 + 0.0 +
                    0.0 + interference_plus_beam_minus_target + 0.0 + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
    
    elif lep_helicity == -1.0:
        if target_polar == 0.0:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + 0.0 + bh_minus_beam_plus_target + bh_minus_beam_minus_target +
                    0.0 + 0.0 + dvcs_minus_beam_plus_target + dvcs_minus_beam_minus_target +
                    0.0 + 0.0 + interference_minus_beam_plus_target + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
        elif target_polar == +0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + 0.0 + bh_minus_beam_plus_target + 0.0 +
                    0.0 + 0.0 + dvcs_minus_beam_plus_target + 0.0 +
                    0.0 + 0.0 + interference_minus_beam_plus_target + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
        elif target_polar == -0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + 0.0 + 0.0 + bh_minus_beam_minus_target +
                    0.0 + 0.0 + 0.0 + dvcs_minus_beam_minus_target +
                    0.0 + 0.0 + 0.0 + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*np.sqrt(1. + ep**2)))
            
    return tf_cross_section

def bkm10_bsa(
    target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    plus_beam_cross_section = bkm10_cross_section(
        +1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_cross_section = bkm10_cross_section(
        -1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    
    tf_bsa = (
        (plus_beam_cross_section - minus_beam_cross_section) / 
        (plus_beam_cross_section + minus_beam_cross_section))
        
    return tf_bsa

def bkm10_tsa(
    lep_helicity, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    plus_beam_plus_lp_target = bkm10_cross_section(
    +1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_plus_lp_target = bkm10_cross_section(
    -1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    plus_beam_minus_lp_target = bkm10_cross_section(
    +1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_minus_lp_target = bkm10_cross_section(
    -1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    if lep_helicity == 0.0:

        sigma_plus = 0.5 * (plus_beam_plus_lp_target + minus_beam_plus_lp_target)
        sigma_minus = 0.5 * (plus_beam_minus_lp_target + minus_beam_minus_lp_target)

    if lep_helicity == 1.0:

        sigma_plus = plus_beam_plus_lp_target
        sigma_minus = plus_beam_minus_lp_target

    if lep_helicity == -1.0:

        sigma_plus = minus_beam_plus_lp_target
        sigma_minus = minus_beam_minus_lp_target

    numerator = sigma_plus - sigma_minus
    denominator = sigma_plus + sigma_minus
    tsa = numerator / denominator
    return tsa

def bkm10_dsa(
    q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    plus_beam_plus_lp_target = bkm10_cross_section(
    +1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_plus_lp_target = bkm10_cross_section(
    -1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    plus_beam_minus_lp_target = bkm10_cross_section(
    +1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_minus_lp_target = bkm10_cross_section(
    -1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    numerator = ((plus_beam_plus_lp_target-plus_beam_minus_lp_target)-(minus_beam_plus_lp_target-minus_beam_minus_lp_target))
    denominator = plus_beam_plus_lp_target+plus_beam_minus_lp_target+minus_beam_plus_lp_target+minus_beam_minus_lp_target
    dsa = numerator / denominator
    return dsa

##########################################
# THE MAIN LOOP!
##########################################

rows = []
kinematic_set_number = 0 
total_kinematic_settings = NUMBER_OF_BEAM_K * NUMBER_OF_Q_SQUARED * NUMBER_OF_X_B * NUMBER_OF_T
total_settings = NUMBER_OF_BEAM_K * NUMBER_OF_Q_SQUARED * NUMBER_OF_X_B * NUMBER_OF_T * NUMBER_OF_PHI_POINTS

print(f"[INFO]: Ready to iterate over {total_kinematic_settings} kinematic settings.")
print(f"[INFO]: A total of {total_settings} combinations points.")

for fixed_k in K_RANGE:
    for fixed_q_squared in Q2_RANGE:
        for fixed_x_bjorken in X_B_RANGE:
            for fixed_t in T_RANGE:

                print(f"[INFO]: Iteration with bin: k = {fixed_k}, Q^2 = {fixed_q_squared}, xb = {fixed_x_bjorken}, t = {fixed_t}")

                # increment set counter
                kinematic_set_number += 1

                ##########################################
                # Making required directories
                ##########################################

                os.makedirs(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data", exist_ok = True)
                print(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data")

                os.makedirs(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/plots", exist_ok = True)
                print(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/plots")
                
                os.makedirs(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/learning_curves", exist_ok = True)
                print(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/learning_curves")
                
                os.makedirs(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/replicas", exist_ok = True)
                print(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/replicas")

                # using gepard's DataPoint with phi varying:
                current_kinematic_setting = [g.DataPoint(
                    xB = fixed_x_bjorken, t = fixed_t, Q2 = fixed_q_squared, phi = fixed_phi,
                    process = "ep2epgamma", exptype = 'fixed target',
                    in1energy = fixed_k, in1charge = -1, in1polarization = +1,
                    observable = 'XS',
                    fname = 'Trento') for fixed_phi in phi_array_in_radians]

                assert len(current_kinematic_setting) == len(phi_array_in_radians), "[ASSERT]: Datapoint array has wrong size."

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
                fixed_kdd = compute_k_dot_delta(fixed_q_squared, fixed_x_bjorken, fixed_t, phi_array_in_radians, fixed_ep, fixed_y, fixed_big_k)
                fixed_p1 = prop_1(fixed_q_squared, fixed_kdd)
                fixed_p2 = prop_2(fixed_q_squared, fixed_t, fixed_kdd)

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
                
                tf_cross_section_km15 = bkm10_cross_section(
                    0.0, 0.0,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])
                
                ##########################################
                # d^{4}\sigma(lambda = +1, Lambda = 0)
                ##########################################
                    
                bkm10_plus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
                    phi_array_in_radians,
                    lepton_helicity = +1.0,
                    target_polarization = 0.0).real
                
                tf_plus_beam_unp_target_km15 = bkm10_cross_section(
                    +1.0, 0.0,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])
                    
                ##########################################
                # d^{4}\sigma(lambda = -1, Lambda = 0)
                ##########################################

                bkm10_minus_beam_unp_target_km15 = km15_cross_section.compute_cross_section(
                    phi_array_in_radians,
                    lepton_helicity = -1.0,
                    target_polarization = 0.0).real
                
                tf_minus_beam_unp_target_km15 = bkm10_cross_section(
                    -1.0, 0.0,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])
                    
                ##########################################
                # d^{4}\sigma(lambda = 0, Lambda = 1/2)
                ##########################################

                bkm10_unp_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
                    phi_array_in_radians,
                    lepton_helicity = 0.0,
                    target_polarization = +0.5).real

                tf_lp_target_km15 = bkm10_cross_section(
                    0.0, +0.5,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])
                    
                ##########################################
                # d^{4}\sigma(lambda = +1, Lambda = 1/2)
                ##########################################
                    
                bkm10_plus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
                    phi_array_in_radians,
                    lepton_helicity = +1.0,
                    target_polarization = +0.5).real

                tf_plus_beam_lp_target_km15 = bkm10_cross_section(
                    +1.0, +0.5,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])
                    
                ##########################################
                # d^{4}\sigma(lambda = -1, Lambda = 1/2)
                ##########################################

                bkm10_minus_beam_lp_target_km15 = km15_cross_section.compute_cross_section(
                    phi_array_in_radians,
                    lepton_helicity = -1.0,
                    target_polarization = +0.5).real
                
                tf_minus_beam_lp_target_km15 = bkm10_cross_section(
                    -1.0, +0.5,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                ##########################################
                # BSA(Lambda = 0)
                ##########################################
                    
                bkm10_bsa_km15 = km15_cross_section.compute_bsa(
                    phi_array_in_radians, 
                    target_polarization = 0.0).real

                tf_unp_bsa_km15 = bkm10_bsa(
                    0.0,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                ##########################################
                # BSA(Lambda = 1/2)
                ##########################################
                    
                bkm10_plus_lp_bsa_km15 = km15_cross_section.compute_bsa(
                    phi_array_in_radians,
                    target_polarization = +0.5).real
                
                tf_plus_lp_bsa_km15 = bkm10_bsa(
                    +0.5,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                ##########################################
                # BSA(Lambda = -1/2)
                ##########################################
                    
                bkm10_minus_lp_bsa_km15 = km15_cross_section.compute_bsa(
                    phi_array_in_radians,
                    target_polarization = -0.5).real
                
                tf_minus_lp_bsa_km15 = bkm10_bsa(
                    -0.5,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                # ##########################################
                # # TSA(lambda = 0.0)
                # ##########################################

                bkm10_tsa_km15 = km15_cross_section.compute_tsa(
                    phi_array_in_radians,
                    lepton_polarization = 0.0).real

                tf_tsa_km15 = bkm10_tsa(
                    0.0,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                # ##########################################
                # # TSA(lambda = +1.0)
                # ##########################################

                bkm10_plus_beam_tsa_km15 = km15_cross_section.compute_tsa(
                    phi_array_in_radians,
                    lepton_polarization = +1.0).real
                
                tf_plus_tsa_km15 = bkm10_tsa(
                    +1.0,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                # ##########################################
                # # TSA(lambda = -1.0)
                # ##########################################

                bkm10_minus_beam_tsa_km15 = km15_cross_section.compute_tsa(
                    phi_array_in_radians,
                    lepton_polarization = -1.0).real

                tf_minus_tsa_km15 = bkm10_tsa(
                    -1.0,
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                # ##########################################
                # # DSA
                # ##########################################

                bkm10_dsa_km15 = km15_cross_section.compute_dsa(
                    phi_array_in_radians).real
                
                tf_dsa_km15 = bkm10_dsa(
                    fixed_q_squared, fixed_x_bjorken, fixed_t, fixed_ep, fixed_y,
                    fixed_xi, fixed_big_k, fixed_f1, fixed_f2, fixed_k_tilde, fixed_t_prime, phi_array_in_radians,
                    fixed_p1, fixed_p2,
                    real_h_values[0], real_ht_values[0], real_e_values[0], real_et_values[0],
                    imag_h_values[0], imag_ht_values[0], imag_e_values[0], imag_et_values[0])

                if not np.all(np.isfinite(bkm10_unp_beam_unp_target_km15)):
                    print("[ERROR]: KM15 observables are unphysical: not finite")
                    continue

                if np.any((bkm10_unp_beam_unp_target_km15 < 0.)):
                    print("[ERROR]: KM15 observables are unphysical: cross-sections are negative")
                    continue
                    
                if np.any(np.isnan(bkm10_unp_beam_unp_target_km15)):
                    print("[ERROR]: KM15 observables are unphysical: encountered NaN")
                    continue

                ##########################################
                # Computed strings for plots:
                ##########################################

                title_string = (
                    rf"$Q^2 = {fixed_q_squared:.2f}$ GeV$^2$, "
                    rf"$x_B = {fixed_x_bjorken:.2f}$, "
                    rf"$t = {fixed_t:.2f}$, "
                    rf"$k = {fixed_k:.2f}$ GeV"
                )

                km15_cff_string = (
                    rf"$\mathcal{{H}} = {CFF_H_KM15:.3f}$, "
                    rf"$\mathcal{{E}} = {CFF_E_KM15:.3f}$, "
                    rf"$\widetilde{{\mathcal{{H}}}} = {CFF_H_TILDE_KM15:.3f}$, "
                    rf"$\widetilde{{\mathcal{{E}}}} = {CFF_E_TILDE_KM15:.3f}$ "
                )

                ##########################################
                # Plotting Observable Values
                # d^{4}\sigma(lambda = 0, Lambda = 0)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_cross_section_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_unp_beam_unp_target_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_unp_beam_unp_target_km15 - tf_cross_section_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"$d^{4}\sigma^{UU}$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_beam_unp_target_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_beam_unp_target_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # d^{4}\sigma(lambda = +1, Lambda = 0)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_plus_beam_unp_target_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_plus_beam_unp_target_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_plus_beam_unp_target_km15 - tf_plus_beam_unp_target_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"$d^{4}\sigma^{+U}$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_beam_unp_target_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_beam_unp_target_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # d^{4}\sigma(lambda = -1, Lambda = 0)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_minus_beam_unp_target_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_minus_beam_unp_target_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_minus_beam_unp_target_km15 - tf_minus_beam_unp_target_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"$d^{4}\sigma^{-U}$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_beam_unp_target_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_beam_unp_target_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # d^{4}\sigma(lambda = 0, Lambda = +1/2)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_lp_target_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_unp_beam_lp_target_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_unp_beam_lp_target_km15 - tf_lp_target_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"$d^{4}\sigma^{UL}$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_beam_lp_target_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_beam_lp_target_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # d^{4}\sigma(lambda = +1, Lambda = +1/2)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_plus_beam_lp_target_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_plus_beam_lp_target_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_plus_beam_lp_target_km15 - tf_plus_beam_lp_target_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"$d^{4}\sigma^{+L}$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_beam_lp_target_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_beam_lp_target_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)
                
                ##########################################
                # Plotting Observable Values
                # d^{4}\sigma(lambda = -1, Lambda = +1/2)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_minus_beam_lp_target_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_minus_beam_lp_target_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_minus_beam_lp_target_km15 - tf_minus_beam_lp_target_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"$d^{4}\sigma^{-L}$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_beam_lp_target_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_beam_lp_target_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # BSA(Lambda = 0)
                ##########################################

                bsa_fig, bsa_axes = plt.subplots(1, 2, figsize = (14, 7))
                bsa_axes[0].scatter(
                    phi_array_in_radians, tf_unp_bsa_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                bsa_axes[0].scatter(
                    phi_array_in_radians, bkm10_bsa_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                bsa_axes[1].scatter(
                    phi_array_in_radians, (bkm10_bsa_km15 - tf_unp_bsa_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                bsa_axes[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                bsa_axes[0].set_ylabel(r"BSA$\left( \Lambda = 0 \right)$", fontsize = 14)
                bsa_axes[1].set_ylabel(r"Residuals", fontsize = 14)
                bsa_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                bsa_axes[0].legend(fontsize = 14.3)
                bsa_axes[1].legend(fontsize = 14.3)
                bsa_axes[0].grid(visible = True)
                bsa_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_bsa_check_v{MAJOR_MINOR_NUMBER}.png")
                bsa_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_bsa_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(bsa_fig)

                ##########################################
                # Plotting Observable Values
                # BSA(Lambda = +1/2)
                ##########################################

                bsa_fig, bsa_axes = plt.subplots(1, 2, figsize = (14, 7))
                bsa_axes[0].scatter(
                    phi_array_in_radians, tf_plus_lp_bsa_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                bsa_axes[0].scatter(
                    phi_array_in_radians, bkm10_plus_lp_bsa_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                bsa_axes[1].scatter(
                    phi_array_in_radians, (bkm10_plus_lp_bsa_km15 - tf_plus_lp_bsa_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                bsa_axes[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                bsa_axes[0].set_ylabel(r"BSA$\left( \Lambda = +1/2 \right)$", fontsize = 14)
                bsa_axes[1].set_ylabel(r"Residuals", fontsize = 14)
                bsa_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                bsa_axes[0].legend(fontsize = 14.3)
                bsa_axes[1].legend(fontsize = 14.3)
                bsa_axes[0].grid(visible = True)
                bsa_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_lp_bsa_check_v{MAJOR_MINOR_NUMBER}.png")
                bsa_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_lp_bsa_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(bsa_fig)

                ##########################################
                # Plotting Observable Values
                # BSA(Lambda = -1/2)
                ##########################################

                bsa_fig, bsa_axes = plt.subplots(1, 2, figsize = (14, 7))
                bsa_axes[0].scatter(
                    phi_array_in_radians, tf_minus_lp_bsa_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                bsa_axes[0].scatter(
                    phi_array_in_radians, bkm10_minus_lp_bsa_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                bsa_axes[1].scatter(
                    phi_array_in_radians, (bkm10_minus_lp_bsa_km15 - tf_minus_lp_bsa_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                bsa_axes[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                bsa_axes[0].set_ylabel(r"BSA$\left( \Lambda = -1/2 \right)$", fontsize = 14)
                bsa_axes[1].set_ylabel(r"Residuals", fontsize = 14)
                bsa_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                bsa_axes[0].legend(fontsize = 14.3)
                bsa_axes[1].legend(fontsize = 14.3)
                bsa_axes[0].grid(visible = True)
                bsa_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_lp_bsa_check_v{MAJOR_MINOR_NUMBER}.png")
                bsa_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_lp_bsa_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(bsa_fig)

                ##########################################
                # Plotting Observable Values
                # TSA(lambda = 0.0)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_tsa_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_tsa_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_tsa_km15 - tf_tsa_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"TSA$\left( \lambda = 0 \right)$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_tsa_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/unp_tsa_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # TSA(lambda = +1.0)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_plus_tsa_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_plus_beam_tsa_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_plus_beam_tsa_km15 - tf_plus_tsa_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"TSA$\left( \lambda = +1 \right)$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_beam_tsa_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/plus_beam_tsa_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # TSA(lambda = -1.0)
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_minus_tsa_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_minus_beam_tsa_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_minus_beam_tsa_km15 - tf_minus_tsa_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"TSA$\left( \lambda = -1 \right)$", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_beam_tsa_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/minus_beam_tsa_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

                ##########################################
                # Plotting Observable Values
                # DSA
                ##########################################

                xsec_test_fig, xsec_test_ax = plt.subplots(1, 2, figsize = (14, 7))
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, tf_dsa_km15, 
                    color = "red", s = 2., label = "TF(KM15) Predictions")
                xsec_test_ax[0].scatter(
                    phi_array_in_radians, bkm10_dsa_km15, 
                    color = "purple", s = 2., label = "BKM10(KM15) Predictions")
                xsec_test_ax[1].scatter(
                    phi_array_in_radians, (bkm10_dsa_km15 - tf_dsa_km15),
                    color = "gray", s = 2., label = "Residuals: BKM10(KM15) - TF(KM15)")
                xsec_test_ax[0].set_xlabel(r"$\phi$ [radians]", fontsize = 14)
                xsec_test_ax[0].set_ylabel(r"DSA", fontsize = 14)
                xsec_test_ax[1].set_ylabel(r"Residuals", fontsize = 14)
                xsec_test_fig.suptitle(f"{title_string}\n(KM15): {km15_cff_string}")
                xsec_test_ax[0].legend(fontsize = 14.3)
                xsec_test_ax[1].legend(fontsize = 14.3)
                xsec_test_ax[0].grid(visible = True)
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/dsa_check_v{MAJOR_MINOR_NUMBER}.png")
                xsec_test_fig.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/dsa_check_v{MAJOR_MINOR_NUMBER}.eps")
                plt.close(xsec_test_fig)

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
                    path_or_buf = f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/data/main_pseudodata_file_v{MAJOR_MINOR_NUMBER}.csv"
                )

                del this_kinematic_set_dataframe

                # clean up:
                del km15_cross_section
                del current_kinematic_setting
                
print(f"[INFO]: Has iteration successfully executed: {kinematic_set_number == total_kinematic_settings}")
print(f"[INFO]: New dataframe has {len(rows)} total rows.")

kinematic_grid_data = pd.DataFrame(rows)
kinematic_grid_data.to_csv(f"kinematic_grid_data_v{MAJOR_MINOR_NUMBER}.csv")

print(f"[INFO]: Saved {len(kinematic_grid_data)} rows.")
print(f"[INFO]: Attempted {total_settings} total kinematic settings.")
print(f"[INFO]: Angles per setting: {NUMBER_OF_PHI_POINTS}")

try:
    with open(file = f"kinematic_grid_settings_v{MAJOR_MINOR_NUMBER}.txt", mode = "w") as new_file:
        new_file.write(f"Total number of proposed (initial) grid points: {total_kinematic_settings}")
        new_file.write(f"Each point has {NUMBER_OF_PHI_POINTS} phi values, which makes the grid's total size {total_settings}")
        new_file.write(f"Saved {len(kinematic_grid_data)} rows in DataFrame called kinematic_grid_data_v{MAJOR_MINOR_NUMBER}.csv")
    new_file.close()
except:
    raise Exception("[ERROR]: Unable to write file!")

##########################################
# Extracting columns with kinematic values:
##########################################

unique_kinematic_sets_dataframe = kinematic_grid_data.groupby(['set'], as_index = False).mean()
number_of_unique_kinematic_sets = len(unique_kinematic_sets_dataframe)

print(f"[INFO]: Total number of unique sets: {number_of_unique_kinematic_sets}")
# if the assert below is False, it means some settings were unphysical --- that's all!
print(f"[ASSERT]: Does it match the expected number? {number_of_unique_kinematic_sets == total_kinematic_settings}")

q_squared_column = unique_kinematic_sets_dataframe["q_squared"]
xb_column = unique_kinematic_sets_dataframe['x_b']
t_column = unique_kinematic_sets_dataframe["t"]

input_space_figure = plt.figure(figsize = (9, 7))
input_space_axis = input_space_figure.add_subplot(1, 1, 1, projection = "3d")
input_space_axis.scatter(xb_column, q_squared_column, t_column, alpha = 0.3, s = 6.4)
input_space_axis.set_xlabel(r"$x_{\textrm{B}}$")
input_space_axis.set_ylabel(r"$Q^{2}$")
input_space_axis.set_zlabel(r"$-t$")
input_space_axis.set_title("Kinematic Settings Space")

for kinematic_set in range(kinematic_set_number):
    input_space_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/plots/kinematic_settings_space_v{MAJOR_MINOR_NUMBER}.png")
    input_space_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/plots/kinematic_settings_space_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(input_space_figure)

t_vs_xb_figure, t_vs_x_axis = plt.subplots(1, 1, figsize = (8, 7))
t_vs_x_axis.scatter(xb_column, t_column, alpha = 0.3, s = 10.4)
t_vs_x_axis.set_xlabel(r"$x_{\textrm{B}}$")
t_vs_x_axis.set_ylabel(r"$-t$")
t_vs_x_axis.set_title(r"$-t$ vs. $x_{\textrm{B}}$")

for kinematic_set in range(kinematic_set_number):
    t_vs_xb_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/plots/t_vs_xb_v{MAJOR_MINOR_NUMBER}.png")
    t_vs_xb_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{MAJOR_MINOR_NUMBER}/plots/t_vs_xb_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(t_vs_xb_figure)

qsq_vs_t_figure, qsq_vs_t_axis = plt.subplots(1, 1, figsize = (8, 7))
qsq_vs_t_axis.scatter(q_squared_column, t_column, alpha = 0.1, s = 10.4)
qsq_vs_t_axis.set_xlabel(r"$Q^{2}$")
qsq_vs_t_axis.set_ylabel(r"$-t$")
qsq_vs_t_axis.set_title(r"$-t$ vs. $Q^{2}$")

for kinematic_set in range(kinematic_set_number):
    qsq_vs_t_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{VERSION_NUMBER}/plots/qsquared_vs_t_v{MAJOR_MINOR_NUMBER}.png")
    qsq_vs_t_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{VERSION_NUMBER}/plots/qsquared_vs_t_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(qsq_vs_t_figure)

xb_vs_qsq_figure, xb_vs_qsq_axis = plt.subplots(1, 1, figsize = (8, 7))
xb_vs_qsq_axis.scatter(xb_column, q_squared_column, alpha = 0.1, s = 10.4)
xb_vs_qsq_axis.set_xlabel(r"$x_{\textrm{B}}$")
xb_vs_qsq_axis.set_ylabel(r"$Q^{2}$")
xb_vs_qsq_axis.set_title(r"$Q^{2}$ vs. $x_{\textrm{B}}$")

for kinematic_set in range(kinematic_set_number):
    xb_vs_qsq_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{VERSION_NUMBER}/plots/xb_vs_qsquared_v{MAJOR_MINOR_NUMBER}.png")
    xb_vs_qsq_figure.savefig(f"{SCRATCH_PATH}/kinematic_set_{kinematic_set_number}/version_{VERSION_NUMBER}/plots/xb_vs_qsquared_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(xb_vs_qsq_figure)

print(f"[INFO]: End of script reached!")
