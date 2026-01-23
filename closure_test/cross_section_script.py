"""
Stuff.
"""

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
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

###########################
# CHECK THESE VALUES
###########################

STARTING_PHI_VALUE_IN_DEGREES = 0
ENDING_PHI_VALUE_IN_DEGREES = 360
NUMBER_OF_PHI_POINTS = 360

phi_array_in_degrees = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
    num = NUMBER_OF_PHI_POINTS)
phi_array_in_radians = np.array([np.radians(degree_value) for degree_value in phi_array_in_degrees])

print(f"We have constructed a Python list of length {len(phi_array_in_radians)} of azimuthal angles from {STARTING_PHI_VALUE_IN_DEGREES} degrees to {ENDING_PHI_VALUE_IN_DEGREES} degrees.")

TEST_BEAM_ENERGY = 5.75
TEST_QSQ = 1.82
TEST_XB = 0.34
TEST_T = -.17

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

# these are test values ONLY! (i.e. not KM15 values)
TEST_REAL_CFF_H = -0.897
TEST_REAL_CFF_HT = 2.444
TEST_REAL_CFF_E = -0.541
TEST_REAL_CFF_ET = 2.207 
TEST_IM_CFF_H = 2.421
TEST_IM_CFF_HT = 1.131
TEST_IM_CFF_E = 0.903
TEST_IM_CFF_ET = 5.383

# WW-relations are on!!
TEST_REAL_CFF_H_EFF = -1.4961696451186222
TEST_REAL_CFF_HT_EFF = 4.0765201924971155
TEST_REAL_CFF_E_EFF = -0.9023721048039851
TEST_REAL_CFF_ET_EFF = 3.6812111558269773
TEST_IM_CFF_H_EFF = 4.038156868263304
TEST_IM_CFF_HT_EFF = 1.8864747699321758
TEST_IM_CFF_E_EFF = 1.5061774688317902
TEST_IM_CFF_ET_EFF = 8.978685841330593

_MASS_OF_PROTON_IN_GEV = 0.93827208816
_ELECTRIC_FORM_FACTOR_CONSTANT = 0.710649
_PROTON_MAGNETIC_MOMENT = 2.79284734463
_CONVERSION_FACTOR = .389379 * 1000000.
_QED_FINE_STRUCTURE = 1./137.035999177

###########################
# ALL THE FUNCTIONS
###########################

def compute_fe(t):
    return np.divide(1., (1. - np.divide(t, _ELECTRIC_FORM_FACTOR_CONSTANT))**2)

def compute_fg(fe):
    return _PROTON_MAGNETIC_MOMENT * fe

def compute_f2(t, fe, fg):
    tau = np.divide(-1. * t, 4. * _MASS_OF_PROTON_IN_GEV**2)
    numerator = fg - fe
    denominator = 1. + tau
    return np.divide(numerator, denominator)

def compute_f1(fg, f2):
    return fg - f2

def compute_epsilon(xb, q_squared):
    return np.divide(2. * xb * _MASS_OF_PROTON_IN_GEV, np.sqrt(q_squared))

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
    return (-1.*q_squared / (2.*y_lep*(1.+ep**2))) * (1. + ((2.*k*np.cos(np.pi- phi_azi)) - ((t / q_squared)*(1.-(xb * (2. - y_lep)) + (y_lep * ep**2 / 2.))) + (y_lep * ep**2 / 2.)))

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
    if target_polar == -0.5 or target_polar == 0.5:
        bh_c0 = bh_unp_c0(q_sq, xb, t, ep, y, k, f1, f2)
        bh_c1 = bh_unp_c1(q_sq, xb, t, ep, y, k, f1, f2)
        bh_c2 = 0.0
    
    return ((
        bh_c0 + 
        bh_c1 * np.cos(1.* (np.pi - phi)) + 
        bh_c2 * np.cos(2.* (np.pi - phi))) / (xb * xb * y * y * (1.+ep**2)**2 * t * p1 * p2))

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
    third_line=-4.*((1.-xb)*(q_sq+xb*t)*t)+(q_sq+t)**2*ep*ep*xb*(cff_re_h*cff_re_et_star-cff_im_et*cff_im_h_star+cff_re_et*cff_re_h_star-cff_im_h*cff_im_et_star)/(2.*q_sq*(q_sq+xb*t))
    fourth_line=-(2.-xb)*q_sq+xb*t*((xb*xb*(q_sq*t)**2)/(2.*q_sq*((2.-xb)*q_sq+xb*t))+(t/(4.*_MASS_OF_PROTON_IN_GEV**2)))*(cff_re_e*cff_re_et_star-cff_im_e*cff_im_et_star+cff_re_et*cff_re_e_star-cff_im_et*cff_im_e_star)*xb

    return ((first_line+second_line+third_line+fourth_line)*q_sq*(q_sq+xb*t)/(np.sqrt(1.+ep*ep)*((2.-xb)*q_sq+xb*t)**2))

def curly_c_imag_lp(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):

    first_line=(4.*(1.-xb+(((3.-2.*xb)*q_sq + t))*ep*ep/(4.*(q_sq+xb*t)))*(cff_im_h*cff_re_ht_star+cff_re_ht*cff_im_h_star+cff_im_ht*cff_re_h_star+cff_re_h*cff_im_ht_star))
    second_line=-xb*xb*(q_sq-xb*t*(1.-2.*xb))*(cff_im_h*cff_re_et_star+cff_re_et*cff_im_h_star+cff_im_et*cff_re_h_star+cff_re_h*cff_im_et_star+cff_im_ht*cff_re_e_star+cff_re_e*cff_im_ht_star+cff_im_e*cff_re_ht_star+cff_re_ht*cff_im_e_star)/(q_sq+xb*t)
    third_line=-4.*((1.-xb)*(q_sq+xb*t)*t)+(q_sq+t)**2*ep*ep*xb*(cff_im_h*cff_re_et_star+cff_re_et*cff_im_h_star+cff_im_et*cff_re_h_star+cff_re_h*cff_im_et_star)/(2.*q_sq*(q_sq+xb*t))
    fourth_line=-(2.-xb)*q_sq+xb*t*((xb*xb*(q_sq*t)**2)/(2.*q_sq*((2.-xb)*q_sq+xb*t))+(t/(4.*_MASS_OF_PROTON_IN_GEV**2)))*(cff_im_e*cff_re_et_star+cff_re_e*cff_im_et_star+cff_im_et*cff_re_e_star+cff_re_et*cff_im_e_star)*xb

    return ((first_line+second_line+third_line+fourth_line)*q_sq*(q_sq+xb*t)/(np.sqrt(1.+ep*ep)*((2.-xb)*q_sq+xb*t)**2))

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
    lep_helicity: float,q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
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

    prefactor = 2.*lep_helicity*target_polar*y**(2.-y)/np.sqrt(1.+ep*ep)
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

    if (target_polar == -0.5 or target_polar == +0.5):
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
         dvcs_c1 * np.cos(1.* (np.pi - phi)) + 
         dvcs_s1 * np.sin(1.* (np.pi - phi))) / (y * y * q_sq))

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
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_first_term = (1. + (1. - xb) * (root_one_plus_epsilon_squared - 1.) / (2. * xb) + ep**2 / (4. * xb)) * xb * t_over_Q_squared
    first_bracket_term = first_bracket_first_term - 3. * ep**2 / 4.
    second_bracket_term = 1. - (1. - 3. * xb) * t_over_Q_squared + (1. - root_one_plus_epsilon_squared + 3. * ep**2) * xb * t_over_Q_squared / (one_plus_root_epsilon_stuff - ep**2)
    fancy_y_coefficient = 2. - 2. * y + y**2 + ep**2 * y**2 / 2.
    second_term = -4. * shorthand_k * fancy_y_coefficient * (one_plus_root_epsilon_stuff - ep**2) * second_bracket_term / root_one_plus_epsilon_squared**5
    first_term = -16. * shorthand_k * (1. - y - ep**2 * y**2 / 4.) * first_bracket_term / root_one_plus_epsilon_squared**5
    c_1_plus_plus_unp = first_term + second_term
    return c_1_plus_plus_unp

def i_c_unp_v_pp_1(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    first_bracket_term = (2. - y)**2 * (1. - (1. - 2. * xb) * t_over_Q_squared)
    second_bracket_term_first_part = 1. - y - ep**2 * y**2 / 4.
    second_bracket_term_second_part = 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb) * t_prime / q_sq
    coefficient_prefactor = 16. * shorthand_k * xb * t_over_Q_squared / np.power(root_one_plus_epsilon_squared, 5)
    c_1_plus_plus_V_unp = coefficient_prefactor * (first_bracket_term + second_bracket_term_first_part * second_bracket_term_second_part)
    return c_1_plus_plus_V_unp

def i_c_unp_a_pp_1(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = t_prime / q_sq
    one_minus_xb = 1. - xb
    one_minus_2xb = 1. - 2. * xb
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    first_bracket_term_second_part = 1. - one_minus_2xb * t_over_Q_squared + (4. * xb * one_minus_xb + ep**2) * t_prime_over_Q_squared / (4. * root_one_plus_epsilon_squared)
    second_bracket_term = 1. - 0.5 * xb + 0.25 * (one_minus_2xb + root_one_plus_epsilon_squared) * (1. - t_over_Q_squared) + (4. * xb * one_minus_xb + ep**2) * t_prime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -16. * shorthand_k * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_1_plus_plus_A_unp = prefactor * (fancy_y_stuff * first_bracket_term_second_part - (2. - y)**2 * second_bracket_term)
    return c_1_plus_plus_A_unp

def i_c_unp_0p_1(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = t_prime / q_sq
    one_minus_xb = 1. - xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    first_bracket_term = (2. - y)**2 * t_prime_over_Q_squared * (one_minus_xb + (one_minus_xb * xb + (ep**2 / 4.)) * t_prime_over_Q_squared / root_one_plus_epsilon_squared)
    second_bracket_term = y_quantity * (1. - (1. - 2. * xb) * t_over_Q_squared) * (ep**2 - 2. * (1. + (ep**2 / (2. * xb))) * xb * t_over_Q_squared) / root_one_plus_epsilon_squared
    prefactor = 8. * np.sqrt(2. * y_quantity) / root_one_plus_epsilon_squared**4
    c_1_zero_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_1_zero_plus_unp

def i_c_unp_v_0p_1(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float):
    t_over_Q_squared = t / q_sq
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    major_part = (2 - y)**2 * k_tilde**2 / q_sq + (1. - (1. - 2. * xb) * t_over_Q_squared)**2 * y_quantity
    prefactor = 16. * np.sqrt(2. * y_quantity) * xb * t_over_Q_squared / (1. + ep**2)**2.5
    c_1_zero_plus_V_unp = prefactor * major_part
    return c_1_zero_plus_V_unp

def i_c_unp_a_0p_1(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_minus_2xb = 1. - 2. * xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    second_term_first_part = (1. - one_minus_2xb * t_over_Q_squared) * y_quantity
    second_term_second_part = 4. - 2. * xb + 3. * ep**2 + t_over_Q_squared * (4. * xb * (1. - xb) + ep**2)
    first_term = k_tilde**2 * one_minus_2xb * (2. - y)**2 / q_sq
    prefactor = 8. * np.sqrt(2. * y_quantity) * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_1_zero_plus_unp_A = prefactor * (first_term + second_term_first_part * second_term_second_part)
    return c_1_zero_plus_unp_A

def i_s_unp_pp_1(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    tPrime_over_Q_squared = t_prime / q_sq
    bracket_term = 1. + ((1. - xb + 0.5 * (root_one_plus_epsilon_squared - 1.)) / root_one_plus_epsilon_squared**2) * tPrime_over_Q_squared
    prefactor = 8. * lep_helicity * shorthand_k * y * (2. - y) / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp = prefactor * bracket_term
    return s_1_plus_plus_unp

def i_s_unp_v_pp_1(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    bracket_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb) * t_over_Q_squared
    prefactor = -8. * lep_helicity * shorthand_k * y * (2. - y) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_1_plus_plus_unp_V = prefactor * bracket_term
    return s_1_plus_plus_unp_V

def i_s_unp_a_pp_1(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    tPrime_over_Q_squared = t_prime / q_sq
    one_minus_2xb = 1. - 2. * xb
    bracket_term = 1. - one_minus_2xb * (one_minus_2xb + root_one_plus_epsilon_squared) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = 8. * lep_helicity * shorthand_k * y * (2. - y) * t_over_Q_squared / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp_A = prefactor * bracket_term
    return s_1_plus_plus_unp_A

def i_s_unp_0p_1(
    lep_helicity: float,
    q_sq: float, 
    ep: float,
    y: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = (1. + ep**2)**2
    y_quantity = np.sqrt(1. - y - (ep**2 * y**2 / 4.))
    s_1_zero_plus_unp = 8. * np.sqrt(2.) * lep_helicity * (2. - y) * y * y_quantity * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)
    return s_1_zero_plus_unp

def i_s_unp_v_0p_1(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float):
    one_plus_epsilon_squared_squared = (1. + ep**2)**2
    t_over_Q_squared = t / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    bracket_term = 4. * (1. - 2. * xb) * t_over_Q_squared * (1. + xb * t_over_Q_squared) + ep**2 * (1. + t_over_Q_squared)**2
    prefactor = 4. * np.sqrt(2. * fancy_y_stuff) * lep_helicity * y * (2. - y) * xb * t_over_Q_squared / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_V = prefactor * bracket_term
    return s_1_zero_plus_unp_V

def i_s_unp_a_0p_1(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    one_plus_epsilon_squared_squared = (1. + ep**2)**2
    fancy_y_stuff = np.sqrt(1. - y - ep**2 * y**2 / 4.)
    prefactor = -8. * np.sqrt(2.) * lep_helicity * y * (2. - y) * (1. - 2. * xb) / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_A = prefactor * fancy_y_stuff * t * shorthand_k**2 / q_sq
    return s_1_zero_plus_unp_A

def i_c_unp_pp_2(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    first_bracket_term = 2. * ep**2 * k_tilde**2 / (root_one_plus_epsilon_squared * (1. + root_one_plus_epsilon_squared) * q_sq)
    second_bracket_term = xb * t_prime * t_over_Q_squared * (1. - xb - 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep**2 / xb) / q_sq
    prefactor = 8. * (2. - y) * (1. - y - ep**2 * y**2 / 4.) / root_one_plus_epsilon_squared**4
    c_2_plus_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_2_plus_plus_unp

def i_c_unp_v_pp_2(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = t_prime / q_sq
    major_term = (4. * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)) + 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb) * (1. + t_over_Q_squared) * t_prime_over_Q_squared
    prefactor = 8. * (2. - y) * (1. - y - ep**2 * y**2 / 4.) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_V_unp = prefactor * major_term
    return c_2_plus_plus_V_unp

def i_c_unp_a_pp_2(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = t_prime / q_sq
    first_bracket_term = 4. * (1. - 2. * xb) * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)
    second_bracket_term = (3.  - root_one_plus_epsilon_squared - 2. * xb + ep**2 / xb ) * xb * t_prime_over_Q_squared
    prefactor = 4. * (2. - y) * (1. - y - ep**2 * y**2 / 4.) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_A_unp = prefactor * (first_bracket_term - second_bracket_term)
    return c_2_plus_plus_A_unp

def i_c_unp_0p_2(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    epsilon_squared_over_2 = ep**2 / 2.
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb) / (1. + epsilon_squared_over_2)) * xb * t / q_sq
    prefactor = -8. * np.sqrt(2. * y_quantity) * shorthand_k * (2. - y) / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return c_2_zero_plus_unp

def i_c_unp_v_0p_2(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    y_quantity = np.sqrt(1. - y - (ep**2 * y**2 / 4.))
    prefactor = 8. * np.sqrt(2.) * y_quantity * shorthand_k * (2. - y) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * xb) * t_over_Q_squared)
    return c_2_zero_plus_unp_V

def i_c_unp_a_0p_2(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = t_prime / q_sq
    one_minus_xb = 1. - xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    bracket_term = one_minus_xb + 0.5 * t_prime_over_Q_squared * (4. * xb * one_minus_xb + ep**2) / root_one_plus_epsilon_squared
    prefactor = 8. * np.sqrt(2. * y_quantity) * shorthand_k * (2. - y) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * bracket_term
    return c_2_zero_plus_unp_A

def i_s_unp_pp_2(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    ep: float,
    y: float,
    t_prime: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    tPrime_over_Q_squared = t_prime / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    first_bracket_term = (ep**2 - xb * (root_one_plus_epsilon_squared - 1.)) / (1. + root_one_plus_epsilon_squared - 2. * xb)
    second_bracket_term = (2. * xb + ep**2) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -4. * lep_helicity * fancy_y_stuff * y * (1. + root_one_plus_epsilon_squared - 2. * xb) * tPrime_over_Q_squared / root_one_plus_epsilon_squared**3
    s_2_plus_plus_unp = prefactor * (first_bracket_term - second_bracket_term)
    return s_2_plus_plus_unp

def i_s_unp_v_pp_2(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    one_minus_2xb = 1. - 2. * xb
    bracket_term = root_one_plus_epsilon_squared - 1. + (one_minus_2xb + root_one_plus_epsilon_squared) * t_over_Q_squared
    parentheses_term = 1. - one_minus_2xb * t_over_Q_squared
    prefactor = -4. * lep_helicity * fancy_y_stuff * y * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_V = prefactor * parentheses_term * bracket_term
    return s_2_plus_plus_unp_V

def i_s_unp_a_pp_2(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    tPrime_over_Q_squared = t_prime / q_sq
    fancy_y_stuff = 1. - y - ep**2 * y**2 / 4.
    last_term = 1. + (4. * (1. - xb) * xb + ep**2) * t_over_Q_squared / (4. - 2. * xb + 3. * ep**2)
    middle_term = 1. + root_one_plus_epsilon_squared - 2. * xb
    prefactor = -8. * lep_helicity * fancy_y_stuff * y * t_over_Q_squared * tPrime_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_A = prefactor * middle_term * last_term
    return s_2_plus_plus_unp_A

def i_s_unp_0p_2(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    epsilon_squared_over_2 = ep**2 / 2.
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb) / (1. + epsilon_squared_over_2)) * xb * t / q_sq
    prefactor = 8. * lep_helicity * np.sqrt(2. * y_quantity) * shorthand_k * y / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return s_2_zero_plus_unp

def i_s_unp_v_0p_2(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    y_quantity = np.sqrt(1. - y - (ep**2 * y**2 / 4.))
    prefactor = -8. * np.sqrt(2.) * lep_helicity * y_quantity * shorthand_k * y * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * xb) * t_over_Q_squared)
    return s_2_zero_plus_unp_V

def i_s_unp_a_0p_2(
    lep_helicity: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_minus_xb = 1. - xb
    y_quantity = 1. - y - (ep**2 * y**2 / 4.)
    main_term = 4. * one_minus_xb + 2. * ep**2 + 4. * t_over_Q_squared * (4. * xb * one_minus_xb + ep**2)
    prefactor = -2. * np.sqrt(2. * y_quantity) * lep_helicity * shorthand_k * y * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * main_term
    return c_2_zero_plus_unp_A

def i_c_unp_pp_3(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    major_term = (1. - xb) * t_over_Q_squared + 0.5 * (root_one_plus_epsilon_squared - 1.) * (1. + t_over_Q_squared)
    intermediate_term = (root_one_plus_epsilon_squared - 1.) / root_one_plus_epsilon_squared**5
    prefactor = -8. * shorthand_k * (1. - y - ep**2 * y**2 / 4.)
    c_3_plus_plus_unp = prefactor * intermediate_term * major_term
    return c_3_plus_plus_unp

def i_c_unp_v_pp_3(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = np.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    major_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb) * t_over_Q_squared
    prefactor = -8. * shorthand_k * (1. - y - ep**2 * y**2 / 4.) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_3_plus_plus_V_unp = prefactor * major_term
    return c_3_plus_plus_V_unp

def i_c_unp_a_pp_3(
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    shorthand_k: float):
    main_term = t * t_prime * (xb * (1. - xb) + ep**2 / 4.) / q_sq**2
    prefactor = 16. * shorthand_k * (1. - y - ep**2 * y**2 / 4.) / np.power(1. + ep**2, 2.5)
    c_3_plus_plus_A_unp = prefactor * main_term
    return c_3_plus_plus_A_unp

def i_curly_c_unp(
    q_sq: float,
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_h: float,
    cff_h_tilde: float,
    cff_e: float) -> float:
    weighted_cffs = (f1 * cff_h) - (t * f2 * cff_e / (4. * _MASS_OF_PROTON_IN_GEV**2))
    second_term = xb * (f1 + f2) * cff_h_tilde / (2. - xb + (xb * t / q_sq))
    curly_C_unpolarized_interference = weighted_cffs + second_term
    return curly_C_unpolarized_interference

def i_curly_c_v_unp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_h: float,
    cff_e: float) -> float:
    cff_term = cff_h + cff_e
    second_term = xb * (f1 + f2) / (2. - xb + (xb * t / q_sq))
    curly_C_unpolarized_interference_V = cff_term * second_term
    return curly_C_unpolarized_interference_V

def i_curly_c_a_unp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_h: float) -> float:
    xb_modulation = xb * (f1 + f2) / (2. - xb + (xb * t / q_sq))
    curly_C_unpolarized_interference_A = cff_h * xb_modulation
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
    prefactor = -2. * lep_helicity * target_polar * y * (1. - y - (y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**5
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

def i_s_lp_0p_2(
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
    i_c_0p_a_2 = i_c_lp_v_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    
    return (i_c_pp_2*i_curly_c + i_c_pp_v_2*i_curly_c_v + i_c_pp_a_2*i_curly_c_a + i_c_0p_2*i_curly_c_eff + i_c_0p_v_2*i_curly_c_eff_v + i_c_0p_a_2*i_curly_c_eff_a)

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

    i_s_0p_2 = i_s_lp_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y)
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
    i_s_pp_a_3 = i_s_lp_a_pp_3(target_polar, q_sq, xb, t, ep, y, tprime, ktilde)

    i_s_0p_3 = 0.0
    i_s_0p_v_3 = 0.0
    i_s_0p_a_3 = 0.0
    
    return (i_s_pp_3*i_curly_c + i_s_pp_v_3*i_curly_c_v + i_s_pp_a_3*i_curly_c_a + i_s_0p_3*i_curly_c_eff + i_s_0p_v_3*i_curly_c_eff_v + i_s_0p_a_3*i_curly_c_eff_a)

def interference_amplitude(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, cff_re_et, cff_im_et, use_ww: bool = True):

    if (target_polar == -0.5 or target_polar == +0.5):
        if lep_helicity == 0.0:
            i_c0 = 0.5 * (
                i_lp_c0(+1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww) +
                i_lp_c0(-1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
            )

            i_c1 = 0.5 * (
                i_lp_c1(+1.0, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww) +
                i_lp_c1(-1.0, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
            )

            i_c2 = 0.5 * (
                i_lp_c2(+1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww) +
                i_lp_c2(+1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
            )

            i_c3 = 0.5 * 0.0

            i_s1 = 0.5 * (
                i_lp_s1(target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww) + 
                i_lp_s1(target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
            )

            i_s2 = 0.5 * (
                i_lp_s2(+1.0, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww) +
                i_lp_s2(-1.0, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
            )

            i_s3 = 0.5 * (
                i_lp_s3(1.0, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww) +
                i_lp_s3(-1.0, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
            )

        else:
            
            i_c0 = i_lp_c0(q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
            i_c1 = i_lp_c1(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
            i_c2 = i_lp_c2(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
            i_c3 = 0.0
            i_s1 = i_lp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
            i_s2 = i_lp_s2(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
            i_s3 = i_lp_s3(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)

    elif target_polar == 0.0:
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

            i_s2 = 0.5 * 0.0

        else:
            
            i_c0 = i_unp_c0(q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
            i_c1 = i_unp_c1(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
            i_c2 = i_unp_c2(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
            i_c3 = i_unp_c3(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
            i_s1 = i_unp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
            i_s2 = i_unp_s2(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
            i_s3 = 0.0

    return (
        (
            i_c0 * np.cos(0. * (np.pi - phi)) +
            i_c1 * np.cos(1. * (np.pi - phi)) + 
            i_c2 * np.cos(2. * (np.pi - phi)) + 
            i_c3 * np.cos(3. * (np.pi - phi)) + 
            i_s1 * np.sin(1. * (np.pi - phi)) + 
            i_s2 * np.sin(2. * (np.pi - phi)) + 
            i_s3 * np.sin(3. * (np.pi - phi))
        )/(xb * y * y * y * t * p1 * p2))
    
def bkm10_cross_section(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):
    
    bh_km15_plus_beam = bh_squared(+1.0, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)
    bh_km15_minus_beam = bh_squared(-1.0, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)

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
        cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    interference_km15_minus_beam = interference_amplitude(
        -1.0, target_polar,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    tf_cross_section_km15 = 0.0
    
    if lep_helicity == 0.0:
        tf_cross_section_km15 = 0.5 * (
            _CONVERSION_FACTOR*_QED_FINE_STRUCTURE**3*xb*y*y*(
                bh_km15_plus_beam + bh_km15_minus_beam +
                dvcs_km15_plus_beam + dvcs_km15_minus_beam +
                interference_km15_plus_beam + interference_km15_minus_beam) / (8.*np.pi*q_sq*q_sq*np.sqrt(1. + ep**2)))
        
    elif lep_helicity == 1.0:
        tf_cross_section_km15 = (
            _CONVERSION_FACTOR*_QED_FINE_STRUCTURE**3*xb*y*y*(
                bh_km15_plus_beam + 0.0 + dvcs_km15_plus_beam + 0.0 + interference_km15_plus_beam + 0.0) / (8.*np.pi*q_sq*q_sq*np.sqrt(1. + ep**2)))
        
    elif lep_helicity == -1.0:
        tf_cross_section_km15 = (
            _CONVERSION_FACTOR*_QED_FINE_STRUCTURE**3*xb*y*y*(
                0.0 + bh_km15_minus_beam + 0.0 + dvcs_km15_minus_beam + 0.0 + interference_km15_minus_beam) / (8.*np.pi*q_sq*q_sq*np.sqrt(1. + ep**2)))
        
    return tf_cross_section_km15

def bkm10_bsa(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    if (target_polar == -0.5 or target_polar == +0.5):
        # [TODO]: Code polarized target coefficients. @Woofmagic
        raise NotImplementedError("NO POLARIZED TARGET YET!")

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

    plus_longitudinally_polarized_cross_section = bkm10_cross_section(
        lep_helicity, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_longitudinally_polarized_cross_section = bkm10_cross_section(
        lep_helicity, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    
    tf_bsa = (
        (plus_longitudinally_polarized_cross_section - minus_longitudinally_polarized_cross_section) / 
        (plus_longitudinally_polarized_cross_section + minus_longitudinally_polarized_cross_section))
        
    return tf_bsa

###########################
# CHECKING THE VALUES
###########################

bh_unpolarized_c0 = bh_unp_c0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K, TEST_F1, TEST_F2) # correct: 2025-10-28
bh_unpolarized_c1 = bh_unp_c1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K, TEST_F1, TEST_F2) # correct: 2025-10-28
bh_unpolarized_c2 = bh_unp_c2(TEST_XB, TEST_T, TEST_K, TEST_F1, TEST_F2) # correct: 2025-10-28

curly_c_real_f_fstar = curly_c_real(
        TEST_QSQ, TEST_XB, TEST_T, TEST_EP,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, -TEST_IM_CFF_H, -TEST_IM_CFF_HT, -TEST_IM_CFF_E, -TEST_IM_CFF_ET)

curly_c_real_feff_feffstar = curly_c_real(
        TEST_QSQ, TEST_XB, TEST_T, TEST_EP,
        f_eff(TEST_XI, TEST_REAL_CFF_H), f_eff(TEST_XI, TEST_REAL_CFF_HT), f_eff(TEST_XI, TEST_REAL_CFF_E), 
        f_eff(TEST_XI, TEST_REAL_CFF_ET), f_eff(TEST_XI, TEST_IM_CFF_H), f_eff(TEST_XI, TEST_IM_CFF_HT), 
        f_eff(TEST_XI, TEST_IM_CFF_E), f_eff(TEST_XI, TEST_IM_CFF_ET),
        f_eff(TEST_XI, TEST_REAL_CFF_H), f_eff(TEST_XI, TEST_REAL_CFF_HT), f_eff(TEST_XI, TEST_REAL_CFF_E), f_eff(TEST_XI, TEST_REAL_CFF_ET), 
        -f_eff(TEST_XI, TEST_IM_CFF_H), -f_eff(TEST_XI, TEST_IM_CFF_HT), -f_eff(TEST_XI, TEST_IM_CFF_E), -f_eff(TEST_XI, TEST_IM_CFF_ET))

curly_c_imag_f_fstar = curly_c_imag(
        TEST_QSQ, TEST_XB, TEST_T, TEST_EP,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, -TEST_IM_CFF_H, -TEST_IM_CFF_HT, -TEST_IM_CFF_E, -TEST_IM_CFF_ET)

curly_c_imag_feff_feffstar = curly_c_imag(
        TEST_QSQ, TEST_XB, TEST_T, TEST_EP,
        f_eff(TEST_XI, TEST_REAL_CFF_H), f_eff(TEST_XI, TEST_REAL_CFF_HT), f_eff(TEST_XI, TEST_REAL_CFF_E), f_eff(TEST_XI, TEST_REAL_CFF_ET),
        f_eff(TEST_XI, TEST_IM_CFF_H), f_eff(TEST_XI, TEST_IM_CFF_HT), f_eff(TEST_XI, TEST_IM_CFF_E), f_eff(TEST_XI, TEST_IM_CFF_ET),
        f_eff(TEST_XI, TEST_REAL_CFF_H), f_eff(TEST_XI, TEST_REAL_CFF_HT), f_eff(TEST_XI, TEST_REAL_CFF_E), f_eff(TEST_XI, TEST_REAL_CFF_ET),
        -f_eff(TEST_XI, TEST_IM_CFF_H), -f_eff(TEST_XI, TEST_IM_CFF_HT), -f_eff(TEST_XI, TEST_IM_CFF_E), -f_eff(TEST_XI, TEST_IM_CFF_ET))

dvcs_unpolarized_c0 = dvcs_unp_c0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K,
                  TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)

dvcs_unpolarized_c1 = dvcs_unp_c1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K,
                  TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)

dvcs_unpolarized_s1 = dvcs_unp_s1(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K,
                  TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)

i_c_pp_0_unp = i_c_unp_pp_0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_c_v_pp_0_unp = i_c_unp_v_pp_0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_c_a_pp_0_unp = i_c_unp_a_pp_0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)

i_c_pp_1_unp = i_c_unp_pp_1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_v_pp_1_unp = i_c_unp_v_pp_1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)
i_c_a_pp_1_unp = i_c_unp_a_pp_1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)

i_c_pp_2_unp = i_c_unp_pp_2(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K_TILDE)
i_c_v_pp_2_unp = i_c_unp_v_pp_2(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K_TILDE)
i_c_a_pp_2_unp = i_c_unp_a_pp_2(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K_TILDE)

i_c_pp_3_unp = i_c_unp_pp_3(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_v_pp_3_unp = i_c_unp_v_pp_3(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_a_pp_3_unp = i_c_unp_a_pp_3(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)

i_c_0p_0_unp = i_c_unp_0p_0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_v_0p_0_unp = i_c_unp_v_0p_0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_a_0p_0_unp = i_c_unp_a_0p_0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_c_0p_1_unp = i_c_unp_0p_1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME)
i_c_v_0p_1_unp = i_c_unp_v_0p_1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_c_a_0p_1_unp = i_c_unp_a_0p_1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)

i_c_0p_2_unp = i_c_unp_0p_2(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_v_0p_2_unp = i_c_unp_v_0p_2(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_a_0p_2_unp = i_c_unp_a_0p_2(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)

i_s_pp_1_unp = i_s_unp_pp_1(1.0, TEST_QSQ, TEST_XB, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)
i_s_v_pp_1_unp = i_s_unp_v_pp_1(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_s_a_pp_1_unp = i_s_unp_a_pp_1(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)

i_s_pp_2_unp = i_s_unp_pp_2(1.0, TEST_QSQ, TEST_XB, TEST_EP, TEST_Y, TEST_T_PRIME)
i_s_v_pp_2_unp = i_s_unp_v_pp_2(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y)
i_s_a_pp_2_unp = i_s_unp_a_pp_2(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME)

i_s_0p_1_unp = i_s_unp_0p_1(1.0, TEST_QSQ, TEST_EP, TEST_Y, TEST_K_TILDE)
i_s_v_0p_1_unp = i_s_unp_v_0p_1(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y)
i_s_a_0p_1_unp = i_s_unp_a_0p_1(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_s_0p_2_unp = i_s_unp_0p_2(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_s_v_0p_2_unp = i_s_unp_v_0p_2(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_s_a_0p_2_unp = i_s_unp_a_0p_2(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_curly_c_unpolarized_real = i_curly_c_unp(TEST_QSQ, TEST_XB, TEST_T, TEST_F1, TEST_F2, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E)
i_curly_c_v_unpolarized_real = i_curly_c_v_unp(TEST_QSQ, TEST_XB, TEST_T, TEST_F1, TEST_F2, TEST_REAL_CFF_H, TEST_REAL_CFF_E)
i_curly_c_a_unpolarized_real = i_curly_c_a_unp(TEST_QSQ, TEST_XB, TEST_T, TEST_F1, TEST_F2, TEST_REAL_CFF_HT)

i_curly_c_unpolarized_imag = i_curly_c_unp(TEST_QSQ, TEST_XB, TEST_T, TEST_F1, TEST_F2, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E)
i_curly_c_v_unpolarized_imag = i_curly_c_v_unp(TEST_QSQ, TEST_XB, TEST_T, TEST_F1, TEST_F2, TEST_IM_CFF_H, TEST_IM_CFF_E)
i_curly_c_a_unpolarized_imag = i_curly_c_a_unp(TEST_QSQ, TEST_XB, TEST_T, TEST_F1, TEST_F2, TEST_IM_CFF_HT)

i_c0_unp = i_unp_c0(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E)
i_c1_unp = i_unp_c1(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_T_PRIME, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E)
i_c2_unp = i_unp_c2(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_T_PRIME, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E)
i_c3_unp = i_unp_c3(TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_T_PRIME, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E)

i_s1_unp = i_unp_s1(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_T_PRIME, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E) 
i_s2_unp = i_unp_s2(1.0, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_T_PRIME, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E)

bh_c0_lp= bh_lp_c0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_F1, TEST_F2)
bh_c1_lp = bh_lp_c1(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K, TEST_F1, TEST_F2)
bh_c2_lp = 0.

curly_c_real_f_fstar_lp = curly_c_real_lp(
        TEST_QSQ, TEST_XB, TEST_T, TEST_EP,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, -TEST_IM_CFF_H, -TEST_IM_CFF_HT, -TEST_IM_CFF_E, -TEST_IM_CFF_ET)

curly_c_imag_f_fstar_lp = curly_c_imag_lp(
        TEST_QSQ, TEST_XB, TEST_T, TEST_EP,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET,
        TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, -TEST_IM_CFF_H, -TEST_IM_CFF_HT, -TEST_IM_CFF_E, -TEST_IM_CFF_ET)

curly_c_imag_feff_feffstar_lp = curly_c_imag_lp(
        TEST_QSQ, TEST_XB, TEST_T, TEST_EP,
        f_eff(TEST_XI, TEST_REAL_CFF_H), f_eff(TEST_XI, TEST_REAL_CFF_HT), f_eff(TEST_XI, TEST_REAL_CFF_E), f_eff(TEST_XI, TEST_REAL_CFF_ET),
        f_eff(TEST_XI, TEST_IM_CFF_H), f_eff(TEST_XI, TEST_IM_CFF_HT), f_eff(TEST_XI, TEST_IM_CFF_E), f_eff(TEST_XI, TEST_IM_CFF_ET),
        f_eff(TEST_XI, TEST_REAL_CFF_H), f_eff(TEST_XI, TEST_REAL_CFF_HT), f_eff(TEST_XI, TEST_REAL_CFF_E), f_eff(TEST_XI, TEST_REAL_CFF_ET),
        -f_eff(TEST_XI, TEST_IM_CFF_H), -f_eff(TEST_XI, TEST_IM_CFF_HT), -f_eff(TEST_XI, TEST_IM_CFF_E), -f_eff(TEST_XI, TEST_IM_CFF_ET))

dvcs_c0_lp = dvcs_lp_c0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K,
                  TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)

dvcs_c1_lp = dvcs_lp_c1(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K,
                  TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)

dvcs_s1_lp = dvcs_lp_s1(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K,
                  TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)

i_c_pp_0_lp= i_c_lp_pp_0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_c_v_pp_0_lp = i_c_lp_v_pp_0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_c_a_pp_0_lp = i_c_lp_a_pp_0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)

i_c_pp_1_lp = i_c_lp_pp_1(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_v_pp_1_lp = i_c_lp_v_pp_1(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)
i_c_a_pp_1_lp = i_c_lp_a_pp_1(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_c_pp_2_lp = i_c_lp_pp_2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y)
i_c_v_pp_2_lp = i_c_lp_v_pp_2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y)
i_c_a_pp_2_lp = i_c_lp_a_pp_2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y)

i_c_0p_0_lp = i_c_lp_0p_0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_v_0p_0_lp = i_c_lp_v_0p_0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_a_0p_0_lp = i_c_lp_a_0p_0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_c_0p_1_lp = i_c_lp_0p_1(1.0, 0.5, TEST_QSQ, TEST_EP, TEST_Y, TEST_K_TILDE, TEST_K)
i_c_v_0p_1_lp = i_c_lp_v_0p_1(1.0, 0.5, TEST_QSQ, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_c_a_0p_1_lp = 0.0

i_c_0p_2_lp = i_c_lp_0p_2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_v_0p_2_lp = i_c_lp_v_0p_2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_c_a_0p_2_lp = i_c_lp_a_0p_2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_s_pp_1_lp = i_s_lp_pp_1(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_s_v_pp_1_lp = i_s_lp_v_pp_1(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K_TILDE)
i_s_a_pp_1_lp = i_s_lp_a_pp_1(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_s_pp_2_lp = i_s_lp_pp_2(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K_TILDE)
i_s_v_pp_2_lp = i_s_lp_v_pp_2(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K_TILDE)
i_s_a_pp_2_lp = i_s_lp_a_pp_2(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K_TILDE)

i_s_pp_3_lp = i_s_lp_pp_3(0.5, TEST_QSQ, TEST_XB, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)
i_s_v_pp_3_lp = i_s_lp_v_pp_3(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)
i_s_a_pp_3_lp = i_s_lp_a_pp_3(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_T_PRIME, TEST_K)

i_s_0p_1_lp = i_s_lp_0p_1(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_s_v_0p_1_lp = i_s_lp_v_0p_1(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K_TILDE)
i_s_a_0p_1_lp = i_s_lp_a_0p_1(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y)

i_s_0p_2_lp = i_s_lp_0p_2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y)
i_s_v_0p_2_lp = i_s_lp_v_0p_2(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)
i_s_a_0p_2_lp = i_s_lp_a_0p_2(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_K)

i_c0_lp = i_lp_c0(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET)
i_c1_lp = i_lp_c1(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_T_PRIME, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET)
i_c2_lp = i_lp_c2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET)

i_s1_lp = i_lp_s1(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_T_PRIME, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET) 
i_s2_lp = i_lp_s2(1.0, 0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_T_PRIME, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)
i_s3_lp = i_lp_s3(0.5, TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_T_PRIME, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET)

print(f"BH c0 unp: {bh_unpolarized_c0}") # correct, 2026/01/23
print(f"BH c1 unp: {bh_unpolarized_c1}") # correct, 2026/01/23
print(f"BH c2 unp: {bh_unpolarized_c2}") # correct, 2026/01/23

print(f"Re[Curly C(F | F*)] = {curly_c_real_f_fstar}") # correct, 2026/01/23
print(f"Re[Curly C(Feff | Feff*)] = {curly_c_real_feff_feffstar}") # correct, 2026/01/23
print(f"Im[Curly C(F | F*)] = {curly_c_imag_f_fstar}") # correct, 2026/01/23
print(f"Im[Curly C(Feff | Feff*)] = {curly_c_imag_feff_feffstar}") # correct, 2026/01/23

print(f"DVCS c0 unp: {dvcs_unpolarized_c0}") # correct, 2026/01/23
print(f"DVCS c1 unp: {dvcs_unpolarized_c1}") # correct, 2026/01/23
print(f"DVCS s1 unp: {dvcs_unpolarized_s1}") # correct, 2026/01/23

print(f"Int C++(n = 0) unp: {i_c_pp_0_unp}") # correct, 2026/01/23
print(f"Int CV++(n = 0) unp: {i_c_v_pp_0_unp}") # correct, 2026/01/23
print(f"Int CA++(n = 0) unp: {i_c_a_pp_0_unp}") # correct, 2026/01/23

print(f"Int C++(n = 1) unp: {i_c_pp_1_unp}") # correct, 2026/01/23
print(f"Int CV++(n = 1) unp: {i_c_v_pp_1_unp}") # correct, 2026/01/23
print(f"Int CA++(n = 1) unp: {i_c_a_pp_1_unp}") # correct, 2026/01/23

print(f"Int C++(n = 2) unp: {i_c_pp_2_unp}") # correct, 2026/01/23
print(f"Int CV++(n = 2) unp: {i_c_v_pp_2_unp}") # correct, 2026/01/23
print(f"Int CA++(n = 2) unp: {i_c_a_pp_2_unp}") # correct, 2026/01/23

print(f"Int C++(n = 3) unp: {i_c_pp_3_unp}") # correct, 2026/01/23
print(f"Int CV++(n = 3) unp: {i_c_v_pp_3_unp}") # correct, 2026/01/23
print(f"Int CA++(n = 3) unp: {i_c_a_pp_3_unp}") # correct, 2026/01/23

print(f"Int C0+(n = 0) unp: {i_c_0p_0_unp}") # correct, 2026/01/23
print(f"Int CV0+(n = 0) unp: {i_c_v_0p_0_unp}") # correct, 2026/01/23
print(f"Int CA0+(n = 0) unp: {i_c_a_0p_0_unp}") # correct, 2026/01/23

print(f"Int C0+(n = 1) unp: {i_c_0p_1_unp}") # correct, 2026/01/23
print(f"Int CV0+(n = 1) unp: {i_c_v_0p_1_unp}") # correct, 2026/01/23
print(f"Int CA0+(n = 1) unp: {i_c_a_0p_1_unp}") # correct, 2026/01/23

print(f"Int C0+(n = 2) unp: {i_c_0p_2_unp}") # correct, 2026/01/23
print(f"Int CV0+(n = 2) unp: {i_c_v_0p_2_unp}") # correct, 2026/01/23
print(f"Int CA0+(n = 2) unp: {i_c_a_0p_2_unp}") # correct, 2026/01/23

print(f"Int S++(n = 1) unp: {i_s_pp_1_unp}") # correct, 2026/01/23
print(f"Int SV++(n = 1) unp: {i_s_v_pp_1_unp}") # correct, 2026/01/23
print(f"Int SA++(n = 1) unp: {i_s_a_pp_1_unp}") # correct, 2026/01/23

print(f"Int S++(n = 2) unp: {i_s_pp_2_unp}") # correct, 2026/01/23
print(f"Int SV++(n = 2) unp: {i_s_v_pp_2_unp}") # correct, 2026/01/23
print(f"Int SA++(n = 2) unp: {i_s_a_pp_2_unp}") # correct, 2026/01/23

print(f"Int S0+(n = 1) unp: {i_s_0p_1_unp}") # correct, 2026/01/23
print(f"Int SV0+(n = 1) unp: {i_s_v_0p_1_unp}") # correct, 2026/01/23
print(f"Int SA0+(n = 1) unp: {i_s_a_0p_1_unp}") # correct, 2026/01/23

print(f"Int S0+(n = 2) unp: {i_s_0p_2_unp}") # correct, 2026/01/23
print(f"Int SV0+(n = ) unp: {i_s_v_0p_2_unp}") # correct, 2026/01/23
print(f"Int SA0+(n = 2) unp: {i_s_a_0p_2_unp}") # correct, 2026/01/23

print(f"Re[Curly C(F)] = {i_curly_c_unpolarized_real}") # correct, 2026/01/23
print(f"Re[Curly CV(F)] = {i_curly_c_v_unpolarized_real}") # correct, 2026/01/23
print(f"Re[Curly CA(F)] = {i_curly_c_a_unpolarized_real}") # correct, 2026/01/23

print(f"Im[Curly C(F)] = {i_curly_c_unpolarized_imag}") # correct, 2026/01/23
print(f"Im[Curly CV(F)] = {i_curly_c_v_unpolarized_imag}") # correct, 2026/01/23
print(f"Im[Curly CA(F)] = {i_curly_c_a_unpolarized_imag}") # correct, 2026/01/23

print(f"Int c0 unp: {i_c0_unp}") # correct, 2026/01/23
print(f"Int c1 unp: {i_c1_unp}") # correct, 2026/01/23
print(f"Int c2 unp: {i_c2_unp}") # correct, 2026/01/23
print(f"Int c3 unp: {i_c3_unp}") # correct, 2026/01/23

print(f"Int s1 unp: {i_s1_unp}") # correct, 2026/01/23
print(f"Int s2 unp: {i_s2_unp}") # correct, 2026/01/23

print(f"BH c0 LP: {bh_c0_lp}") # correct, 2026/01/23
print(f"BH c1 LP: {bh_c1_lp}") # correct, 2026/01/23
print(f"BH c2 LP: {bh_c2_lp}") # correct, 2026/01/23

print(f"Re[Curly C(F | F*)] = {curly_c_real_f_fstar_lp}") # correct, 2026/01/23
print(f"Im[Curly C(F | F*)] = {curly_c_imag_f_fstar_lp}") # correct, 2026/01/23
print(f"Im[Curly C(Feff | Feff*)] = {curly_c_imag_feff_feffstar_lp}") # correct, 2026/01/23

print(f"DVCS c0 LP: {dvcs_c0_lp}") # correct, 2026/01/23
print(f"DVCS c1 LP: {dvcs_c1_lp}") # correct, 2026/01/23
print(f"DVCS s1 LP: {dvcs_s1_lp}") # correct, 2026/01/23

print(f"Int C++(n = 0) LP: {i_c_pp_0_lp}") # correct, 2026/01/23
print(f"Int CV++(n = 0) LP: {i_c_v_pp_0_lp}") # correct, 2026/01/23
print(f"Int CA++(n = 0) LP: {i_c_a_pp_0_lp}") # correct, 2026/01/23

print(f"Int C++(n = 1) LP: {i_c_pp_1_lp}") # correct, 2026/01/23
print(f"Int CV++(n = 1) LP: {i_c_v_pp_1_lp}") # correct, 2026/01/23
print(f"Int CA++(n = 1) LP: {i_c_a_pp_1_lp}") # correct, 2026/01/23

print(f"Int C++(n = 2) LP: {i_c_pp_2_lp}") # correct, 2026/01/23
print(f"Int CV++(n = 2) LP: {i_c_v_pp_2_lp}") # correct, 2026/01/23
print(f"Int CA++(n = 2) LP: {i_c_a_pp_2_lp}") # correct, 2026/01/23

print(f"Int C++(n = 3) LP: {0}") # correct, 2026/01/23
print(f"Int CV++(n = 3) LP: {0}") # correct, 2026/01/23
print(f"Int CA++(n = 3) LP: {0}") # correct, 2026/01/23

print(f"Int C0+(n = 0) LP: {i_c_0p_0_lp}") # correct, 2026/01/23
print(f"Int CV0+(n = 0) LP: {i_c_v_0p_0_lp}") # correct, 2026/01/23
print(f"Int CA0+(n = 0) LP: {i_c_a_0p_0_lp}") # correct, 2026/01/23

print(f"Int C0+(n = 1) LP: {i_c_0p_1_lp}") # correct, 2026/01/23
print(f"Int CV0+(n = 1) LP: {i_c_v_0p_1_lp}") # correct, 2026/01/23
print(f"Int CA0+(n = 1) LP: {i_c_a_0p_1_lp}") # correct, 2026/01/23

print(f"Int C0+(n = 2) LP: {i_c_0p_2_lp}") # correct, 2026/01/23
print(f"Int CV0+(n = 2) LP: {i_c_v_0p_2_lp}") # correct, 2026/01/23
print(f"Int CA0+(n = 2) LP: {i_c_a_0p_2_lp}") # correct, 2026/01/23

print(f"Int S++(n = 1) LP: {i_s_pp_1_lp}") # correct, 2026/01/23
print(f"Int SV++(n = 1) LP: {i_s_v_pp_1_lp}") # correct, 2026/01/23
print(f"Int SA++(n = 1) LP: {i_s_a_pp_1_lp}") # correct, 2026/01/23

print(f"Int S++(n = 2) LP: {i_s_pp_2_lp}") # correct, 2026/01/23
print(f"Int SV++(n = 2) LP: {i_s_v_pp_2_lp}") # correct, 2026/01/23
print(f"Int SA++(n = 2) LP: {i_s_a_pp_2_lp}") # correct, 2026/01/23

print(f"Int S0+(n = 1) LP: {i_s_0p_1_lp}") # correct, 2026/01/23
print(f"Int SV0+(n = 1) LP: {i_s_v_0p_1_lp}") # correct, 2026/01/23
print(f"Int SA0+(n = 1) LP: {i_s_a_0p_1_lp}") # correct, 2026/01/23

print(f"Int S0+(n = 2) LP: {i_s_0p_2_lp}") # correct, 2026/01/23
print(f"Int SV0+(n = ) LP: {i_s_v_0p_2_lp}") # correct, 2026/01/23
print(f"Int SA0+(n = 2) LP: {i_s_a_0p_2_lp}") # correct, 2026/01/23

print(f"Int c0 LP: {i_c0_lp}") # correct, 2026/01/23
print(f"Int c1 LP: {i_c1_lp}") # correct, 2026/01/23
print(f"Int c2 LP: {i_c2_lp}") # correct, 2026/01/23
print(f"Int c3 LP: {0}") # correct, 2026/01/23

print(f"Int s1 LP: {i_s1_lp}") # correct, 2026/01/23
print(f"Int s2 LP: {i_s2_lp}") # correct, 2026/01/23
print(f"Int s3 LP: {i_s3_lp}") # correct, 2026/01/23

###########################
# COMPUTING OBSERVABLES
###########################

unp_beam_unp_target_cross_section_xdj = bkm10_cross_section(
    0.0, 0.0,
    TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_T_PRIME, phi_array_in_radians, TEST_P1, TEST_P2,
    TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET
)

unp_beam_unp_target_bsa_xdj = bkm10_bsa(
    0.0, 0.0,
    TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_T_PRIME, phi_array_in_radians, TEST_P1, TEST_P2,
    TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET
)

unp_beam_unp_target_tsa_xdj = bkm10_tsa(
    0.0,
    TEST_QSQ, TEST_XB, TEST_T, TEST_EP, TEST_Y, TEST_XI, TEST_K, TEST_F1, TEST_F2, TEST_K_TILDE, TEST_T_PRIME, phi_array_in_radians, TEST_P1, TEST_P2,
    TEST_REAL_CFF_H, TEST_REAL_CFF_HT, TEST_REAL_CFF_E, TEST_REAL_CFF_ET, TEST_IM_CFF_H, TEST_IM_CFF_HT, TEST_IM_CFF_E, TEST_IM_CFF_ET
)

xsec_test_fig, xsec_test_ax = plt.subplots(1, 1, figsize = (14, 7))
xsec_test_ax.scatter(phi_array_in_radians, unp_beam_unp_target_cross_section_xdj, color = "blue", s = 2., label = "TF(KM15)")
xsec_test_ax.set_xlabel(r"Radians ($\phi$)", fontsize = 14)
xsec_test_ax.legend()
plt.show()
plt.close(xsec_test_fig)

bsa_test_fig, bsa_test_ax = plt.subplots(1, 1, figsize = (14, 7))
bsa_test_ax.scatter(phi_array_in_radians, unp_beam_unp_target_bsa_xdj, color = "blue", s = 2., label = "TF(KM15)")
bsa_test_ax.set_xlabel(r"Radians ($\phi$)", fontsize = 14)
bsa_test_ax.legend()
plt.show()
plt.close(bsa_test_fig)

tsa_test_fig, tsa_test_ax = plt.subplots(1, 1, figsize = (14, 7))
tsa_test_ax.scatter(phi_array_in_radians, unp_beam_unp_target_tsa_xdj, color = "blue", s = 2., label = "TF(KM15)")
tsa_test_ax.set_xlabel(r"Radians ($\phi$)", fontsize = 14)
tsa_test_ax.legend()
plt.show()
plt.close(tsa_test_fig)