##########################################
# FILE INFORMATION:
# Purpose: make a huge dataset with extant
# experimental data and *then* partition it
# accordingly into kinematic settings.
# Created: 20260325
# Last changed: 20260325
##########################################

print(f"[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import os

import pandas as pd
import numpy as np
import gepard as g
from gepard.fits import th_KM15
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

#################################################################################
# Scratch path
#################################################################################

# verify this is what you want
SCRATCH_PATH = 'placeholder!'

#################################################################################
# Version numbers!
#################################################################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

#################################################################################
# Now, we find which gepard sets contain valid kinematics *and* observables!
#################################################################################

target_observables = {'XGAMMA', 'XUU', 'AC', 'ALU', 'AUL'}

# dictionary of string-to-list key-value pairs:
desired_observable_dictionary = {observable: [] for observable in target_observables}
required_attributes = ['xB', 't', 'Q2', 'in1energy', 'observable', 'val', 'err']
valid_datasets = []

for dataset_index, dataset in g.dset.items():

    # check the first datapoint in the DataSet for contents!
    first_gepard_datapoint = dataset[0] if len(dataset) > 0 else None
    
    if first_gepard_datapoint and all(hasattr(first_gepard_datapoint, kinematic_attribute) for kinematic_attribute in required_attributes):
        valid_datasets.append(dataset_index)

print(f"[INFO]: Valid dataset indices are:\n{sorted(valid_datasets)}")
print(f"[INFO]: Length of valid datasets = {len(valid_datasets)}")
print(f"[INFO]: Compare length of *all* dataset = {len(g.dset)}")
print(f"[INFO]: Total invalid (according to our criteria) datasets = {len(g.dset) - len(valid_datasets)}")

for dataset_index in valid_datasets:
    if not hasattr(g.dset[dataset_index][0], "observable"):
        print(f"[WARN]: Dataset {g.dset[dataset_index]} had datapoint without observable property...")
        continue
    
    observable_name = g.dset[dataset_index][0].observable
    
    if observable_name in desired_observable_dictionary:
        desired_observable_dictionary[observable_name].append(dataset_index)

for name, indices in desired_observable_dictionary.items():
    print(f"{name}: {indices}")
    
rows_for_experimentally_derived_pseudodata = []
rows_for_experimental_data_only = []
global_set_counter = 1

for experiment_id in desired_observable_dictionary['XUU']:

    dataset = g.dset[experiment_id]
    print(f"[INFO]: Now iterating on Experiment {dataset.collaboration} ({dataset.year})")

    kinematic_set_number = 1 
    
    for datapoint in dataset:
        if not hasattr(datapoint, "observable"):
            print(f"[WARN]: Datapoint for Experiment {g.dset[experiment_id].collaboration} ({g.dset[experiment_id].year}) has no observable...")
        else:
            assert datapoint.observable == "XUU", "[ASSERT]: Current datapoint observable not matching!"

        if all(hasattr(datapoint, attr) for attr in ["in1energy", "xB", "Q2", "t", "phi"]):
            
            # extact the CFFs using km15
            km15_real_h = th_KM15.ReH(datapoint)
            km15_imag_h = th_KM15.ImH(datapoint)
            km15_real_e = th_KM15.ReE(datapoint)
            km15_imag_e = th_KM15.ImE(datapoint)
            km15_real_ht = th_KM15.ReHt(datapoint)
            km15_imag_ht = th_KM15.ImHt(datapoint)
            km15_real_et = th_KM15.ReEt(datapoint)
            km15_imag_et = th_KM15.ImEt(datapoint)

            km15_bkm10_cross_section = DifferentialCrossSection(
                configuration = {
                    "kinematics": BKM10Inputs(
                        lab_kinematics_k = datapoint.in1energy,
                        squared_Q_momentum_transfer = datapoint.Q2,
                        x_Bjorken = datapoint.xB,
                        squared_hadronic_momentum_transfer_t = datapoint.t),
                    "cff_inputs": CFFInputs(
                        compton_form_factor_h = complex(km15_real_h, km15_imag_h),
                        compton_form_factor_h_tilde = complex(km15_real_ht, km15_imag_ht),
                        compton_form_factor_e = complex(km15_real_e, km15_imag_e),
                        compton_form_factor_e_tilde = complex(km15_real_et, km15_imag_et)),
                    "using_ww": True
                },
                verbose = False,
                debugging = False)
            
            unpolarized_cross_section = km15_bkm10_cross_section.compute_cross_section(
                datapoint.phi,
                lepton_helicity = 0.0,
                target_polarization = 0.0).real
            
            bkm10_bsa_km15 = km15_bkm10_cross_section.compute_bsa(
                datapoint.phi, 
                target_polarization = 0.0).real

            experimentally_derived_pseudodata = {
                "set": global_set_counter,
                "k": datapoint.in1energy,
                "q_squared": datapoint.Q2,
                "x_b": datapoint.xB,
                "t": datapoint.t,
                "phi": datapoint.phi,
                "unp_beam_unp_target_xsec": unpolarized_cross_section[0], # [0] index needed because datapoint.phi is *not* an array...
                "unp_beam_unp_target_xsec_err": 0.0,
                "unp_beam_unp_target_xsec_errstat": 0.0,
                "unp_beam_unp_target_xsec_errsyst": 0.0,
                "unp_target_bsa": bkm10_bsa_km15[0], # [0] index needed because datapoint.phi is *not* an array...
                "unp_target_bsa_err": 0.0,
                "unp_target_bsa_errstat": 0.0,
                "unp_target_bsa_errsyst": 0.0,
                "Re[H]": km15_real_h, "Im[H]": km15_imag_h,
                "Re[E]": km15_real_e, "Im[E]": km15_imag_e,
                "Re[Ht]": km15_real_ht, "Im[Ht]": km15_imag_ht,
                "Re[Et]": km15_real_et, "Im[Et]": km15_imag_et,
                "experiment_year": f"{dataset.collaboration}_{dataset.year}",
                "flag": "unknown"
            }

            experimental_data_point = {
                "set": global_set_counter,
                "k": datapoint.in1energy,
                "q_squared": datapoint.Q2,
                "x_b": datapoint.xB,
                "t": datapoint.t,
                "phi": datapoint.phi,
                "unp_beam_unp_target_xsec": datapoint.val,
                "unp_beam_unp_target_xsec_err": datapoint.err,
                "unp_beam_unp_target_xsec_errstat": datapoint.errstat,
                "unp_beam_unp_target_xsec_errsyst": datapoint.errsyst,
                "unp_target_bsa": 0.0,
                "unp_target_bsa_err": 0.0,
                "unp_target_bsa_errstat": 0.0,
                "unp_target_bsa_errsyst": 0.0,
                "Re[H]": km15_real_h, "Im[H]": km15_imag_h,
                "Re[E]": km15_real_e, "Im[E]": km15_imag_e,
                "Re[Ht]": km15_real_ht, "Im[Ht]": km15_imag_ht,
                "Re[Et]": km15_real_et, "Im[Et]": km15_imag_et,
                "coordinate_frame": datapoint.frame,
                "experiment_year": f"{dataset.collaboration}_{dataset.year}",
                "flag": "unknown"
            }

            rows_for_experimentally_derived_pseudodata.append(experimentally_derived_pseudodata)
            rows_for_experimental_data_only.append(experimental_data_point)

            del experimentally_derived_pseudodata
            del experimental_data_point
            del km15_bkm10_cross_section
        else:
            print(f"[WARN]: Missing kinematics in {dataset.collaboration}")

    global_set_counter += 1
    del dataset


df_expermentally_derived = pd.DataFrame(rows_for_experimentally_derived_pseudodata)
df_expermental_data = pd.DataFrame(rows_for_experimental_data_only)

print(f"[INFO]: Total number of rows in exp-derived DF: {len(df_expermentally_derived)}")
print(f"[INFO]: Total number of rows in exp-only DF: {len(df_expermental_data)}")

print(f"[INFO]: We are now reassigning the kinematic set index such that unique (k, x_b, q_squared, t) values each get their own set index.")

# previous loop did *not* correctly assign kinematic settings; we now remedy the issue using ngroup()... thanks to Google Gemini!
# [NOTE]: WE ROUND BY 4 BELOW: THAT IS EFFECTIVELY BINNING THE KINEMATIC SPACE!
df_expermentally_derived['set'] = df_expermentally_derived.round(4).groupby(['k', 'x_b', 't', 'q_squared'], sort = False).ngroup() + 1
df_expermental_data['set'] = df_expermental_data.round(4).groupby(['k', 'x_b', 't', 'q_squared'], sort = False).ngroup() + 1
print(f"[INFO]: Dataframe kinematic bins now should be uniquely indexed by kinematic set index! Checking...")

unique_sets_exp_derived = df_expermentally_derived['set'].nunique()
unique_sets_exp_data = df_expermental_data['set'].nunique()
print(f"[INFO]: Total Datapoints in exp-derived DF: {unique_sets_exp_derived}")
print(f"[INFO]: Total Datapoints in exp-only DF: {unique_sets_exp_data}")

df_expermentally_derived.to_csv(path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/main_pseudodata_file_v{MAJOR_MINOR_NUMBER}.csv")
df_expermental_data.to_csv(path_or_buf = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/experimental_data_v{MAJOR_MINOR_NUMBER}.csv")

#################################################################################
# Required functions for numerical computations:
#################################################################################

_CONVERSION_GEV6_GEV4NB = .389379 * 1000000.
_MASS_OF_PROTON_IN_GEV = 0.93827208816
_QED_FINE_STRUCTURE = 1./137.035999177
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

def compute_k_dot_delta(q_squared, xb, t, phi_azi, ep, y_lep, k):
    return (-1.*q_squared / (2.*y_lep*(1.+ep**2))) * (1. + ((2.*k*np.cos(np.pi- phi_azi)) - ((t / q_squared)*(1.-(xb * (2. - y_lep)) + (y_lep * ep**2 / 2.))) + (y_lep * ep**2 / 2.)))

def prop_1(q_squared, kdd):
    return (1. + (2. * (kdd / q_squared)))

def prop_2(q_squared, t, kdd):
    return ((-2. * (kdd / q_squared)) + (t / q_squared))
    
#################################################################################
# We are now prepared to actually *make* all the required directories!
#################################################################################

main_experimental_datafile = pd.read_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/experimental_data_v{MAJOR_MINOR_NUMBER}.csv")

for kinematic_set_number, kinematic_group in main_experimental_datafile.groupby('set'):

    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data", exist_ok = True)
    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots", exist_ok = True)
    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/learning_curves", exist_ok = True)
    os.makedirs(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas", exist_ok = True)
    
    set_data = main_experimental_datafile[main_experimental_datafile['set'] == kinematic_set_number]

    # if it's the same kinematic setting, then these mean()s should be the equivalence class value...
    fixed_k = kinematic_group['k'].mean()
    fixed_q_squared = kinematic_group['q_squared'].mean()
    fixed_x_bjorken = kinematic_group['x_b'].mean()
    fixed_t = kinematic_group['t'].mean()
    number_of_phi_points = kinematic_group['phi'].nunique()

    set_data.to_csv(
        f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data/main_experimental_file_v{MAJOR_MINOR_NUMBER}.csv", 
        index = False)
    
    print(f"[INFO]: Processed Set {kinematic_set_number}!")

    with open(
        file = f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/log_set_{kinematic_set_number}_v{MAJOR_MINOR_NUMBER}.txt",
        mode = "w",
        encoding = "utf-8",
        buffering = 1) as logfile:

        logfile.write(f"[INFO]: Kinematic Set Number: {kinematic_set_number}\n")
        logfile.write(f"[INFO]: k = {fixed_k}\n")
        logfile.write(f"[INFO]: Q^2 = {fixed_q_squared}\n")
        logfile.write(f"[INFO]: xb = {fixed_x_bjorken}\n")
        logfile.write(f"[INFO]: t = {fixed_t}\n")

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

        logfile.write(f"[INFO]: ep = {fixed_ep}\n")
        logfile.write(f"[INFO]: y = {fixed_y}\n")
        logfile.write(f"[INFO]: xi = {fixed_xi}\n")
        logfile.write(f"[INFO]: tmin = {fixed_t_min}\n")
        logfile.write(f"[INFO]: Ktilde = {fixed_k_tilde}\n")
        logfile.write(f"[INFO]: K = {fixed_big_k}\n")
        logfile.write(f"[INFO]: tprime = {fixed_t_prime}\n")
        logfile.write(f"[INFO]: FE = {fixed_fe}\n")
        logfile.write(f"[INFO]: FG = {fixed_fg}\n")
        logfile.write(f"[INFO]: F2 = {fixed_f2}\n")
        logfile.write(f"[INFO]: F1 = {fixed_f1}\n")

        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/data\n")
        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots\n")
        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/replicas\n")
        logfile.write(f"[INFO]: Made new directory at path: {SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/learning_curves\n")

        logfile.write(f"[INFO]: Each point has {number_of_phi_points} phi values\n")
        logfile.close()

print(f"[INFO]: End of script reached!")
