##########################################
# FILE INFORMATION:
# Purpose: find any duplicate rows in the 
# original file and combine the observables
# along that row.
# Created: 20260504
# Last changed: 20260504
##########################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

import pandas as pd

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
# Define the static variables:
#################################################################################

kinematic_set_columns = ['set', 'k', 'x_b', 'q_squared', 't', 'phi']

observable_columnnames = [
    "unp_beam_unp_target_xsec", "unp_beam_unp_target_xsec_err", "unp_beam_unp_target_xsec_errsyst", "unp_beam_unp_target_xsec_errstat",
    "unp_target_bsa", "unp_target_bsa_err", "unp_target_bsa_errsyst", "unp_target_bsa_errstat", 
    "unp_target_bca", "unp_target_bca_err", "unp_target_bca_errsyst", "unp_target_bca_errstat", 
    "unp_target_lp_target_xsec", "unp_target_lp_target_xsec_err", "unp_target_lp_target_xsec_errsyst", "unp_target_lp_target_xsec_errstat", 
    "unp_target_tp_target_xsec", "unp_target_tp_target_xsec_err", "unp_target_tp_target_xsec_errsyst", "unp_target_tp_target_xsec_errstat", 
    "unp_target_xgamma", "unp_target_xgamma_err", "unp_target_xgamma_errsyst", "unp_target_xgamma_errstat", 
    ]

#################################################################################
# Read the two files:
#################################################################################

df_experimental_data = pd.read_csv(
    f'{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/main_experimental_file_v{MAJOR_MINOR_NUMBER}.csv')

df_length_before_merge = len(df_experimental_data)

#################################################################################
# Begin the logic of combining redundant data:
#################################################################################

# group DF along set and phi to find duplicates...
set_phi_tally_values = df_experimental_data.groupby(['set', 'phi']).size()
duplicate_indices = set_phi_tally_values[set_phi_tally_values > 1].index

if duplicate_indices.empty:
    print("[INFO]: No duplicate rows for (set, phi) found. Everything looks clean.")
else:
    print(f"[INFO]: Found {len(duplicate_indices)} instances where a (set, phi) point is split.")

for set_index, phi_value in duplicate_indices:
    print(f"[INFO]: Examining set = {set_index}, phi = {phi_value}")
    fragmented_rows = df_experimental_data[(df_experimental_data['set'] == set_index) & (df_experimental_data['phi'] == phi_value)]

    for observable in observable_columnnames:
        values = fragmented_rows[observable].values
        has_zero = 0 in values
        has_nonzero = any(value != 0 for value in values)
        
        if has_zero and has_nonzero:
            print(f"[INFO]: Found mergeable data in '{observable}': {values}")
        elif all(value == 0 for value in values):
            print(f"[WARN]: '{observable}' is zero in all rows for this point.")

other_columns = [
    column for column in df_experimental_data.columns if column not in kinematic_set_columns and column not in observable_columnnames
    ]

# this is required for Pandas to use agg()
aggregation_logic = { column: 'sum' for column in observable_columnnames }

for column in other_columns:
    aggregation_logic[column] = 'first'

merged_df_experimental_data = df_experimental_data.groupby(kinematic_set_columns, as_index = False).agg(aggregation_logic)

df_length_after_merge = len(merged_df_experimental_data)

print(f"[INFO]: Rows reduced from {df_length_before_merge} to {df_length_after_merge} (Removed {df_length_before_merge - df_length_after_merge} redundancies).")

merged_df_experimental_data.to_csv(
    f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/data/combined_experimental_file_v{MAJOR_MINOR_NUMBER}.csv", 
    index = False)

print("[INFO]: End of script reached!")
