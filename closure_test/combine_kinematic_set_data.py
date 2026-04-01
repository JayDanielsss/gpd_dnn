##########################################
# FILE INFORMATION:
# Purpose: generates a .csv file containing
# pseudodata for training DNN models down
# the line.
# Created: 20260308
# Last changed: 20260401
##########################################

print("[INFO]: Script began running!")

##########################################
# Importing Python Libraries
##########################################

import pandas as pd
import glob
import os

print("[INFO]: Libraries imported!")

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

file_pattern = os.path.join(
    SCRATCH_PATH, 
    f"version_{MAJOR_MINOR_NUMBER}", 
    "kinematic_set_*", 
    "data", 
    f"main_pseudodata_file_set_*_v{MAJOR_MINOR_NUMBER}.csv"
)

# use glob for the asterisk pattern above:
kinematic_set_files = glob.glob(file_pattern)

kinematic_set_pseudodata_list = [
    pd.read_csv(kinematic_set_pseudodata) for kinematic_set_pseudodata in kinematic_set_files
    ]

combined_df = pd.concat(kinematic_set_pseudodata_list, ignore_index = True)
combined_df.drop(columns = ['Unnamed: 0'], errors = 'ignore')

output_directory = os.path.join(SCRATCH_PATH,f"version_{MAJOR_MINOR_NUMBER}")
total_pseudodata_file = os.path.join(
    output_directory,
    f"combined_pseudodata_v{MAJOR_MINOR_NUMBER}.csv")
combined_df.to_csv(total_pseudodata_file, index = False)

print(f"[INFO]: Combined {len(kinematic_set_pseudodata_list)} files into {total_pseudodata_file}")

print("[INFO]: End of script reached!")
