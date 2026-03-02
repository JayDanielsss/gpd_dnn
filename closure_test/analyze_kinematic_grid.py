##########################################
# FILE INFORMATION:
# Purpose: in case `kinematic_grid.py` does
# not finish running entirely, we run this
# analysis script to extract the data from
# each kinematic setting anyway.
# Created: 20260302
# Last changed: 20260302
##########################################

print(f"[INFO]: Script began running!")

##########################################
# Importing Python Libraries
##########################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print(f"[INFO]: Libraries imported!")

##########################################
# [IMPORTANT]: Static quantities parametrizing
# the program. Change these if you need!
##########################################

VERSION_NUMBER = 1
MINOR_NUMBER = 1
MAJOR_MINOR_NUMBER = f"{VERSION_NUMBER}_{MINOR_NUMBER}"

print(f"[INFO]: We are saving figures and data with the following appendage: {MAJOR_MINOR_NUMBER}")

# read the dataframe:
test_dataframe = pd.read_csv(
        filepath_or_buffer = f"./kinematic_grid_data_v{VERSION_NUMBER}.csv"
    )

unique_kinematic_sets_dataframe = test_dataframe.groupby(['set'], as_index = False).mean()
number_of_unique_kinematic_sets = len(unique_kinematic_sets_dataframe)

print(f"[INFO]: Total number of unique sets: {number_of_unique_kinematic_sets}")
# if the assert below is False, it means some settings were unphysical --- that's all!

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

input_space_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/kinematic_settings_space_v{MAJOR_MINOR_NUMBER}.png")
input_space_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/kinematic_settings_space_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(input_space_figure)

t_vs_xb_figure, t_vs_x_axis = plt.subplots(1, 1, figsize = (8, 7))
t_vs_x_axis.scatter(xb_column, t_column, alpha = 0.3, s = 10.4)
t_vs_x_axis.set_xlabel(r"$x_{\textrm{B}}$")
t_vs_x_axis.set_ylabel(r"$-t$")
t_vs_x_axis.set_title(r"$-t$ vs. $x_{\textrm{B}}$")

t_vs_xb_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/t_vs_xb_v{MAJOR_MINOR_NUMBER}.png")
t_vs_xb_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/t_vs_xb_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(t_vs_xb_figure)

qsq_vs_t_figure, qsq_vs_t_axis = plt.subplots(1, 1, figsize = (8, 7))
qsq_vs_t_axis.scatter(q_squared_column, t_column, alpha = 0.1, s = 10.4)
qsq_vs_t_axis.set_xlabel(r"$Q^{2}$")
qsq_vs_t_axis.set_ylabel(r"$-t$")
qsq_vs_t_axis.set_title(r"$-t$ vs. $Q^{2}$")

qsq_vs_t_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/qsquared_vs_t_v{MAJOR_MINOR_NUMBER}.png")
qsq_vs_t_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/qsquared_vs_t_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(qsq_vs_t_figure)

xb_vs_qsq_figure, xb_vs_qsq_axis = plt.subplots(1, 1, figsize = (8, 7))
xb_vs_qsq_axis.scatter(xb_column, q_squared_column, alpha = 0.1, s = 10.4)
xb_vs_qsq_axis.set_xlabel(r"$x_{\textrm{B}}$")
xb_vs_qsq_axis.set_ylabel(r"$Q^{2}$")
xb_vs_qsq_axis.set_title(r"$Q^{2}$ vs. $x_{\textrm{B}}$")

xb_vs_qsq_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/xb_vs_qsquared_v{MAJOR_MINOR_NUMBER}.png")
xb_vs_qsq_figure.savefig(f"./version_{MAJOR_MINOR_NUMBER}/plots/xb_vs_qsquared_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(xb_vs_qsq_figure)

print(f"[INFO]: End of script reached!")
