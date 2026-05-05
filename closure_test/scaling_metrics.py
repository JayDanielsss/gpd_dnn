##########################################
# FILE INFORMATION:
# Purpose: makes plots of how simultaneous
# fits scale with varying parameters
# Created: 20260414
# Last changed: 20260415
##########################################

print("[INFO]: Script began running!")

#################################################################################
# Libraries
#################################################################################

# native libraries:
import datetime

# 3rd party libraries:
import matplotlib.pyplot as plt
import numpy as np

##########################################
# Matplotlib Plotting Customizability
##########################################

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['xtick.major.size'] = 8.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['ytick.major.size'] = 8.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['savefig.dpi'] = 300

#################################################################################
# Residuals vs. number of phi points
#################################################################################

residuals_vs_phi_figure, residuals_vs_phi_axis = plt.subplots(1, 1, figsize = (10, 9))

residuals_vs_phi_axis.scatter(
    np.array([360 + 1, 270 + 1, 180 + 1, 90 + 1, 45 + 1, 30 + 1]), # phi values
    np.array([ # make sure you remember where this comes from!
        0.4466762396400543 / (360 + 1),
        0.3866621844571099 / (270 + 1),
        0.2927690358326991 / (180 + 1),
        0.18239360909334892 / (90 + 1),
        0.12830471562822562 / (45 + 1),
        0.06698662700850821 / (30 + 1)
        ]),
    label = r"$d^{4}\sigma^{UU}$ $\frac{1}{N_{\phi}} \sum$ Residuals$^{2}$",
    color = 'orange'
)

residuals_vs_phi_axis.scatter(
    np.array([360 + 1, 270 + 1, 180 + 1, 90 + 1, 45 + 1, 30 + 1]), # phi values
    np.array([ # make sure you remember where this comes from!
        0.06753363563017946 / (360 + 1),
        0.05844831213130888 / (270 + 1),
        0.04102633496344951 / (180 + 1),
        0.018802095575724768 / (90 + 1),
        0.007979609612198702 / (45 + 1),
        0.04467551880247527 / (30 + 1)
        ]),
    label = r"\textrm{BSA} $\frac{1}{N_{\phi}} \sum$ Residuals$^{2}$",
    color = 'purple'
)

residuals_vs_phi_axis.set_xlabel(r"$N_{\phi}$ (number of phi points)", fontsize = 18.)
residuals_vs_phi_axis.set_ylabel(r"Residuals", fontsize = 18.)
residuals_vs_phi_axis.set_title(
    r"(Closure Test - No Error) Simultaneous Fit Residuals Trend with $N_{\phi}$",
    fontsize = 18.)

residuals_vs_phi_axis.legend(fontsize = 16.)

residuals_vs_phi_axis.grid(visible = False)
residuals_vs_phi_axis.text(
    0.00, -0.05,
    f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
    transform = residuals_vs_phi_axis.transAxes,
    verticalalignment = 'top',
    horizontalalignment = 'left',
    fontsize = 9.,
)

for extension in ['png', 'eps']:
    residuals_vs_phi_figure.savefig(
        fname = f"./extra_plots/residuals_no_error_vs_phi.{extension}",
        facecolor = 'white',
        transparent = False)
    
plt.close(residuals_vs_phi_figure)

del residuals_vs_phi_figure
del residuals_vs_phi_axis

#################################################################################
# Re[H] fits with Number of Phi
#################################################################################

cff_re_h_fits_vs_phi_figure, cff_re_h_fits_vs_phi_axis = plt.subplots(1, 1, figsize = (10, 9))

cff_re_h_fits_vs_phi_axis.errorbar(
    x = np.array([360 + 1, 270 + 1, 180 + 1, 90 + 1, 45 + 1, 30 + 1]),
    y = np.array([-5.870, -5.870, -5.846, -5.795, -5.824, -5.885]),
    yerr = np.array([0.001, 0.001, 0.247, 0.529, 0.447, 0.566]),
    label = r"Re[$\mathcal{H}$] Replica Average",
    color = 'red', capsize = 4, fmt = 'o', alpha = 0.6)

cff_re_h_fits_vs_phi_axis.scatter(
    np.array([360 + 1, 270 + 1, 180 + 1, 90 + 1, 45 + 1, 30 + 1]),
    np.array([-5.870, -5.870, -5.870, -5.870, -5.870, -5.870]),
    label = r"Re[$\mathcal{H}$] KM15 True Value",
    color = 'blue')

cff_re_h_fits_vs_phi_axis.set_xlabel(r"$N_{\phi}$ (number of phi points)", fontsize = 16.)
cff_re_h_fits_vs_phi_axis.set_ylabel(r"CFF Re[$\mathcal{H}$] Value", fontsize = 16.)
cff_re_h_fits_vs_phi_axis.set_title(
    r"(Closure Test - No Error) CFF Re[$\mathcal{H}$] Accuracy Trend with $N_{\phi}$",
    fontsize = 18.)

cff_re_h_fits_vs_phi_axis.legend(fontsize = 16.)

cff_re_h_fits_vs_phi_axis.grid(visible = False)
cff_re_h_fits_vs_phi_axis.text(
    0.00, -0.05,
    f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
    transform = cff_re_h_fits_vs_phi_axis.transAxes,
    verticalalignment = 'top',
    horizontalalignment = 'left',
    fontsize = 9.,
)

for extension in ['png', 'eps']:
    cff_re_h_fits_vs_phi_figure.savefig(
        fname = f"./extra_plots/cff_re_h_accuracy_no_error_vs_phi.{extension}",
        facecolor = 'white',
        transparent = False)

plt.close(cff_re_h_fits_vs_phi_figure)

del cff_re_h_fits_vs_phi_figure
del cff_re_h_fits_vs_phi_axis

#################################################################################
# Im[H] fits with Number of Phi
#################################################################################

cff_im_h_fits_vs_phi_figure, cff_im_h_fits_vs_phi_axis = plt.subplots(1, 1, figsize = (10, 9))

cff_im_h_fits_vs_phi_axis.errorbar(
    x = np.array([360 + 1, 270 + 1, 180 + 1, 90 + 1, 45 + 1, 30 + 1]),
    y = np.array([6.712, 6.712, 6.683, 6.623, 6.578, 6.099]),
    yerr = np.array([0.001, 0.001, 0.282, 0.617, 0.697, 1.343]),
    label = r"Im[$\mathcal{H}$] Replica Average",
    color = 'orange', capsize = 4, fmt = 'o', alpha = 0.6)

cff_im_h_fits_vs_phi_axis.scatter(
    np.array([360 + 1, 270 + 1, 180 + 1, 90 + 1, 45 + 1, 30 + 1]),
    np.array([6.712, 6.712, 6.712, 6.712, 6.712, 6.712]),
    label = r"Im[$\mathcal{H}$] KM15 True Value",
    color = 'blue')

cff_im_h_fits_vs_phi_axis.set_xlabel(r"$N_{\phi}$ (number of phi points)", fontsize = 16.)
cff_im_h_fits_vs_phi_axis.set_ylabel(r"CFF Im[$\mathcal{H}$] Value", fontsize = 16.)
cff_im_h_fits_vs_phi_axis.set_title(
    r"(Closure Test - No Error) CFF Im[$\mathcal{H}$] Accuracy Trend with $N_{\phi}$",
    fontsize = 18.)

cff_im_h_fits_vs_phi_axis.legend(fontsize = 16.)

cff_im_h_fits_vs_phi_axis.grid(visible = False)
cff_im_h_fits_vs_phi_axis.text(
    0.00, -0.05,
    f"Figure rendered {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", 
    transform = cff_im_h_fits_vs_phi_axis.transAxes,
    verticalalignment = 'top',
    horizontalalignment = 'left',
    fontsize = 9.,
)

for extension in ['png', 'eps']:
    cff_im_h_fits_vs_phi_figure.savefig(
        fname = f"./extra_plots/cff_im_h_accuracy_no_error_vs_phi.{extension}",
        facecolor = 'white',
        transparent = False)

plt.close(cff_im_h_fits_vs_phi_figure)

del cff_im_h_fits_vs_phi_figure
del cff_im_h_fits_vs_phi_axis

print("[INFO]: Script finished running!")

#### for error pseudodata

## for UNPOLARIZED CROSS-SECTION!!
np.array([360 + 1, 330 + 1, 300 + 1, 270 + 1, 240 + 1, 210 + 1, 180 + 1, 150 + 1, 120 + 1, 90 + 1, 60 + 1, 30 + 1, 15 + 1, 10 + 1]), # phi values
np.array([ # make sure you remember where this comes from!
    0.3960375303940763 / (360 + 1),
    0.35377598117899756 / (330 + 1),
    0.31322912155979943 / (300 + 1)
    ]),

## for BSA DATA!!!
np.array([360 + 1, 330 + 1, 300 + 1, 270 + 1, 240 + 1, 210 + 1, 180 + 1, 150 + 1, 120 + 1, 90 + 1, 60 + 1, 30 + 1, 15 + 1, 10 + 1]), # phi values
np.array([ # make sure you remember where this comes from!
    0.08434800603689552 / (360 + 1),
    0.09612863350808214 / (330 + 1),
    0.07006224233637139 / (300 + 1)
    ]),