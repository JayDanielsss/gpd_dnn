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

residuals_vs_phi_axis.plot(
    # N_{phi} --- number of phi points per observable!
    np.array([360 + 1, 330 + 1, 300 + 1, 270 + 1, 240 + 1, 210 + 1, 180 + 1, 150 + 1, 120 + 1, 90 + 1, 60 + 1, 30 + 1, 15 + 1, 10 + 1]),
    # reduced chi^{2} values --- numerator is chi^{2}, entire expression is reduced chi^{2}
    np.array([
        0.44669239773223424 / (360 + 1),
        0.4276730487848443 / (330 + 1),
        0.4077751510211567 / (300 + 1),
        0.3868545147633446 / (270 + 1),
        0.3647311027200774 / (240 + 1),
        0.341178824839648 / (210 + 1),
        0.3158763215376582 / (180 + 1),
        0.2883538543181037 / (150 + 1),
        0.2579251457687367 / (120 + 1),
        0.22337648878188404 / (90 + 1),
        0.1824066677579252 / (60 + 1),
        0.1290164627004284 / (30 + 1),
        0.0782480769819978 / (15 + 1),
        0.026071161064779507 / (10 + 1),
    ]),
    marker = 'o', linestyle = '--', alpha = 0.5, color = "orange",
    label = r"$d^{4}\sigma^{UU}$ $\frac{1}{N_{\phi}} \sum$ Residuals$^{2}$"
)

residuals_vs_phi_axis.plot(
    # N_{phi} --- number of phi points per observable!
    np.array([360 + 1, 330 + 1, 300 + 1, 270 + 1, 240 + 1, 210 + 1, 180 + 1, 150 + 1, 120 + 1, 90 + 1, 60 + 1, 30 + 1, 15 + 1, 10 + 1]),
    # reduced chi^{2} values --- numerator is chi^{2}, entire expression is reduced chi^{2}
    np.array([
        0.06759701634515794 / (360 + 1),
        0.0647192252598918 / (330 + 1),
        0.06170819981214945 / (300 + 1),
        0.05854050160340848 / (270 + 1),
        0.05519164888540696 / (240 + 1),
        0.05162834711456606 / (210 + 1),
        0.04779762789554262 / (180 + 1),
        0.04363432710014461 / (150 + 1),
        0.03902670120388904 / (120 + 1),
        0.033798130933022155 / (90 + 1),
        0.0275960209499296 / (60 + 1),
        0.019513297778208805 / (30 + 1),
        0.004419572219014582 / (15 + 1),
        0.03143422193597916 / (10 + 1)
        ]),
    marker = 'o', linestyle = '--', alpha = 0.5, color = 'purple',
    label = r"\textrm{BSA} $\frac{1}{N_{\phi}} \sum$ Residuals$^{2}$",
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

_BURNER_PHI_RANGE = np.linspace(0, 360, 361)

cff_re_h_fits_vs_phi_figure, cff_re_h_fits_vs_phi_axis = plt.subplots(1, 1, figsize = (10, 9))

cff_re_h_fits_vs_phi_axis.errorbar(
    # N_{phi} --- number of phi points per observable!
    x = np.array([360 + 1, 330 + 1, 300 + 1, 270 + 1, 240 + 1, 210 + 1, 180 + 1, 150 + 1, 120 + 1, 90 + 1, 60 + 1, 30 + 1, 15 + 1, 10 + 1]),
    # this was the *replica mean value* of Re[H]:
    y = np.array([
        -5.870304592, -5.870303846, -5.870303144000001, -5.870305162000001, -5.870307656, -5.870304358000001, -5.870305558, 
        -5.870301137999999, -5.870306536, -5.870303801086957, -5.870305274000001, -5.8703047250000004, -5.856491877, -5.821318186999998
        ]),
    # this was the *replica 1-sigma value* of Re[H]:
    yerr = np.array([
        4.630025485898249e-06, 5.71944787539117e-06, 6.0212510327789405e-06, 6.5372437616981225e-06, 6.8550028445953195e-06, 5.869994548542529e-06, 
        7.2051256754177235e-06, 8.092153977724048e-06, 7.255418940331929e-06, 9.258891080249337e-06, 1.1452839123983363e-05, 1.3459460427536122e-05,
        0.05802449436068501, 0.25084951133522093
    ]),
    label = r"Re[$\mathcal{H}$] Replica Average",
    color = 'red', capsize = 4, fmt = 'o', alpha = 0.6)

cff_re_h_fits_vs_phi_axis.plot(
    # this is the KM15 true value:
    _BURNER_PHI_RANGE, np.full_like(_BURNER_PHI_RANGE, -5.870),
    label = r"Re[$\mathcal{H}$] KM15 True Value",
    color = 'blue'
)

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
    # N_{phi} --- number of phi points per observable!
    x = np.array([360 + 1, 330 + 1, 300 + 1, 270 + 1, 240 + 1, 210 + 1, 180 + 1, 150 + 1, 120 + 1, 90 + 1, 60 + 1, 30 + 1, 15 + 1, 10 + 1]),
    # this was the *replica mean value* of Im[H]:
    y = np.array([
        6.711987729000001, 6.7119875360000005, 6.711990064, 6.7119875360000005, 6.711983474, 6.711988638000001, 6.711984837999999,
        6.711990277, 6.71198566, 6.711984754347827, 6.711985178, 6.711984033, 6.575240028999999, 6.0003537676
        ]),
    # this was the *replica 1-sigma value* of Im[H]:
    yerr = np.array([
        9.094364133886702e-06, 1.068736188218006e-05, 1.1743836851738879e-05, 1.068736188218006e-05, 1.119433445995629e-05, 9.354226638248969e-06,
        1.2147590543024508e-05, 1.2048641873680377e-05, 1.3073584053395403e-05, 1.893147490203912e-05, 1.9887702632572043e-05, 3.1395503037893695e-05,
        0.729956227990608, 1.5821107929009497
    ]),
    label = r"Im[$\mathcal{H}$] Replica Average",
    color = 'orange', capsize = 4, fmt = 'o', alpha = 0.6)

cff_im_h_fits_vs_phi_axis.plot(
    # this is the KM15 true value:
    _BURNER_PHI_RANGE, np.full_like(_BURNER_PHI_RANGE, 6.712),
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