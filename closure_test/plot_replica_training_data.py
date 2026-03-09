##########################################
# Finding Observable Values Labeled by
# Training/Validation/Testing
##########################################

# a numpy array of all the corresponding phi-points
# if you do NOT UNDERSTAND why the code below selects the phi-values that are used in training, please run each piece interactively
x_training_phi_points = np.array(x_training["phi"])
x_validation_phi_points = np.array(x_validation["phi"])
x_testing_phi_points = np.array(x_testing["phi"])

xsecs = DifferentialCrossSection(
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
        "using_ww": True
    },
    verbose = False,
    debugging = False)

x_training_bkm10_xsec = xsecs.compute_cross_section(
    x_training_phi_points,
    lepton_helicity = 0.0,
    target_polarization = 0.0).real

x_validation_bkm10_xsec = xsecs.compute_cross_section(
    x_validation_phi_points,
    lepton_helicity = 0.0,
    target_polarization = 0.0).real

x_testing_bkm10_xsec = xsecs.compute_cross_section(
    x_testing_phi_points,
    lepton_helicity = 0.0,
    target_polarization = 0.0).real

x_training_bkm10_bsa = xsecs.compute_bsa(
    x_training_phi_points,
    target_polarization = 0.0).real

x_validation_bkm10_bsa = xsecs.compute_bsa(
    x_validation_phi_points,
    target_polarization = 0.0).real

x_testing_bkm10_bsa = xsecs.compute_bsa(
    x_testing_phi_points,
    target_polarization = 0.0).real

title_string = (
    rf"$Q^2 = {FIXED_Q_SQUARED:.2f}$ GeV$^2$, "
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

data_vis_figure, data_vis_axis = plt.subplots(1, 1, figsize = (8, 7))
data_vis_axis.scatter(x_training_phi_points, x_training_bkm10_xsec, color = 'green', s = 4., label = rf"(KM15) Training Points $N = {len(x_training_phi_points)}$")
data_vis_axis.scatter(x_validation_phi_points, x_validation_bkm10_xsec, color = 'orange', s = 4., label = rf"(KM15) Validation Points $N = {len(x_validation_phi_points)}$")
data_vis_axis.scatter(x_testing_phi_points, x_testing_bkm10_xsec, color = 'red', s = 4., label = rf"(KM15) Testing Points $N = {len(x_testing_phi_points)}$")
data_vis_axis.legend(fontsize = 14.)
data_vis_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16.)
data_vis_axis.set_ylabel(r"$d^{4}\sigma$", fontsize = 16.)
data_vis_axis.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
data_vis_axis.grid(visible = True)
data_vis_figure.savefig(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots/training_set_xsec_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png")
data_vis_figure.savefig(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots/training_set_xsec_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(data_vis_figure)

bsa_train_test_split_figure, bsa_train_test_split_axis = plt.subplots(1, 1, figsize = (8, 7))
bsa_train_test_split_axis.scatter(x_training_phi_points, x_training_bkm10_bsa, color = 'green', s = 4., label = rf"(KM15) Training Points $N = {len(x_training_phi_points)}$")
bsa_train_test_split_axis.scatter(x_validation_phi_points, x_validation_bkm10_bsa, color = 'orange', s = 4., label = rf"(KM15) Validation Points $N = {len(x_validation_phi_points)}$")
bsa_train_test_split_axis.scatter(x_testing_phi_points, x_testing_bkm10_bsa, color = 'red', s = 4., label = rf"(KM15) Testing Points $N = {len(x_testing_phi_points)}$")
bsa_train_test_split_axis.legend(fontsize = 14.)
bsa_train_test_split_axis.set_xlabel(r"$\phi$ [radians]", fontsize = 16.)
bsa_train_test_split_axis.set_ylabel(r"BSA", fontsize = 16.)
bsa_train_test_split_axis.set_title(f"{title_string}\n(KM15): {km15_cff_string}")
bsa_train_test_split_axis.grid(visible = True)
bsa_train_test_split_figure.savefig(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots/training_set_bsa_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.png")
bsa_train_test_split_figure.savefig(f"{SCRATCH_PATH}/version_{MAJOR_MINOR_NUMBER}/kinematic_set_{kinematic_set_number}/plots/training_set_bsa_replica_{replica_number}_v{MAJOR_MINOR_NUMBER}.eps")
plt.close(bsa_train_test_split_figure)