"""Central hub for BKM10 functions."""

import numpy as np

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

km15_cross_section = DifferentialCrossSection(
    configuration = {
        "kinematics": BKM10Inputs(
            lab_kinematics_k = TEST_BEAM_ENERGY,
            squared_Q_momentum_transfer = TEST_Q_SQUARED,
            x_Bjorken = TEST_X_BJORKEN,
            squared_hadronic_momentum_transfer_t = TEST_T_VALUE),
        "cff_inputs": CFFInputs(
            compton_form_factor_h = CFF_H_KM15,
            compton_form_factor_h_tilde = CFF_H_TILDE_KM15,
            compton_form_factor_e = CFF_E_KM15,
            compton_form_factor_e_tilde = CFF_E_TILDE_KM15),
        "using_ww": True
    },
    verbose = False,
    debugging = False)