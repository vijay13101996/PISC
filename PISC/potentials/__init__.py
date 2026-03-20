import numpy as np

from PISC.potentials.double_well_potential import double_well
from PISC.potentials.TD_DW import td_dw
from PISC.potentials.Adams_function import adams_function
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.harmonic_1D import harmonic as harmonic1D
from PISC.potentials.harmonic_2D import Harmonic as harmonic2D
from PISC.potentials.harmonic_oblique_2D import Harmonic_oblique
from PISC.potentials.Four_well import four_well
from PISC.potentials.Heller_Davis import heller_davis
from PISC.potentials.Henon_Heiles import henon_heiles
from PISC.potentials.Matsubara_M3_Quartic import Matsubara_Quartic
from PISC.potentials.Morse_1D import morse
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.potentials.Quartic import quartic
from PISC.potentials.trunc_harmonic import trunc_harmonic
from PISC.potentials.asym_double_well_potential import asym_double_well
from PISC.potentials.mildly_anharmonic import mildly_anharmonic
from PISC.potentials.mildly_anharmonic_2D import mildly_anharmonic_2D
from PISC.potentials.inv_harmonic import InvHarmonic
from PISC.potentials.pot_adder import PotAdder
from PISC.potentials.Coupled_quartic import coupled_quartic
#from PISC.potentials.DW_Morse_harm import DW_Morse_harm
# from PISC.potentials.Tanimura_SB import Tanimura_SB
from PISC.potentials.triple_well_potential import triple_well
#from PISC.potentials.Morse_harm_2D import Morse_harm_2D
from PISC.potentials.Coupled_harmonic_oblique import coupled_harmonic_oblique


potentials = {
    "harmonic1D": harmonic1D,
    "mildly_anharmonic": mildly_anharmonic,
    "mildly_anharmonic_2D": mildly_anharmonic_2D,
    "double_well": double_well,
    "morse": morse,
    "quartic": quartic,
    "adams_function": adams_function,
    "coupled_harmonic": coupled_harmonic,
    "harmonic_2D": harmonic2D,
    "Harmonic_oblique": Harmonic_oblique,
    "four_well": four_well,
    "heller_davis": heller_davis,
    "henon_heiles": henon_heiles,
    "Matsubara_Quartic": Matsubara_Quartic,
    "quartic_bistable": quartic_bistable,
    "trunc_harmonic": trunc_harmonic,
    "asym_double_well": asym_double_well,
}


def check_pes_param(pes_param):
    """Checks the consistency of pes related paramenters"""
    dimension = pes_param["dimension"]
    name = pes_param["pes_name"]
    parameters = pes_param["pes_param"]
    if name not in potentials.keys():
        raise NotImplementedError("{} potential is not in implemented".format(name))

    if name == "harmonic1D":
        assert len(parameters) == 2, "{} requires 2 parameters (m, omega)".format(name)
        assert (
            dimension == 1
        ), "{} is a 1D model, but the dimension variable is set to {}".format(dimension)
        pes = potentials[name](parameters[0], parameters[1])
        pot_key = "{}_m_{}_omega_{}".format(name, parameters[0], parameters[1])
    elif name == "mildly_anharmonic":
        assert len(parameters) == 3, "{} requires 3 parameters (m,a,b)".format(name)
        assert (
            dimension == 1
        ), "{} is a 1D model, but the dimension variable is set to {}".format(dimension)
        pes = potentials[name](parameters[0], parameters[1], parameters[2])
        pot_key = "{}_m_{}_a_{}_b_{}".format(
            name, parameters[0], parameters[1], parameters[2]
        )
    elif name == "mildly_anharmonic_2D":
        assert len(parameters) == 8, "{} requires 8 parameters (m,w1,a1,b1,w2,a2,b2,c)".format(name)
        assert (
            dimension == 2
        ), "{} is a 2D model, but the dimension variable is set to {}".format(dimension)
        pes = potentials[name](parameters[0], parameters[1], parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7])
        pot_key = "{}_m_{}_a1_{}_b1_{}_a2_{}_b2_{}_c{}".format(
            name, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5]
        )
    elif name == "morse":
        assert len(parameters) == 2, "{} requires 2 parameters (D,alpha)".format(name)
        assert (
            dimension == 1
        ), "{} is a 1D model, but the dimension variable is set to {}".format(dimension)
        pes = potentials[name](parameters[0], parameters[1])
        pot_key = "{}_D_{}_alpha_{}".format(name, parameters[0], parameters[1])
    elif name == "double_well":
        assert len(parameters) == 2, "{} requires 2 parameters (lambda,g)".format(name)
        assert (
            dimension == 1
        ), "{} is a 1D model, but the dimension variable is set to {}".format(dimension)
        pes = potentials[name](parameters[0], parameters[1])
        pot_key = "{}_lambda_{}_g_{}".format(name, parameters[0], parameters[1])
    elif name == "Quartic":
        assert len(parameters) == 1, "{} requires 1 parameter (a)".format(name)
        assert (
            dimension == 1
        ), "{} is a 1D model, but the dimension variable is set to {}".format(dimension)
        pes = potentials[name](parameters[0])
        pot_key = "{}_a_{}".format(name, parameters[0])
    elif name == "harmonic_2D":
        assert len(parameters) == 1, "{} requires 1 parameter (omega)".format(name)
        assert (
            dimension == 2
        ), "{} is a 2D model, but the dimension variable is set to {}".format(dimension)
        pes = potentials[name](parameters[0])
        pot_key = "{}_omega_{}".format(name, parameters[0])
    else:
        raise NotImplementedError(
            "Please implement an argument  check for {} ".format(name)
        )

    return pes, pot_key
