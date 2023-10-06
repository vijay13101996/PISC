# from matplotlib import pyplot as plt
# import multiprocessing as mp
import argparse
import numpy as np
import time
import os

from functools import partial
from PISC.utils.mptools import chunks
from PISC.utils.mptools import batching
from PISC.engine.PI_sim_core import SimUniverse

test=True
def main(system_param, pes_param, ensemble_param, simulation_param):
    path = os.path.dirname(os.path.abspath(__file__))

    # pes  = check_pes_param(pes_param)
    # check_ensemble_param(ensemble_param)
    # check_simulation_param(simulation_param)

    if test:
        times = 20
        lamda = 2.0
        g = 0.02  # 8
        Tc = 0.5 * lamda / np.pi
        times = 20.0
        ensemble_param["temperature"] = times * Tc
        from PISC.potentials.double_well_potential import double_well
        pes = double_well(lamda, g)
        potkey = "inv_harmonic_lambda_{}_g_{}".format(lamda, g)
        Tkey = "T_{}Tc".format(times)  #'{}Tc'.format(times)

    Sim_class = SimUniverse(
        simulation_param["method"],
        path,
        system_param["sys_name"],
        potkey,
        simulation_param["CFtype"],
        ensemble_param["ensemble"],
        Tkey,
    )
    Sim_class.set_sysparams(
        pes, ensemble_param["temperature"], system_param["mass"], system_param["dimension"]
    )
    Sim_class.set_simparams(
        simulation_param["n_traj"],
        simulation_param["dt_therma"],
        simulation_param["dt"],
    )
    Sim_class.set_methodparams()
    Sim_class.set_ensparams(ensemble_param["tau"])
    Sim_class.set_runtime(
        simulation_param["time_therma"], simulation_param["time_total"]
    )

    start_time = time.time()
    func = partial(Sim_class.run_seed)
    seeds = range(100)
    seed_split = chunks(seeds, 10)

    with open("{}/Datafiles/input_log_{}.txt".format(path, potkey), "a") as f:
        f.write("\n" + str(system_param))
        f.write("\n" + str(pes_param))
        f.write("\n" + str(ensemble_param))
        f.write("\n" + str(simulation_param))

    batching(func, seed_split, max_time=1e6)
    print("time", time.time() - start_time)
    print("Have a nice day")


parser = argparse.ArgumentParser(description="""Master script run  simulations.\n""")

# ------ System definition --------------#
parser.add_argument(
    "-dim",
    "--dimension",
    type=int,
    required=True,
    choices=[1, 2],
    help="Dimensionality of the system",
)
parser.add_argument(
    "-pes",
    "--pes",
    type=str,
    required=True,
    choices=["double_well"],
    help="Potential energy surface",
)
parser.add_argument(
    "-m", "--mass", type=float, required=True, help="mass in atomic units (ALBERTO)"
)
parser.add_argument("-sys_name", "--sys_name", type=str, required=True, help="ALBERTO")

# ------- Method ------------#
parser.add_argument(
    "-method",
    "--method",
    type=str,
    required=True,
    choices=["Classical"],
    help="Type of simulation to be performed",
)
parser.add_argument(
    "-nbeads",
    "--nbeads",
    type=int,
    required=False,
    default=1,
    help="Number of beads",
)

# ------- Ensemble --------------------------------------------#
parser.add_argument(
    "-ens",
    "--ensemble",
    type=str,
    required=True,
    choices=["thermal"],
    help="Ensemble",
)
parser.add_argument(
    "-temp",
    "--temp",
    type=float,
    required=True,
    help="Temperature in atomic units (ALBERTO)",
)
parser.add_argument(
    "-temp_tau",
    "--temp_tau",
    type=float,
    required=True,
    help="Time constant of the langeving thermostat in atomic units (ALBERTO)",
)

# ------------ Simulation parameters --------------------#
parser.add_argument(
    "-dt", "--dt", type=float, required=True, help="Time step in atomic units (ALBERTO)"
)
parser.add_argument(
    "-dt_therma",
    "--dt_therma",
    type=float,
    required=True,
    help="Time step for thermalization in atomic units (ALBERTO)",
)
parser.add_argument(
    "-time_therma",
    "--time_therma",
    type=float,
    default=-1.0,
    help="Thermalization simulation time in atomic units (ALBERTO)",
)
parser.add_argument(
    "-time_total",
    "--time_total",
    type=float,
    required=True,
    help="Production simulation time in atomic units (ALBERTO)",
)
parser.add_argument(
    "-n_traj",
    "--n_traj",
    type=int,
    required=True,
    help="Number of parallel trajectories",
)

# Target quantity
parser.add_argument(
    "-corr_func",
    "--corr_func",
    type=str,
    required=True,
    choices=["OTOC"],
    help="Correlation function to be computed",
)

args = parser.parse_args()

system_param = {
    "sys_name": args.sys_name,
    "dimension": args.dimension,
    "mass": args.mass,
}

pes_param = {"dimension": args.dimension, "pes_name": args.pes}

ensemble_param = {
    "ensemble": args.ensemble,
    "temperature": args.temp,
    "tau": args.temp_tau,
}

simulation_param = {
    "method": args.method,
    "nbeads": args.nbeads,
    "n_traj": args.n_traj,
    "time_therma": args.time_therma,
    "time_total": args.time_total,
    "dt": args.dt,
    "dt_therma": args.dt_therma,
    "CFtype": args.corr_func,
}

# ALBERTO ADD gamma, seeds, chunks
# ALBERTO ADD pess parameters, 
# ALBERTO: create Datafile if doesn't exist
# ALBERTO: add R2, qq_tcf in choices   

main(system_param, pes_param, ensemble_param, simulation_param)
