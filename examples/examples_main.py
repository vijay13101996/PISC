import argparse
import numpy as np
import pathlib as Path
import time

from functools import partial
from PISC.utils.mptools import chunks
from PISC.utils.mptools import batching
from PISC.engine.PI_sim_core import SimUniverse
from PISC.engine.PI_sim_core import check_parameters
from PISC.potentials import check_pes_param


def main(system_param, pes_param, ensemble_param, simulation_param):
    path = Path.Path(__file__).parent.resolve()
    pes, pot_key = check_pes_param(pes_param)
    check_parameters(simulation_param, ensemble_param)
    Tkey = "T_{}K".format(ensemble_param["temperature"])

    Sim_class = SimUniverse(
        simulation_param["method"],
        str(path),
        system_param["sys_name"],
        pot_key,
        simulation_param["CFtype"],
        ensemble_param["ensemble"],
        Tkey,
        folder_name=simulation_param["folder_name"],
    )
    Sim_class.set_sysparams(
        pes,
        ensemble_param["temperature"],
        system_param["mass"],
        system_param["dimension"],
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
    seeds = range(simulation_param["nseeds"])
    seed_split = chunks(seeds, simulation_param["chunk_size"])

    output_folder = path / simulation_param["folder_name"]
    Path.Path(output_folder).mkdir(parents=True, exist_ok=True)
    with open(output_folder / "input_log_{}.txt".format(pot_key), "w") as f:
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
    "-m", "--mass", type=float, required=True, help="mass in atomic units"
)
parser.add_argument("-sys_name", "--sys_name", type=str, required=False, default='Selene', help="System name (Legacy flag)")

# ------- Method ------------#
parser.add_argument(
    "-method",
    "--method",
    type=str,
    required=True,
    choices=["Classical", "cmd", "rpmd"],
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
parser.add_argument(
    "-cmd_gamma",
    "--cmd_gamma",
    type=float,
    required=False,
    help="Parameter that determines the adiabatic separation for cmd calculations",
)
parser.add_argument(
    "-pile_lambda",
    "--pile_lambda",
    type=float,
    required=False,
    help="Lambda parameter for PILE thermostats  (see Eq. 30 in J. Chem. Phys. 140, 234116 (2014))",
)

# ------- Ensemble --------------------------------------------#
parser.add_argument(
    "-ens",
    "--ensemble",
    type=str,
    required=True,
    choices=["thermal", "mc"],
    help="Ensemble. 'Thermal' refers to canonical, 'mc' refers to microcanonical",
)

parser.add_argument(
    "-temp",
    "--temp",
    type=float,
    required=False,
    default=-1.0,
    help="Temperature in atomic units ",
)
parser.add_argument(
    "-temp_tau",
    "--temp_tau",
    type=float,
    required=False,
    default=-1.0,
    help="Time constant of the Langevin thermostat in atomic units",
)
parser.add_argument(
    "-energy",
    "--energy_mc",
    type=float,
    required=False,
    help="Energy (needed for microcanonical ensemble)",
)

# ------------ Simulation parameters --------------------#
parser.add_argument(
    "-dt", "--dt", type=float, required=True, help="Time step in atomic units "
)
parser.add_argument(
    "-dt_therma",
    "--dt_therma",
    type=float,
    required=True,
    help="Time step for thermalization in atomic units ",
)
parser.add_argument(
    "-time_therma",
    "--time_therma",
    type=float,
    default=-1.0,
    help="Thermalization simulation time in atomic units",
)
parser.add_argument(
    "-time_total",
    "--time_total",
    type=float,
    required=True,
    help="Production simulation time in atomic units",
)
parser.add_argument(
    "-n_traj",
    "--n_traj",
    type=int,
    required=True,
    help="Number of parallel trajectories",
)
parser.add_argument(
    "-nseeds",
    "--nseeds",
    type=int,
    required=True,
    help="Number of seeds",
)
parser.add_argument(
    "-chunk",
    "--chunk_size",
    type=int,
    required=True,
    help="Number of seeds simulated at the same time",
)
parser.add_argument(
    "-folder",
    "--folder_name",
    type=str,
    required=True,
    default="DataFile",
    help="Name of the folder to write the simulation output",
)
parser.add_argument(
    "-label",
    "--sim_label",
    type=str,
    required=True,
    help="Simulation label",
)

parser.add_argument(
    "-pes_param",
    "--pes_parameters",
    nargs="+",
    default=None,
    help="Parameters needed for the evalution of the PES",
)

# ----------- Target quantity --------------------
parser.add_argument(
    "-corr_func",
    "--corr_func",
    type=str,
    required=True,
    choices=[
        "qq_TCF",
        "pp_TCF",
        "qq2_TCF",
        "pp2_TCF",
        "qp_TCF",
        "pq_TCF",
        "OTOC",
        "R2",
    ],
    help="Correlation function to be computed",
)

args = parser.parse_args()

system_param = {
    "sys_name": args.sys_name,
    "dimension": args.dimension,
    "mass": args.mass,
}
aux = list(map(float, args.pes_parameters))
pes_param = {"dimension": args.dimension, "pes_name": args.pes, "pes_param": aux}

ensemble_param = {
    "ensemble": args.ensemble,
    "temperature": args.temp,
    "tau": args.temp_tau,
    "energy_mc": args.energy_mc,
}

simulation_param = {
    "method": args.method,
    "nbeads": args.nbeads,
    "cmd_gamma": args.cmd_gamma,
    "pile_lambda": args.pile_lambda,
    "n_traj": args.n_traj,
    "time_therma": args.time_therma,
    "time_total": args.time_total,
    "dt": args.dt,
    "dt_therma": args.dt_therma,
    "CFtype": args.corr_func,
    "nseeds": args.nseeds,
    "chunk_size": args.chunk_size,
    "folder_name": args.folder_name,
    "simulation_label": args.sim_label,
}

main(system_param, pes_param, ensemble_param, simulation_param)