import numpy as np
from PISC.engine.integrators import Symplectic
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.utils.readwrite import (
    store_1D_plotdata,
    store_2D_imagedata,
    store_3D_imagedata,
    read_arr,
)
from PISC.utils.tcf_fft import gen_tcf, gen_2pt_tcf, gen_R2_tcf, gen_R3_tcf
from PISC.engine.thermalize_PILE_L import thermalize_rp
from PISC.engine.gen_stable_manifold import generate_stable_manifold_rp
from PISC.engine.gen_mc_ensemble import generate_rp
from PISC.engine.gen_const_qp_ensemble import thermalize_rp_const_qp
from PISC.utils.time_order import reorder_time

debug = False


class SimUniverse(object):
    """
    Base class for the simulation universe. It contains all the parameters
    required to run path-integral simulations. The class also contains the methods to
    generate the ensembles and run the simulations.

    Parameters:
    method          : 'classical', 'RPMD', 'CMD' (so far)
    pathname        : Path to the folder where the data is stored
    sysname         : Name of the workstation/cluster where the simulation is run
    potkey          : Name of the potential energy surface
    corrkey         : Name of the correlation function to be computed
    enskey          : Name of the ensemble to be used
    Tkey            : Code for the temperature to be used (e.g. T=xTc)
    ext_kwlist      : List of external parameters to be used (e.g. E)
    folder_name     : Name of the folder where the data is stored (duplicate)
    symplectic_order: Order of the symplectic integrator
    """
    def __init__(
        self,
        method,
        pathname,
        sysname,
        potkey,
        corrkey,
        enskey,
        Tkey,
        ext_kwlist=None,
        folder_name="Datafiles",
        symplectic_order=None,
    ):
        self.method = method
        self.pathname = pathname
        self.sysname = sysname
        self.potkey = potkey
        self.corrkey = corrkey
        self.enskey = enskey
        self.Tkey = Tkey
        self.ext_kwlist = ext_kwlist
        self.folder_name = folder_name
        if symplectic_order is None: 
            if self.corrkey == "OTOC":
                symplectic_order = 4
            else:
                symplectic_order = 2
        self.symplectic_order = symplectic_order

    def set_sysparams(self, pes, T, mass, dim):
        """
        Set system parameters
        pes   : potential energy surface
        T     : temperature
        mass  : particle mass
        dim   : dimensionality of the classical system
        """

        self.pes = pes
        self.T = T
        self.beta = 1.0 / self.T
        self.m = mass
        self.dim = dim

    def set_simparams(
        self, N, dt_ens=1e-2, dt=5e-3, extparam=None, coordinate_list=None, 
        pes_fort=False, propa_fort=False,transf_fort=False):
        """
        Set simulation parameters
        N               : Number of trajectories
        dt_ens          : Time step for the ensemble propagation
        dt              : Time step for the 'production' step
        extparam        : External parameter (e.g. E)
        coordinate_list : List of coordinates to be used for the 2D simulations
        pes_fort        : Boolean to specify whether the Fortran code is used for the PES
        propa_fort      : Boolean to specify whether the Fortran code is used for the propagator
        transf_fort      : Boolean to specify whether the Fortran code is used for the Cartesian/
                          Matsubara mode transformation
        
        The respective fortran subroutines for
        1. B, b, A, O and M steps of the propagator are called when propa_fort=True
            a. For the O step, the thalfstep function in thermostat.py is called from fortran; 
                the random number generator is passed from python to the fortran code
            b. In the M step, the matrix multiplication of the stability matrix with the hessian 
                is done in fortran
        2. potential, dpotential and ddpotential functions of the PES are called when pes_fort=True
        3. Transformation functions cart2mats_hess and mats2cart_hess (only the
           hessian part of the transformation is implemented in Fortran, transforming
           q and p is rather quick in Python with scipy's fft function) are called when transf_fort=True

        This is potentially all the avenues to speed up the code.
        """
        self.N = N
        self.dt_ens = dt_ens
        self.dt = dt
        if extparam is not None:
            self.extparam = extparam

        if coordinate_list is not None:
            self.coord_list = coordinate_list

        self.pes_fort = pes_fort
        self.propa_fort = propa_fort
        self.transf_fort = transf_fort

    def set_methodparams(self, nbeads=1, gamma=1):
        """
        Set method-specific parameters
        nbeads : Number of beads
        gamma  : Adiabaticity parameter for CMD
        """
        if self.method == "Classical" or self.method == "classical":
            self.nbeads = 1
        else:
            self.nbeads = nbeads
        if self.method == "CMD" or self.method == "cmd":
            self.gamma = gamma

    def set_runtime(self, time_ens=100.0, time_run=5.0):
        """ Set time to equilibrate ensemble and time to run the simulation """
        self.time_ens = time_ens
        self.time_run = time_run

    def set_ensparams(
        self,
        tau0=1.0,
        pile_lambda=100.0,
        E=None,
        qlist=None,
        plist=None,
        filt_func=None,
        Am=0.0,
        lamda=1.0,
        Q0 = 0.0,
        P0 = 0.0
    ):
        """
        Set ensemble parameters
        tau0        : PILE_L thermostat parameter
        pile_lambda : PILE_L thermostat parameter
        E           : Energy of the microcanonical ensemble
        qlist       : List of positions to initialize the RP ensemble
        plist       : List of momenta to initialize the RP ensemble
        filt_func   : Filter function to be used for 'filtering' ensemble
        lamda       : Parameter that defines the stable manifold (around a saddle point)
        Am          : Position along the stable manifold where the initial conditions are generated
        Q0          : Centroid position (for const_qp ensemble)
        P0          : Centroid momentum (for const_qp ensemble)
        """
        self.tau0 = tau0
        self.pile_lambda = pile_lambda
        self.E = E
        self.qlist = qlist
        self.plist = plist
        self.filt_func = filt_func
        self.lamda = lamda
        self.Am = Am
        self.Q0 = Q0
        self.P0 = P0

    def gen_ensemble(self, ens, rng, rngSeed):
        """
        Generate the ensembles and store it in pickle files.
        (Currently only canonical and microcanonical ensembles are implemented)
        """
        if self.enskey == "thermal":
            thermalize_rp(
                self.pathname,
                self.m,
                self.dim,
                self.N,
                self.nbeads,
                ens,
                self.pes,
                rng,
                self.time_ens,
                self.dt_ens,
                self.potkey,
                rngSeed,
                self.qlist,
                self.tau0,
                self.pile_lambda,
                propa_fort=self.propa_fort,
                pes_fort=self.pes_fort,
                transf_fort=self.transf_fort,
                folder_name=self.folder_name,
            )
            qcart = read_arr(
                "Thermalized_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                    self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                ),
                "{}/{}".format(self.pathname, self.folder_name),
            )
            pcart = read_arr(
                "Thermalized_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                    self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                ),
                "{}/{}".format(self.pathname, self.folder_name),
            )
            return qcart, pcart

        elif self.enskey == "stable_manifold":
            generate_stable_manifold_rp(
                self.pathname,
                self.m,
                self.dim,
                self.N,
                self.nbeads,
                ens,
                self.pes,
                rng,
                self.time_ens,
                self.dt_ens,
                self.potkey,
                rngSeed,
                Am=self.Am,
                lamda=self.lamda,
                qlist=self.qlist,
                tau0=self.tau0,
                pile_lambda=self.pile_lambda,
                folder_name=self.folder_name,
                store_thermalization=True,
                propa_fort=self.propa_fort,
                pes_fort=self.pes_fort,
                transf_fort=self.transf_fort,
            )
            qcart = read_arr(
                "Stable_manifold_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                    self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                ),
                "{}/{}".format(self.pathname, self.folder_name),
            )
            pcart = read_arr(
                "Stable_manifold_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                    self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                ),
                "{}/{}".format(self.pathname, self.folder_name),
            )
            return qcart, pcart

        elif self.enskey == "mc":
            # Fortran mode to be enabled here
            generate_rp(
                self.pathname,
                self.m,
                self.dim,
                self.N,
                self.nbeads,
                ens,
                self.pes,
                rng,
                self.time_ens,
                self.dt_ens,
                self.potkey,
                rngSeed,
                self.E,
                self.qlist,
                self.plist,
                self.filt_func,
                folder_name=self.folder_name,
            )
            qcart = read_arr(
                "Microcanonical_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                    self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                ),
                "{}/{}".format(self.pathname, self.folder_name),
            )
            pcart = read_arr(
                "Microcanonical_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                    self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                ),
                "{}/{}".format(self.pathname, self.folder_name),
            )
            return qcart, pcart

        elif  "const" in self.enskey:
            qp = False
            if "qp" in self.enskey:
                qp = True
            qpkey = "qp" if qp else "q"
            thermalize_rp_const_qp(
                self.pathname,
                self.m,
                self.dim,
                self.N,
                self.nbeads,
                ens,
                self.pes,
                rng,
                self.time_ens,
                self.dt_ens,
                self.potkey,
                rngSeed,
                tau0=self.tau0,
                pile_lambda=self.pile_lambda,
                folder_name="Datafiles",
                store_thermalization=True,
                pes_fort=False,
                propa_fort=False,
                transf_fort=False,
                qp=qp,
                Q0 = self.Q0,
                P0 = self.P0
            )
            qcart = read_arr(
                    "Const_{}_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                        qpkey, self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                    ),
                    "{}/{}".format(self.pathname, self.folder_name),
                )
            pcart = read_arr(
                    "Const_{}_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}".format(
                        qpkey, self.N, self.nbeads, ens.beta, self.potkey, rngSeed
                    ),
                    "{}/{}".format(self.pathname, self.folder_name),
                )
            return qcart, pcart

    def run_OTOC(self, sim, single=False):
        """ 
        Run simulation to compute out-of-time-order correlation function
        C(t) = < [A(t),B(0)]^2 > (Quantum)
        C(t) = < {A(t),B(0)}^2 > (Semiclassical)

        Note: By default we assume A = x and B = p_x, so that 
        C_sc(t) = < |Mqq(t)|^2 > 

        sim   : Simulation object
        single: Boolean to specify whether we compute the single commutator or the double commutator

        """
        tarr = []
        Marr = [] # List to record the monodromy variables
        if self.method == "CMD":
            stride = self.gamma
            dt = self.dt / self.gamma
            nsteps = int(self.time_run / dt) 
            sim.therm = PILE_L(tau0=self.tau0, pile_lambda=1.0) 
            sim.motion = Motion(dt=dt, symporder=sim.motion.order) # reinitialise motion object with new dt
            sim.bind(sim.ens, sim.motion, sim.rng, sim.rp, sim.pes, sim.propa, sim.therm)
            for i in range(nsteps):
                sim.step(mode="nvt", var="monodromy", pc=False)
                if i % stride == 0:
                    if (self.corrkey=='OTOC_qq' or self.corrkey=='OTOC'):
                        M = np.mean(abs(sim.rp.Mqq[:, 0, 0, 0, 0] ** 2))
                    elif (self.corrkey=='OTOC_ApAp'):
                        Mpp = sim.rp.Mpp[:, 0, 0, 0, 0]
                        Mpq = sim.rp.Mpq[:, 0, 0, 0, 0]
                        Mqp = sim.rp.Mqp[:, 0, 0, 0, 0]
                        Mqq = sim.rp.Mqq[:, 0, 0, 0, 0]
                        M = Mpp + Mqq + self.rp.dynm3*lamda*Mqp + Mpq/(self.rp.dynm3*lamda)
                        M = np.mean(abs(M**2))
                    tarr.append(sim.t)
                    Marr.append(M)

        else:
            dt = self.dt
            nsteps = int(self.time_run / dt)
            for i in range(nsteps):
                sim.step(mode="nve", var="monodromy")
                if single:
                    Mqq = np.mean(
                        sim.rp.Mqp[:, 0, 0, 0, 0]
                    )  # Change notations when required!
                    tarr.append(sim.t)
                    Marr.append(Mqq)
                else: 
                    if (self.corrkey=='OTOC_qq' or self.corrkey=='OTOC'):
                        M = np.mean(abs(sim.rp.Mqq[:, 0, 0, 0, 0] ** 2))
                    elif (self.corrkey=='OTOC_ApAp'):
                        Mpp = sim.rp.Mpp[:, 0, 0, 0, 0]
                        Mpq = sim.rp.Mpq[:, 0, 0, 0, 0]
                        Mqp = sim.rp.Mqp[:, 0, 0, 0, 0]
                        Mqq = sim.rp.Mqq[:, 0, 0, 0, 0]
                        #Note: Vectorized mass should be used here when the masses
                        #are different for different beads. It is suppressed here for
                        #making the code quicker.
                        pref = sim.rp.dynm3[0,0,0]*self.extparam['lamda']
                        M = 0.5*(Mpp + Mqq + Mpq + pref*Mqp + Mpq/pref)
                        M = np.mean(abs(M**2))
                    tarr.append(sim.t)
                    Marr.append(M)

        return tarr, Marr

    def run_R2(self, sim, A="p", B="p", C="q", seed_number=None):
        r"""Run simulation to compute second order (sym and asym) response functions
        symR2  <C(t2)B(t1)A(t0)>
        asymR2 <C(t2)Mqq(t1,t0)>
        with Mqq(t1,t0)= \partial q(t1)/\partial q(t0)
        We propagate the stability matrix (var="monodromy" inside the step call)
        """

        # IMPORTANT: Be careful when you use it for 2D! There are parts used in this code,
        # which are 1D-specific.
        assert (
            self.dim == 1
        ), "dimension = {} but R2 is only implemented for dim==1, sorry".format(
            self.dim
        )

        tarr, qarr, parr, Marr = [], [], [], []
        dt = self.dt
        nsteps = int(self.time_run / dt) + 1
        sqrtnbeads = sim.rp.nbeads**0.5
        # comm_dict = {'qq':[-1.0,'Mqp'],'qp':[1.0,'Mqq'], 'pq':[-1.0,'Mpp'], 'pp':[1.0,'Mpq']}

        def record_var():
            Mqq = sim.rp.Mqq[:, 0, 0, 0, 0].copy()
            # Mqp = sim.rp.Mqp[:,0,0,0,0].copy()
            # Mpq = sim.rp.Mpq[:,0,0,0,0].copy()
            # Mpp = sim.rp.Mpp[:,0,0,0,0].copy()
            q = sim.rp.q[:, 0, 0].copy()
            p = sim.rp.p[:, 0, 0].copy() / sim.rp.m

            Mval = Mqq / sim.rp.m
            tarr.append(sim.t)
            Marr.append(Mval)
            qarr.append(
                q / sqrtnbeads
            )  # Needed to add further scaling to transform the 0 normal mode to centroid
            parr.append(p / sqrtnbeads)

        pcart = sim.rp.pcart.copy()
        qcart = sim.rp.qcart.copy()

        # Forward propagation
        for i in range(nsteps):
            record_var()
            sim.step(mode="nve", var="monodromy")

        # Reinitialising position and momenta for backward propagation
        sim.rp = RingPolymer(
            qcart=qcart, pcart=pcart, m=sim.rp.m, mode="rp"
        )  # Only RPMD here!
        sim.motion = Motion(dt=-self.dt, symporder=sim.motion.order)
        sim.bind(sim.ens, sim.motion, sim.rng, sim.rp, sim.pes, sim.propa, sim.therm)
        sim.t = 0.0

        # Backward propagation
        for i in range(nsteps - 1):
            sim.step(mode="nve", var="monodromy")
            record_var()
        if seed_number is None:
            print("Propagation completed")
        else:
            print("Propagation completed. Seed: {}".format(seed_number))

        op_dict = {"I": np.ones_like(np.array(qarr)), "q": qarr, "p": parr}
        Aarr = np.array(op_dict[A].copy())
        Barr = np.array(op_dict[B].copy())
        Carr = np.array(op_dict[C].copy())

        Marr = np.array(Marr.copy())
        Marr = Marr[:, :, None]  # NEEDS TO CHANGE FOR 2D
        tarr = np.array(tarr)

        # Compute correlation function
        tarr1, Csym = gen_2pt_tcf(
            dt, tarr, Carr, Barr, Aarr
        )  # In the order t2,t1 and t0.
        tarr2, Casym = gen_2pt_tcf(dt, tarr, Carr, Marr)
        if np.alltrue(tarr1 == tarr2):
            tarr = tarr1

        return tarr, Csym, Casym

    def run_R2_eq(self, sim, seed_number=None):
        r"""Run simulation to compute second order response function
        R^(2)(t1,t2) = -\beta < Mqp(t2,t0)p(-t1) >
        with Mqp(t1,t0) = \partial q(t2)/\partial q(t0)
        We propagate the stability matrix (var="monodromy" inside the step call)

        As a reminder for developers:
           - M_xy.shape = (self.nsys,self.ndim,self.ndim,self.nmodes,self.nmodes)
        """
        assert (
            self.dim <= 2
        ), "dimension = {} but R2 is only implemented for dim==1 and dim==2, sorry".format(
            self.dim
        )
        if self.dim == 1:
            icoord_1 = 0  # Coordinate that couples with first pulse
            icoord_2 = 0  # Coordinate that couples with second pulse
            icoord_3 = 0  # Coordinate that couples that emmits light

        elif self.dim == 2:
            icoord_1 = self.coord_list[0]  # Coordinate that couples with first pulse
            icoord_2 = self.coord_list[1]  # Coordinate that couples with second pulse
            icoord_3 = self.coord_list[2]  # Coordinate that couples that emmits light

        assert icoord_1 < self.dim
        assert icoord_2 < self.dim
        assert icoord_3 < self.dim

        tarr, qarr, parr, Marr = [], [], [], []
        dt = self.dt
        nsteps = int(self.time_run / dt) + 1
        sqrtnbeads = sim.rp.nbeads**0.5

        def record_var():
            Mqp = sim.rp.Mqp[:, icoord_3, icoord_2, 0, 0].copy()
            # Mpq = sim.rp.Mpq[:,0,0,0,0].copy()
            # Mpp = sim.rp.Mpp[:,0,0,0,0].copy()
            q = sim.rp.q[:, icoord_1, 0].copy()  # CENTROID
            p = sim.rp.p[:, icoord_1, 0].copy() / sim.rp.m  # CENTROID
            # Needed to add further scaling to transform the 0 normal mode to centroid
            qarr.append(q / sqrtnbeads)
            parr.append(p / sqrtnbeads)

            Mval = Mqp / sim.rp.m
            tarr.append(sim.t)
            Marr.append(Mval)

        pcart = sim.rp.pcart.copy()
        qcart = sim.rp.qcart.copy()

        # Forward propagation
        for i in range(nsteps):
            record_var()
            sim.step(mode="nve", var="monodromy")

        # Reinitialising position and momenta for backward propagation
        sim.rp = RingPolymer(
            qcart=qcart, pcart=pcart, m=sim.rp.m, mode="rp"
        )  # Only RPMD here!
        sim.motion = Motion(dt=-self.dt, symporder=sim.motion.order)
        sim.bind(sim.ens, sim.motion, sim.rng, sim.rp, sim.pes, sim.propa, sim.therm)
        sim.t = 0.0

        # Backward propagation
        for i in range(nsteps - 1):
            sim.step(mode="nve", var="monodromy")
            record_var()
        if seed_number is None:
            print("Propagation completed")
        else:
            print("Propagation completed. Seed: {}".format(seed_number))

        tarr = reorder_time(np.array(tarr), len(tarr), mode=1)
        Aarr = reorder_time(np.array(parr), len(tarr), mode=1)
        Marr = {"qp": reorder_time(np.array(Marr), len(tarr), mode=1)}

        # Compute correlation function
        tar, R2eq = gen_R2_tcf(dt, tarr, Aarr, Marr, self.beta)

        return tar, R2eq

    def run_R3_eq(self, sim, seed_number=None):
        r"""Run simulation to compute third order response functions propagating the stability matrix
        R3 = beta < (Mqq(t3,t0)Mqp(t2,t0) - Mqp(t3,t0)Mqq(t2,t0)) * (Mqq(t1,t0)-beta p(t1)p(t0)) >
        with Mxy(t1,t0)= \partial x(t1)/\partial y(t0)
        We propagate the stability matrix (var="monodromy" inside the step call)
        """

        assert (
            self.dim == 1
        ), "dimension = {} but R3 is only implemented for dim==1, sorry".format(
            self.dim
        )

        tarr, qarr, parr, Marr_qq, Marr_qp, Marr_pq, Marr_pp = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        dt = self.dt
        if debug:
            print("here: dt", dt)
        nsteps = int(self.time_run / dt) + 1
        sqrtnbeads = sim.rp.nbeads**0.5
        # comm_dict = {'qq':[-1.0,'Mqp'],'qp':[1.0,'Mqq'], 'pq':[-1.0,'Mpp'], 'pp':[1.0,'Mpq']}

        def record_var():
            """Define variables to be stored during simulation and required to compute the observables"""
            Mqq = sim.rp.Mqq[:, 0, 0, 0, 0].copy()
            Mqp = sim.rp.Mqp[:, 0, 0, 0, 0].copy()
            Mpq = sim.rp.Mpq[:, 0, 0, 0, 0].copy()
            Mpp = sim.rp.Mpp[:, 0, 0, 0, 0].copy()
            q = sim.rp.q[:, 0, 0].copy()
            p = sim.rp.p[:, 0, 0].copy() / sim.rp.m  # CHECK ALBERTO

            Mval_qq = Mqq  # CHECK ALBERTO
            Mval_qp = Mqp
            Mval_pq = Mpq
            Mval_pp = Mpp
            tarr.append(sim.t)
            Marr_qq.append(Mval_qq)
            Marr_qp.append(Mval_qp)
            Marr_pq.append(Mval_pq)
            Marr_pp.append(Mval_pp)
            qarr.append(
                q / sqrtnbeads
            )  # Needed to add further scaling to transform the 0 normal mode to centroid
            parr.append(p / sqrtnbeads)

        pcart = sim.rp.pcart.copy()
        qcart = sim.rp.qcart.copy()

        # Forward propagation
        for i in range(nsteps):
            record_var()
            sim.step(mode="nve", var="monodromy")

        # Reinitialising position and momenta for backward propagation
        sim.rp = RingPolymer(
            qcart=qcart, pcart=pcart, m=sim.rp.m, mode="rp"
        )  # Only RPMD here!
        sim.motion = Motion(dt=-self.dt, symporder=sim.motion.order)
        sim.bind(sim.ens, sim.motion, sim.rng, sim.rp, sim.pes, sim.propa, sim.therm)
        sim.t = 0.0

        # Backward propagation
        for i in range(nsteps - 1):
            sim.step(mode="nve", var="monodromy")
            record_var()
        if seed_number is None:
            print("Propagation completed")
        else:
            print("Propagation completed. Seed: {}".format(seed_number))
        # End of propagation

        # TCF calculation
        # rearrange arrays to have time in increasing order
        nmode = 1
        tarr = reorder_time(np.array(tarr)[:, np.newaxis], len(tarr), mode=nmode)
        Aarr = reorder_time(np.array(parr)[:, np.newaxis], len(tarr), mode=nmode)
        Barr = reorder_time(np.array(parr)[:, np.newaxis], len(tarr), mode=nmode)

        Marr = {
            "qq": reorder_time(np.array(Marr_qq)[:, np.newaxis], len(tarr), mode=nmode),
            "pp": reorder_time(np.array(Marr_pp)[:, np.newaxis], len(tarr), mode=nmode),
            "qp": reorder_time(np.array(Marr_qp)[:, np.newaxis], len(tarr), mode=nmode),
            "pq": reorder_time(np.array(Marr_pq)[:, np.newaxis], len(tarr), mode=nmode),
        }

        # Compute correlation function
        tar, R3eq = gen_R3_tcf(dt, tarr, Aarr, Barr, Marr, self.beta)

        return tar, R3eq

    def run_TCF(self, sim):
        """Run simulation to compute 'simple' correlation functions, such as qq,qq2,etc
        We do not propagate the stability matrix (var="pq" inside the step call)"""
        tarr = []
        qarr = []
        parr = []
        if self.method == "CMD":
            stride = self.gamma
            dt = self.dt / self.gamma # Smaller time steps required because higher normal modes move faster
            nsteps = int(2*self.time_run / dt) + 1
            sim.motion = Motion(dt=dt, symporder=sim.motion.order) # reinitialise motion object with new dt
            sim.bind(sim.ens, sim.motion, sim.rng, sim.rp, sim.pes, sim.propa, sim.therm)
            for i in range(nsteps):
                if i % stride == 0:
                    q = sim.rp.q[:, :, 0].copy()
                    p = sim.rp.p[:, :, 0].copy()
                    tarr.append(sim.t)
                    qarr.append(q)
                    parr.append(p)
                sim.step(mode="nvt", var="pq", pc=False)
        else:
            dt = self.dt
            nsteps = int(2 * self.time_run / dt) + 1
            for i in range(nsteps):
                q = sim.rp.q[:, :, 0].copy()
                p = sim.rp.p[:, :, 0].copy()
                tarr.append(sim.t)
                qarr.append(q)
                parr.append(p)
                sim.step(mode="nve", var="pq")

        qarr = np.array(qarr)
        parr = np.array(parr)
        # NOTE: The correlation function is between vector quantities unless explicitly specified.
        # This needs to be rewritten at some point to make it more general.
        if self.corrkey == "qq_TCF":
            tarr, tcf = gen_tcf(qarr, qarr, tarr, corraxis=0)
        elif self.corrkey == "qq2_TCF":
            tarr, tcf = gen_tcf(qarr**2, qarr**2, tarr)
        elif self.corrkey == "pp_TCF":
            tarr, tcf = gen_tcf(parr, parr, tarr, corraxis=0)
        elif self.corrkey == "pp2_TCF":
            tarr, tcf = gen_tcf(parr**2, parr**2, tarr)
        elif self.corrkey == "qp_TCF":
            tarr, tcf = gen_tcf(qarr, parr, tarr, corraxis=0)
        elif self.corrkey == "pq_TCF":
            tarr, tcf = gen_tcf(parr, qarr, tarr, corraxis=0)
        return tarr, tcf

    def run_seed(self, rngSeed, op=None):
        """Runs one seed. Note that this is n_traj (~1000) parallel trajectories"""
        print(
            "Start simulation.      Seed: {}  T {}, nbeads {}".format(
                rngSeed, self.T, self.nbeads
            )
        )
        rng = np.random.default_rng(rngSeed)
        ens = Ensemble(beta=1 / self.T, ndim=self.dim)
        qcart, pcart = self.gen_ensemble(ens, rng, rngSeed)

        if self.method == "Classical" or self.method == "RPMD" or self.method == "rpmd":
            rp = RingPolymer(qcart=qcart, pcart=pcart, m=self.m, mode="rp")
        elif self.method == "CMD":
            rp = RingPolymer(
                qcart=qcart,
                pcart=pcart,
                m=self.m,
                mode="rp",
                scaling="cmd",
                nmats=1,
                sgamma=self.gamma,
            )

        therm = PILE_L(tau0=self.tau0, pile_lambda=self.pile_lambda) 
        motion = Motion(dt=self.dt, symporder=self.symplectic_order)
        propa = Symplectic()

        sim = RP_Simulation()
        sim.bind(ens, motion, rng, rp, self.pes, propa, therm, 
                 pes_fort = self.pes_fort, propa_fort = self.propa_fort, transf_fort = self.transf_fort)

        if "OTOC" in self.corrkey:
            tarr, Carr = self.run_OTOC(sim)
        elif "TCF" in self.corrkey:
            tarr, Carr = self.run_TCF(sim)
        elif self.corrkey == "stat_avg":
            if op is 'Hess':
                pes_ddpot_cart = sim.pes.compute_hessian()
                Hess_cart = pes_ddpot_cart + sim.rp.ddpot_cart
                Hess = Hess_cart.reshape(-1,self.dim*sim.rp.nbeads, self.dim*sim.rp.nbeads)
                vals = np.sort( np.linalg.eigvalsh(Hess), axis=1)[:,0]
                self.store_scalar(np.mean(vals), rngSeed, suffix='Hessian')
            
                Hess_norm = sim.pes.nmtrans.cart2mats_hessian(pes_ddpot_cart) + sim.rp.ddpot
                Hess_norm = Hess_norm[:,:,0,:,0]
                vals_cent = np.sort( np.linalg.eigvalsh(Hess_norm), axis=1)[:,0]
                self.store_scalar(np.mean(vals_cent), rngSeed, suffix='centroid_Hessian')
                print('Hessian computed',np.sqrt(-vals.mean()/sim.rp.m),np.sqrt(-vals_cent.mean()/sim.rp.m))
                return
            else:
                # The assumption here is that 'op' is scalar-valued function (i.e. returns a scalar for every bead)
                avg = np.mean(np.mean(op(sim.rp.qcart, sim.rp.pcart), axis=1))
                self.store_scalar(avg, rngSeed)
                return
        elif self.corrkey == "singcomm":
            tarr, Carr = self.run_OTOC(sim, single=True)
        elif self.corrkey == "R2":
            assert self.extparam is not None

            tarr, Csym, Casym = self.run_R2(
                sim,
                self.extparam[0],
                self.extparam[1],
                self.extparam[2],
                seed_number=rngSeed,
            )
            self.store_time_series_2D(tarr, Csym, rngSeed, "sym")
            self.store_time_series_2D(tarr, Casym, rngSeed, "asym")
            return
        elif self.corrkey == "R2eq":
            tarr, R2 = self.run_R2_eq(
                sim,
                seed_number=rngSeed,
            )
            self.store_time_series_2D(tarr, R2, rngSeed, "R2", mode=2)
            return

        elif self.corrkey == "R3eq":
            tarr, R3_eq = self.run_R3_eq(
                sim,
                seed_number=rngSeed,
            )
            self.store_time_series_3D(tarr, R3_eq, rngSeed, "R3_eq")
            return
        else:
            raise NotImplementedError  # YL: I don't think we should reach this line ever. VJ: I agree :)
            return

        self.store_time_series(tarr, Carr, rngSeed)

    def assign_fname(self, rngSeed, suffix=None):
        key = [
            self.method,
            self.enskey,
            self.corrkey,
            self.sysname,
            self.potkey,
            self.Tkey,
            "N_{}".format(self.N),
            "dt_{}".format(self.dt),
        ]
        fext = "_".join(key)
        if self.method == "Classical" or self.method == "classical":
            methext = "_"
        elif self.method == "RPMD" or self.method == "rpmd":
            methext = "_nbeads_{}_".format(self.nbeads)
        elif self.method == "CMD" or self.method == "cmd":
            methext = "_nbeads_{}_gamma_{}_".format(self.nbeads, self.gamma)
        else:
            raise NotImplementedError(
                "{} method is not implemented, sorry".format(self.method)
            )

        if self.corrkey != "stat_avg":
            seedext = "seed_{}".format(rngSeed)
        else:
            seedext = ""

        if self.ext_kwlist is None and suffix is None:
            fname = "".join([fext, methext, seedext])
        elif self.ext_kwlist is None:
            if seedext == "":
                fname = "".join([fext, methext, suffix])
            else:
                fname = "".join([fext, methext, suffix + "_", seedext])
        elif suffix is None:
            fname = "".join([fext, methext, "_".join(self.ext_kwlist) + "_", seedext])
        else:
            namelst = [fext, methext, suffix + "_"]
            namelst.append("_".join(self.ext_kwlist) + "_")
            namelst.append(seedext)
            fname = "".join(namelst)

        return fname

    def store_time_series(self, tarr, Carr, rngSeed):
        fname = self.assign_fname(rngSeed)
        store_1D_plotdata(
            tarr, Carr, fname, "{}/{}".format(self.pathname, self.folder_name)
        )

    def store_time_series_2D(self, tarr, Carr, rngSeed, suffix=None, mode=1):
        fname = self.assign_fname(rngSeed, suffix)
        store_2D_imagedata(
            tarr,
            tarr,
            Carr,
            fname,
            "{}/{}".format(self.pathname, self.folder_name),
            mode=mode,
        )

    def store_time_series_3D(self, tarr, Carr, rngSeed, suffix=None):
        fname = self.assign_fname(rngSeed, suffix)
        store_3D_imagedata(
            tarr,
            tarr,
            tarr,
            Carr,
            fname,
            "{}/{}".format(self.pathname, self.folder_name),
        )

    def store_scalar(self, scalar, rngSeed, suffix=None):
        # Scalar values are stored in the same filename
        fname = self.assign_fname(rngSeed, suffix)
        f = open("{}/{}/{}.txt".format(self.pathname, self.folder_name, fname), "a")
        f.write(str(rngSeed) + "  " + str(scalar) + "\n")
        f.close()


def check_parameters(sim_parameters, ensemble_param, system_param):
    """Checks the consistency of  simulation,  ensemble and system parameters"""
    if (
        sim_parameters["method"] == "classical"
        or sim_parameters["method"] == "Classical"
    ):
        assert (
            sim_parameters["nbeads"] == 1
        ), "Classical simulations requires nbeads == 1"
    elif sim_parameters["method"] == "cmd":
        assert (
            sim_parameters["cmd_gamma"] is not None
        ), "Please specify the cmd_gamma parameter to run cmd simulations"
    elif sim_parameters["method"] == "rpmd":
        assert (
            sim_parameters["pile_lambda"] is not None
        ), "Please specify the pile_lambda parameter to run trpmd simulations"
    else:
        raise NotImplementedError(
            "method {} is not implemented".format(sim_parameters["method"])
        )

    if sim_parameters["CFtype"] == "R2":
        assert (
            len(sim_parameters["operator_list"]) == 3
        ), "Please specify operators list (example -op_list q q q ) to R2 simulations "

    if sim_parameters["CFtype"] == "R2eq":
        assert (
            sim_parameters["operator_list"] is None
        ), "R2eq doesn't need op_list keyword. For safety reasons we abort here"

        if system_param["dimension"] > 1:
            assert (
                sim_parameters["coordinate_list"] is not None
            ), "R2eq for dim>1 requires coordinate list argument"
            assert (
                len(sim_parameters["coordinate_list"]) == 3
            ), "Coordinate list for R2eq should have exactly 3 arguments"

    elif sim_parameters["CFtype"] == "R3eq":
        assert (
            sim_parameters["operator_list"]
        ) is None, "Sorry, we always assume A=q B=q C=q D=q, please do not use the  -op_list keyword "

        if system_param["dimension"] > 1:
            assert (
                sim_parameters["coordinate_list"] is not None
            ), "R3eq for dim>1 requires coordinate list argument"
            assert (
                len(sim_parameters["coordinate_list"]) == 4
            ), "Coordinate list for R3eq should have exactly 4 arguments"

    if ensemble_param["ensemble"] == "thermal":
        assert (
            ensemble_param["temperature"] > 0
        ), "{} K is not a valid temperature for the ensemble {}".format(
            ensemble_param["temperature"], ensemble_param["ensemble"]
        )
        assert (
            ensemble_param["tau"] > 0
        ), "{} K is not a valid thermostat constant for the ensemble {}".format(
            ensemble_param["tau"], ensemble_param["ensemble"]
        )
    elif ensemble_param["ensemble"] == "mc":
        assert (
            ensemble_param["energy_mc"] is not None
        ), "Please specify energy to run in the microcanonical ensemble"
    else:
        raise NotImplementedError
