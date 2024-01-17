"""
This module defines the main 'RingPolymer' class for treating
discretized imaginary time path-integrals. This class can be used
to run CMD, RPMD, Matsubara and mean-field Matsubara simulations
"""

from turtle import setundobuffer
import numpy as np
from PISC.utils import nmtrans, misc


class RingPolymer(object):
    """
    Attributes:
    nbeads : Number of imaginary time points or 'beads'
    nmodes : Number of ring-polymer normal modes (Usually same as nbeads)
    nsys   : Number of parallel ring-polymer units
    nmats  : Number of Matsubara modes (Differs from nbeads for Matsubara simulations)

    q/qcart         : Ring-polymer bead positions in Cartesian/Matsubara coordinates
    p/pcart         : Ring-polymer bead momenta in Cartesian/Matsubara coordinates
                          Dimensions [nsys,ndim,nmodes]

    Mpp,Mpq,Mqp,Mqq : Monodromy matrix elements of the ring-polymer beads in Matsubara coordinates
                      Dimensions [nsys,ndim,nmodes,ndim,nmodes]

    m3   : Vectorized ring-polymer bead masses
    sqm3 : Square root of m3

    omegan : Prefactor for the ring-polymer normal mode frequencies
    freqs  : Ring-polymer normal mode frequencies
    freqs2 : Square of ring-polymer normal mode frequencies
    dynfreqs : Ring-polymer normal mode frequencies with mass scaling
    dynfreq2 : Square of ring-polymer normal mode frequencies with mass scaling

    nmscale : Normal-mode scaling factor
    nm_matrix : Normal-mode transformation matrix

    pot : Potential energy of the ring-polymer spring
    dpot/dpot_cart   : First derivative of the spring potential energy in Matsubara/Cartesian coordinates
    ddpot/ddpot_cart : Second derivative of the spring potential energy in Matsubara/Cartesian coordinates

    dq/dqcart : Tangent variables (positions) in Matsubara/Cartesian coordinates
    dp/dpcart : Tangent variables (momenta) in Matsubara/Cartesian coordinates

    *_f : Fortran contiguous arrays for all variables (for use with Fortran subroutines)  

    """

    def __init__(
        self,
        p=None,
        dp=None,
        q=None,
        dq=None,
        pcart=None,
        dpcart=None,
        qcart=None,
        dqcart=None,
        Mpp=None,
        Mpq=None,
        Mqp=None,
        Mqq=None,
        m=None,
        labels=None,
        nbeads=None,
        nmodes=None,
        scaling=None,
        sgamma=None,
        sfreq=None,
        freqs=None,
        mode="rp",
        nmats=None,
        qdev_eps=1e-4,
    ):
        if qcart is None:
            if p is not None:
                self.p = p
            else:
                self.p = None
            self.pcart = None
            if nmodes is None:
                nmodes = q.shape[-1]
            self.q = q
            self.qcart = None
            self.nbeads = nmodes
            self.nmodes = nmodes
            self.nsys = len(self.q)

        if q is None:
            if pcart is not None:
                self.pcart = pcart
            else:
                self.pcart = None
            self.p = None
            if nbeads is None:
                nbeads = qcart.shape[-1]
            self.qcart = qcart
            self.q = None
            self.nmodes = nbeads
            self.nbeads = nbeads
            self.nsys = len(self.qcart)

        """
        Create variables dq,dp, dqcart, dpcart default set to None
        If not None, they can be passed in cartesian/Matsubara coordinates.
        Like with q/p, qcart/pcart, these variables need to be interconverted.
        """
        if dqcart is not None:
            self.dqcart = dqcart
            if dpcart is not None:
                self.dpcart = dpcart
            else:
                self.dpcart = 0.0
        else:
            self.dqcart = None
            self.dpcart = None
        if dq is not None:
            self.dq = dq
            if dp is not None:
                self.dp = dp
            else:
                self.dp = 0.0
        else:
            self.dq = None
            self.dp = None 

        """
        Create variables qe,qme, qecart,qmecart default set to None
        Also define initial deviation 
        """
        self.qdev = qdev_eps
        self.qe = None
        self.qecart = None
        self.pe = None
        self.pecart = None

        self.qme = None
        self.qmecart = None
        self.pme = None
        self.pmecart = None        

        self.m = m
        self.mode = mode
        if Mpp is None:
            self.Mpp = None
        else:
            self.Mpp = Mpp

        if Mpq is None:
            self.Mpq = None
        else:
            self.Mpq = Mpq

        if Mqp is None:
            self.Mqp = None
        else:
            self.Mqp = Mqp

        if Mqq is None:
            self.Mqq = None
        else:
            self.Mqq = Mqq

        if scaling is None:
            self.scaling = "none"
            self._sgamma = None
            self._sfreq = None
        elif scaling == "MFmats":
            self.nmats = nmats
            self.scaling = scaling
            self._sgamma = sgamma
            self._sfreq = sfreq
        else:
            self.scaling = scaling
            self._sgamma = sgamma
            self._sfreq = sfreq

        if mode == "rp":
            self.nmats = None
        elif mode == "rp/mats" or mode == "mats":
            if nmats is None:
                raise ValueError("Number of Matsubara modes needs to be specified")
            else:
                self.nmats = nmats
        self.freqs = freqs

    def bind(self, ens, motion, rng, fort=False):
        self.ens = ens
        self.motion = motion
        self.rng = rng
        self.ndim = ens.ndim
        self.dt = self.motion.dt

        self.omegan = self.nbeads / self.ens.beta
        self.nmtrans = nmtrans.FFT(self.ndim, self.nbeads, self.nmodes)
        self.nmtrans.compute_nm_matrix()
        self.nm_matrix = self.nmtrans.nm_matrix

        if self.mode == "mats" or self.mode == "rp/mats":
            self.freqs = self.get_rp_freqs()
            self.matsfreqs = self.get_mats_freqs()
        elif self.mode == "rp" or self.freqs is None:
            self.freqs = self.get_rp_freqs()

        if self.qcart is None:
            self.qcart = self.nmtrans.mats2cart(self.q)
        elif self.q is None:
            self.q = self.nmtrans.cart2mats(self.qcart)

        if self.dqcart is not None:
            self.dq = self.nmtrans.cart2mats(self.dqcart)
            self.dp = self.nmtrans.cart2mats(self.dpcart)
        elif self.dq is not None:
            self.dqcart = self.nmtrans.mats2cart(self.dq)
            self.dpcart = self.nmtrans.mats2cart(self.dp)

        if self.qe is not None:
            self.qecart = self.nmtrans.mats2cart(self.qe)
            self.pecart = self.nmtrans.mats2cart(self.pe)
        elif self.qecart is not None:
            self.qe = self.nmtrans.cart2mats(self.qecart)
            self.pe = self.nmtrans.cart2mats(self.pecart)

        if self.qme is not None:
            self.qmecart = self.nmtrans.mats2cart(self.qme)
            self.pmecart = self.nmtrans.mats2cart(self.pme)
        elif self.qmecart is not None:
            self.qme = self.nmtrans.cart2mats(self.qmecart)
            self.pme = self.nmtrans.cart2mats(self.pmecart)

        self.m3 = np.ones_like(self.q) * self.m
        self.sqm3 = np.sqrt(self.m3)

        if self.Mpp is None and self.Mqq is None:
            self.Mpp = np.zeros(
                (self.nsys, self.ndim, self.nmodes, self.ndim, self.nmodes)
            )
            for d in range(self.ndim):
                self.Mpp[:, d, :, d] = np.eye(self.nmodes, self.nmodes)
            self.Mqq = self.Mpp.copy()

        if self.Mqp is None and self.Mpq is None:
            self.Mqp = np.zeros_like(self.Mqq)
            self.Mpq = np.zeros_like(self.Mqq)

        self.freqs2 = self.freqs**2
        self.nmscale = self.get_dyn_scale()
        self.dynm3 = self.m3 * self.nmscale
        self.sqdynm3 = np.sqrt(self.dynm3)
        self.dynfreqs = self.freqs / np.sqrt(self.nmscale)
        self.dynfreq2 = self.dynfreqs**2

        self.get_RSP_coeffs()
        self.ddpot = np.zeros(
            (self.nsys, self.ndim, self.nmodes, self.ndim, self.nmodes)
        )
        self.ddpot_cart = np.zeros(
            (self.nsys, self.ndim, self.nbeads, self.ndim, self.nbeads)
        )

        ##Check here when doing multi-D (more than 2D) simulation with beads.
        if self.nbeads > 1:
            for d in range(self.ndim):
                self.ddpot[:, d, :, d] = np.eye(self.nmodes, self.nmodes)

            self.ddpot *= (self.dynm3 * self.dynfreq2)[:, :, :,None, None]

            for d in range(self.ndim):
                for k in range(self.nbeads - 1):
                    self.ddpot_cart[:, d, k, d, k] = 2
                    self.ddpot_cart[:, d, k, d, k + 1] = -1
                    self.ddpot_cart[:, d, k, d, k - 1] = -1
                self.ddpot_cart[:, d, self.nbeads - 1, d, 0] = -1
                self.ddpot_cart[:, d, self.nbeads - 1, d, self.nbeads - 1] = 2
                self.ddpot_cart[:, d, self.nbeads - 1, d, self.nbeads - 2] = -1
 
            self.ddpot_cart *= (self.dynm3 * self.omegan**2)[:, :, :, None, None]

        if self.p is None and self.pcart is None:
            self.p = self.rng.normal(
                size=self.q.shape, scale=1 / np.sqrt(self.ens.beta)
            )
            self.pcart = self.nmtrans.mats2cart(self.p)
        elif self.pcart is None:
            self.pcart = self.nmtrans.mats2cart(self.p)
        else:
            self.p = self.nmtrans.cart2mats(self.pcart)

        if fort is True:
            self._bind_fort()

    def _bind_fort(self):
        """ Create Fortran contiguous arrays for all variables """
        self.q_f = self.q.T
        self.p_f = self.p.T
        self.qcart_f = self.qcart.T
        self.pcart_f = self.pcart.T

        if self.dqcart is not None:
            self.dqcart_f = self.dqcart.T
            self.dpcart_f = self.dpcart.T
        if self.dq is not None:
            self.dq_f = self.dq.T
            self.dp_f = self.dp.T

        if self.qecart is not None:
            self.qecart_f = self.qecart.T
            self.pecart_f = self.pecart.T
        if self.qe is not None:
            self.qe_f = self.qe.T
            self.pe_f = self.pe.T

        if self.qmecart is not None:
            self.qmecart_f = self.qmecart.T
            self.pmecart_f = self.pmecart.T
        if self.qme is not None:
            self.qme_f = self.qme.T
            self.pme_f = self.pme.T

        self.m3_f = self.m3.T
        self.sqm3_f = self.sqm3.T
        self.dynm3_f = self.dynm3.T
        self.sqdynm3_f = self.sqdynm3.T
       
        self.freqs_f = self.freqs.T
        self.freqs2_f = self.freqs2.T
        self.dynfreqs_f = self.dynfreqs.T
        self.dynfreq2_f = self.dynfreq2.T
        
        self.nmscale_f = self.nmscale.T
        self.nm_matrix_f = self.nm_matrix.T

        self.pot_f = self.pot.T
        self.pot_cart_f = self.pot_cart.T
        self.dpot_f = self.dpot.T
        self.dpot_cart_f = self.dpot_cart.T
        self.ddpot_f = self.ddpot.T
        self.ddpot_cart_f = self.ddpot_cart.T
        
        self.Mpp_f = self.Mpp.T
        self.Mpq_f = self.Mpq.T
        self.Mqp_f = self.Mqp.T
        self.Mqq_f = self.Mqq.T

    def get_dyn_scale(self):
        scale = np.ones(self.nmodes)
        if self.scaling == "none":
            return scale
        elif self.scaling == "MFmats":
            if self._sgamma is None:
                scale = (self.freqs / self._sfreq) ** 2
            elif self._sfreq is None:
                scale = (self.freqs / (self._sgamma * self.omegan)) ** 2
            scale[: self.nmats] = 1.0
            return scale
        elif self.scaling == "cmd":
            if self._sgamma is None:
                scale = (self.freqs / self._sfreq) ** 2
            elif self._sfreq is None:
                scale = (self.freqs / (self._sgamma * self.omegan)) ** 2
            scale[0] = 1.0
            return scale

    def RSP_step(self):
        """ Perform one step using reference-system propagator """
        qpvector = np.empty((self.nmodes, 2, self.ndim, len(self.q)))
        qpvector[:, 0] = (self.p / self.sqdynm3).T
        qpvector[:, 1] = (self.q * self.sqdynm3).T

        qpvector[:] = np.einsum("ijk,ik...->ij...", self.RSP_coeffs, qpvector)

        self.p[:] = qpvector[:, 0].T * self.sqdynm3
        self.q[:] = qpvector[:, 1].T / self.sqdynm3

    def get_rp_freqs(self):
        """ Get ring-polymer normal mode frequencies """
        n = [0]
        for i in range(1, self.nmodes // 2 + 1):
            n.append(-i)
            n.append(i)
        if self.nmodes % 2 == 0:
            n.pop(-2)
        freqs = np.sin(np.array(n) * np.pi / self.nbeads) * (2 * self.omegan)
        return freqs

    def get_mats_freqs(self):
        """ Get Matsubara mode frequencies """
        n = [0]
        if self.mode == "mats":
            for i in range(1, self.nmodes // 2 + 1):
                n.append(-i)
                n.append(i)
            freqs = 2 * np.pi * np.array(n) / (self.ens.beta)
            return freqs
        elif self.mode == "rp/mats":
            for i in range(1, self.nmats // 2 + 1):
                n.append(-i)
                n.append(i)
            freqs = 2 * np.pi * np.array(n) / (self.ens.beta)
            return freqs

    def get_RSP_coeffs(self):
        """ Get reference-system propagator coefficients """
        self.RSP_coeffs = np.empty((self.nmodes, 2, 2))
        for n in range(self.nmodes):
            mat = np.eye(2) * np.cos(self.dynfreqs[n] * self.dt)
            mat[0, 1] = -self.dynfreqs[n] * np.sin(self.dynfreqs[n] * self.dt)
            mat[1, 0] = np.sinc(self.dynfreqs[n] * self.dt / np.pi) * self.dt
            self.RSP_coeffs[n] = mat

    def nm_matrix(self):
        """ Get the normal-mode transformation matrix """
        narr = [0]
        for i in range(1, self.nmodes // 2 + 1):
            narr.append(-i)
            narr.append(i)
        if self.nmodes % 2 == 0:
            narr.pop(-1)

        self.nm_matrix = np.zeros((self.nmodes, self.nmodes))
        self.nm_matrix[:, 0] = 1 / np.sqrt(self.nmodes)
        for l in range(self.nbeads):
            for n in range(1, len(narr)):
                if narr[n] < 0:
                    self.nm_matrix[l, n] = np.sqrt(2 / self.nmodes) * np.cos(
                        2 * np.pi * (l) * narr[n] / self.nmodes
                    )
                else:
                    self.nm_matrix[l, n] = np.sqrt(2 / self.nmodes) * np.sin(
                        2 * np.pi * (l) * narr[n] / self.nmodes
                    )
            if self.nmodes % 2 == 0:
                self.nm_matrix[l, self.nmodes - 1] = (-1) ** l / np.sqrt(self.nmodes)

    def mats_beads(self):
        """ Get the Matsubara bead positions """
        if self.nmats is None:
            return self.qcart
        else:
            ret = np.einsum(
                "ij,...j",
                (self.nmtrans.nm_matrix)[:, : self.nmats],
                self.q[..., : self.nmats],
            )  # Check this!
            return ret

    def mats2cart(self,fortran=False):
        """ Convert Matsubara bead positions to Cartesian coordinates """
        if fortran is True:
            self.nmtrans.mats2cart(self.q_f,self.qcart_f,fortran=True)
            self.nmtrans.mats2cart(self.p_f,self.pcart_f,fortran=True)
            if self.dqcart is not None:
                self.nmtrans.mats2cart(self.dq_f,self.dqcart_f,fortran=True)
                self.nmtrans.mats2cart(self.dp_f,self.dpcart_f,fortran=True)
        else:
            self.qcart[:] = self.nmtrans.mats2cart(self.q)
            self.pcart[:] = self.nmtrans.mats2cart(self.p)
            if self.dq is not None:
                self.dqcart[:] = self.nmtrans.mats2cart(self.dq)
                self.dpcart[:] = self.nmtrans.mats2cart(self.dp)
            if self.qe is not None:
                self.qecart[:] = self.nmtrans.mats2cart(self.qe)
            if self.qme is not None:
                self.qmecart[:] = self.nmtrans.mats2cart(self.qme)
        
        # If dq,dp are not None, convert them too

    def cart2mats(self,fortran=False):
        """ Convert Cartesian bead positions to Matsubara coordinates """
        if fortran is True:
            self.q_f[:] = self.nmtrans.cart2mats(self.qcart_f,fortran=True)
            self.p_f[:] = self.nmtrans.cart2mats(self.pcart_f,fortran=True)
            if self.dqcart is not None:
                self.dq_f[:] = self.nmtrans.cart2mats(self.dqcart_f,fortran=True)
                self.dp_f[:] = self.nmtrans.cart2mats(self.dpcart_f,fortran=True)
        else:
            self.q[:] = self.nmtrans.cart2mats(self.qcart,fortran=fortran)
            self.p[:] = self.nmtrans.cart2mats(self.pcart,fortran=fortran)
            if self.dqcart is not None:
                self.dq[:] = self.nmtrans.cart2mats(self.dqcart,fortran=fortran)
                self.dp[:] = self.nmtrans.cart2mats(self.dpcart,fortran=fortran)
            if self.qe is not None:
                self.qe[:] = self.nmtrans.cart2mats(self.qecart,fortran=fortran)
            if self.qme is not None:
                self.qme[:] = self.nmtrans.cart2mats(self.qmecart,fortran=fortran)
            
        # If dqcart,dpcart are not None, convert them too

    @property
    def theta(self):  
        """ Get the Matsubara phase """
        # Check for more than 1D
        ret = (
            self.matsfreqs
            * misc.pairwise_swap(self.q[..., : self.nmats], self.nmats)
            * self.p[..., : self.nmats]
        )
        return np.sum(ret, axis=2)

    @property
    def kin(self):
        """ Get the kinetic energy """
        return np.sum(0.5 * (self.p / self.sqm3) ** 2)

    @property
    def dynkin(self):
        """ Get the kinetic energy when masses are scaled """
        return np.sum(0.5 * (self.p / self.sqdynm3) ** 2)

    @property
    def pot(self):
        """ Get the potential energy """
        return np.sum(0.5 * self.dynm3 * self.dynfreq2 * self.q**2)

    @property
    def pot_cart(self):
        """ Get the potential energy in Cartesian coordinates """
        return np.sum(
            0.5
            * self.m3
            * self.omegan**2
            * (self.qcart - np.roll(self.qcart, 1, axis=-1)) ** 2
        )

    @property
    def dpot(self):
        """ Get the potential energy gradient in Matsubara coordinates "" """
        return self.dynm3 * self.dynfreq2 * self.q

    @property
    def dpot_cart(self):
        """ Get the potential energy gradient in Cartesian coordinates """
        return (
            self.dynm3
            * self.omegan**2
            * (
                2 * self.qcart
                - np.roll(self.qcart, 1, axis=-1)
                - np.roll(self.qcart, -1, axis=-1)
            )
            )
