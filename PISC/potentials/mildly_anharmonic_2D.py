import numpy as np
from PISC.potentials.base import PES


class mildly_anharmonic_2D(PES):
    def __init__(self, m, a1, b1, a2, b2, c):
        """Initialize Mildly anharmonic potential 2D"""
        super(mildly_anharmonic_2D).__init__()
        self.m = m
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.c = c

    def bind(self, ens, rp):
        """Bind the potential energy surface to the ensemble and the ring polymer"""
        super(mildly_anharmonic, self).bind(ens, rp)

    def potential(self, q):
        """Potential energy surface for the mildly anharmonic potential 2D"""
        x = q[:, 0]
        y = q[:, 1]
        Vx = 0.5 * self.m * x**2 + self.a1 * x**3 + self.b1 * x**4
        Vy = 0.5 * self.m * y**2 + self.a2 * y**3 + self.b2 * y**4
        Vxy = self.c * x * y
        return Vx + Vy

    def dpotential(self, q):
        """Gradient of the potential energy surface for the mildly anharmonic potential 2D"""
        x = q[:, 0]
        y = q[:, 1]
        Vx = self.m * x + 3 * self.a1 * x**2 + 4 * self.b1 * x**3 + self.c * y
        Vy = self.m * y + 3 * self.a2 * y**2 + 4 * self.b2 * y**3 + self.c * x
        return np.transpose([Vx, Vy], axes=[1, 0, 2])

    def ddpotential(self, q):
        """Hessian of the potential energy surface for the mildly anharmonic potential 2D"""
        x = q[:, 0]
        y = q[:, 1]
        Vxx = self.m + 6 * self.a1 * x + 12 * self.b1 * x**2
        Vyy = self.m + 6 * self.a2 * y + 12 * self.b2 * y**2
        Vxy = self.c
        return np.transpose([Vxx, Vxy], [Vxy, Vyy], axes=[2, 0, 1, 3])
