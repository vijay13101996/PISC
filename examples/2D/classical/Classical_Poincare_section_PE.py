import numpy as np
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
#from Saddle_point_finder import separatrix_path, find_minima
import time
import os
import matplotlib

N = 20 #Number of trajectories
m = 0.5
dt = 0.005

g0 = 0.1
omega = 1.0
E = 4.0
pes = coupled_harmonic(omega, g0)

potkey = 'Pullen_Edmonds_omega_{}_g0_{}'.format(omega, g0)

### Temperature is only relevant for the ring-polymer Poincare section
T = 1.0 #Temperature is not relevant in case of a classical simulation
Tkey = 'T_{}'.format(T) 

#times = 0.05
#T = times * Tc
#Tkey = 'T_{}Tc'.format(times)

pathname = '/home/ss3120/PISC/examples/2D/classical/'#os.path.dirname(os.path.abspath(__file__))

#xg = np.linspace(-1, 1, int(1e2)+1)
#yg = np.linspace(-1, 1, int(1e2)+1)

xg = np.linspace(-8, 8, int(1e2) + 1)
yg = np.linspace(-8, 8, int(1e2) + 1)


xgrid, ygrid = np.meshgrid(xg, yg)
potgrid = pes.potential_xy(xgrid, ygrid) 


nbeads = 1
PSOS = Poincare_SOS('Classical', pathname, potkey, Tkey)
PSOS.set_sysparams(pes, T, m, 2)
PSOS.set_simparams(N, dt, dt, nbeads = nbeads, rngSeed = 1)	
PSOS.set_runtime(50.0, 500.0)
qlist = PSOS.find_initcondn(xgrid, ygrid, potgrid, E)
#print(qlist.shape)

X = qlist[:, 0]
Y = qlist[:, 1]
X, Y = np.meshgrid(X, Y)
h = pes.ddpotential_xy(X, Y)
h = np.transpose(h, (0, 3, 1, 2))

eigrid = np.sort(np.linalg.eigvals(h), axis = 2)
print(np.absolute(np.min(eigrid[:, :, 0])))
omega1 = np.sqrt(np.absolute(np.min(eigrid[:, :, 0]))/m)
Tc = omega1 / (2 * np.pi)
print(Tc)


PSOS.bind(qcartg = qlist, E = E, specific_traj = 19, sym_init = False)
X, PX, Y, PY = PSOS.PSOS_Y(x0 = 0.0)
plt.scatter(X, PX, s = 1)
plt.title('Classical')
#plt.title('RPMD T = 0.05$T_c$')
plt.xlabel(r'x')
plt.ylabel(r'$p_x$')
plt.show()

