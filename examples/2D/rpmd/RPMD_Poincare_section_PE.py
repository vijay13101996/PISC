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

N = 10 #Number of trajectories
m = 0.5
dt = 0.005

g0 = 0.1
omega = 1
E = 4.0
Tc = 0.71143064245942033
times = 3.0
pes = coupled_harmonic(omega, g0)

potkey = 'Pullen_Edmonds_omega_{}_g0_{}'.format(omega, g0)

### Temperature is only relevant for the ring-polymer Poincare section
T = times * Tc #Temperature is not relevant in case of a classical simulation
Tkey = 'T_{}'.format(T) 

pathname = '/home/ss3120/PISC/examples/2D/rpmd/'#os.path.dirname(os.path.abspath(__file__))

xg = np.linspace(-1, 1, int(1e2)+1)
yg = np.linspace(-1, 1, int(1e2)+1)

xgrid, ygrid = np.meshgrid(xg, yg)
potgrid = pes.potential_xy(xgrid, ygrid) 


nbeads = 32
PSOS = Poincare_SOS('RPMD', pathname, potkey, Tkey)
PSOS.set_sysparams(pes, T, m, 2)
PSOS.set_simparams(N, dt, dt, nbeads = nbeads, rngSeed = 1)	
PSOS.set_runtime(50.0, 500.0)
qlist = PSOS.find_initcondn(xgrid, ygrid, potgrid, E)

PSOS.bind(qcartg = qlist, E = E, sym_init = False)

X, PX, Y, PY = PSOS.PSOS_Y(x0 = 0.0)
#print('X = ', X)
#print('PX = ', PX)
#print('Y = ', Y)
plt.scatter(X, PX, s = 1)
plt.show()

