import numpy as np
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Henon_Heiles import henon_heiles
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
#from Saddle_point_finder import separatrix_path, find_minima
import time
import os
import matplotlib

N = 30 #Number of trajectories
m = 1.0
dt = 0.005

lamda = 1.0
g = 0.0

E = 0.125
pes = henon_heiles(lamda, g)

potkey = 'Henon_Heiles_lamda_{}_g_{}'.format(lamda, g)

### Temperature is only relevant for the ring-polymer Poincare section
T = 1.0 #Temperature is not relevant in case of a classical simulation
Tkey = 'T_{}'.format(T) 

pathname = '/home/ss3120/PISC/examples/2D/classical/'#os.path.dirname(os.path.abspath(__file__))

xg = np.linspace(-1, 1, int(1e2)+1)
yg = np.linspace(-1, 1, int(1e2)+1)

xgrid,ygrid = np.meshgrid(xg, yg)
potgrid = pes.potential_xy(xgrid, ygrid) 


nbeads = 1
PSOS = Poincare_SOS('Classical', pathname, potkey, Tkey)
PSOS.set_sysparams(pes, T, m, 2)
PSOS.set_simparams(N, dt, dt, nbeads=nbeads, rngSeed=1)	
PSOS.set_runtime(50.0, 500.0)
qlist = PSOS.find_initcondn(xgrid, ygrid, potgrid, E)

PSOS.bind(qcartg = qlist, E = E, sym_init = False)

X,PX,Y = PSOS.PSOS_Y(x0 = 0.0)
plt.scatter(X, PX, s=1)
plt.show()

