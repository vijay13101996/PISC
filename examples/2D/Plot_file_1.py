import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim = 1

# Double Well potential
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

z = 1.0#0.5
 
Tc = 0.5*lamda/np.pi
times = 0.95
T = times*Tc
beta=1/T
Tkey = 'T_{}Tc'.format(times)

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002#05
time_therm = 50.0
time_total = 5.0#5.0

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
enskey = 'const_q'
tarr = np.arange(0,5,dt)

#Path extensions
path = '/home/vgs23/PISC/examples/2D/'#
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

fig,ax = plt.subplots()

td = 1
tu = 2

for nbeads,c in zip([8],['r','b','g']):
    ext = rpext + 'RPMD_{}_OTOC_{}_{}_nbeads_{}_dt_{}'.format(enskey,potkey,Tkey,nbeads,dt)
    plot_1D(ax,ext, label='{} beads'.format(nbeads),color=c, log=True,linewidth=1)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2,3)
    ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color=c)

for nbeads,c in zip([8],['b','g']):
    ext = rpext + 'RPMD_{}_OTOC_{}_{}_nbeads_{}_dt_{}'.format(enskey+'p',potkey,Tkey,nbeads,dt)
    plot_1D(ax,ext, label='{} beads'.format(nbeads),color=c, log=True,linewidth=1)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
    ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color=c)


#plt.plot(tarr,np.log(1 + 2*lamda*tarr + 2*lamda**2*tarr**2),'k--',label='$\sim t^2$')
#plt.plot(tarr, np.log(np.cosh(lamda*tarr)),'k--',label='$\sim \log(\cosh(\lambda t))$')
plt.legend()
plt.show()


