import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim=2

alpha = 0.363
D = 10.0 

lamda = 2.0
g = 0.08

z = 1.5
 
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 10.0

nbeads = 4

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'OTOC'#'qq_TCF'#
beadkey = 'nbeads_{}_'.format(nbeads)
Tkey = 'T_{}Tc'.format(times)
syskey = 'Selene'

fig,ax = plt.subplots()

if(0):	
	ext = 'Classical_OTOC_{}_T_{}Tc_dt_{}'.format(potkey,times,dt)
	extclass = Cext + ext
	print('fname',extclass)
	plot_1D(ax,extclass, label=r'$Classical,T={}Tc$'.format(times),color='g', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(extclass,6.2,8.)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')


if(1):
	ext ='RPMD_{}_{}_{}_nbeads_{}_dt_{}'.format(corrkey,potkey,Tkey,nbeads,dt)
	extclass = rpext + ext
	print('fname',extclass)
	plot_1D(ax,extclass, label=r'$RPMD,T={}Tc, N_b={}$'.format(times,nbeads),color='g', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(extclass,5,8)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

plt.legend()
plt.show()

