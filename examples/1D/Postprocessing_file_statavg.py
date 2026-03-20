import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.potentials import double_well, quartic, morse, mildly_anharmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.utils.nmtrans import FFT

dim = 1

#Double well potential
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

Tc = lamda*(0.5/np.pi)
times = 0.9
T = times*Tc
beta=1/T
print('T',T)

m = 0.5
N = 1000
dt = 0.002

time_total = 5.0#

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
pes = double_well(lamda,g)

Tkey = 'T_{}Tc'.format(times)

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
#path = '/scratch/vgs23/PISC/examples/1D'#
path = os.path.dirname(os.path.abspath(__file__))
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'stat_avg'
syskey = 'Papageno'#'Tosca2'

nbeads = 32
beadkey = 'nbeads_{}_'.format(nbeads)
Tkey = 'T_{}Tc'.format(times)

##Collect files of thermal ensembles
methodkey = 'RPMD'
enskey = 'const_q' #


kwlist = [methodkey,enskey+'_',corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt), 'Hessian']
fname  = seed_finder(kwlist,rpext)

data =[]

for f in fname:
    f = rpext +f
    data.append(np.loadtxt(f))

data = np.array(data)

rp_hess = data[0]
cent_hess = data[1]
mats_hess = data[2]

rp_lamda = np.sqrt(-rp_hess[:,1]/m)
lamda_avg = np.mean(rp_lamda)

cent_lamda = np.sqrt(-cent_hess[:,1]/m)
cent_lamda_avg = np.mean(cent_lamda)

print('# traj', len(rp_lamda))

plt.hist(rp_lamda, bins=10,color='b',label=r'${{\left \langle \frac{\partial^2 U}{\partial \xi^2} \right \rangle}}_{{Q_0}}$')
plt.axvline(x=lamda_avg, color='k')

plt.hist(cent_lamda, bins=10,color='y',label=r'${{\left \langle \frac{\partial^2 U}{\partial Q_0^2} \right \rangle}}_{{Q_0}}$')
plt.axvline(x=cent_lamda_avg, color='k')

plt.legend()
plt.show()

