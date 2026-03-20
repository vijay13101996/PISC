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
times = 0.9#0.95
T = times*Tc
beta=1/T
print('T',T)

m = 0.5
N = 1000
dt = 0.005
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
corrkey = 'OTOC'#'R2'#'singcomm' #
syskey = 'Papageno'#'Tosca2'

nbeads = 32
beadkey = 'nbeads_{}_'.format(nbeads)
Tkey = 'T_{}Tc'.format(times)

##Collect files of thermal ensembles
methodkey = 'RPMD'
enskey = 'thermal' #'const_q' #

kwlist = [methodkey,enskey+'_',corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]

for seedcount in [1000,1400,1800,2600,3000]:#[600,800,1000]:
    tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr,allseeds=False,seedcount=seedcount)
    store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext)

    ext = rpext + 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,'OTOC',potkey,Tkey,nbeads,dt)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,4.05,4.9)#3.4,4.4)
    
    plt.plot(tarr,np.log(abs(OTOCarr)),label='#seeds = {}'.format(seedcount))
plt.plot(t_trunc, slope*t_trunc+ic,linewidth=1.5,color='k')

plt.legend()
plt.show()


