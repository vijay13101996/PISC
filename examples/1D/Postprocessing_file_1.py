import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.potentials import double_well, quartic, morse, mildly_anharmonic, harmonic1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.utils.nmtrans import FFT

dim = 1
m = 1.0

if(1):#Double well potential
    m=0.5
    lamda = 2.0
    g = 0.08
    Vb = lamda**4/(64*g)

    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
    pes = double_well(lamda,g)
    
    Tc = lamda*(0.5/np.pi)
    times = 0.95
    T = times*Tc
    beta=1/T
    print('T',T)

    Tkey = 'T_{}Tc'.format(times)

if(0):  #harmonic oscillator
    m = 1.0
    omega = 1.0
    pes = harmonic1D(m,omega)
    potkey = 'harmonic'

    T = 1.0
    Tkey = 'T_{}'.format(T)

if(1): #quartic
    m = 1.0
    a = 1.0
    pes = quartic(a)
    potkey = 'quartic'

    T = 1.0/8
    Tkey = 'T_{}'.format(T)

N = 1000
dt = 0.01
time_total = 20.0

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
corrkey = 'qq_TCF'#'OTOC'#'R2'#'singcomm' #
syskey = 'Papageno'#'Tosca2'

#CMD
nbeads = 8
beadkey = 'nbeads_{}_'.format(nbeads)
if(1): ##Collect files of thermal ensembles
    methodkey = 'CMD'
    gamma = 16
    enskey = 'thermal'#'const_q'#

    kwlist = [methodkey,enskey+'_',corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt),'gamma_{}'.format(gamma)]

    tarr,OTOCarr,stdarr = seed_collector(kwlist,cext,tarr,OTOCarr)

    if(corrkey!='OTOC'):
        OTOCarr/=nbeads
        plt.plot(tarr,OTOCarr)
    else:
        plt.plot(tarr,np.log(abs(OTOCarr)))
    store_1D_plotdata(tarr,OTOCarr,'CMD_{}_{}_{}_{}_nbeads_{}_dt_{}_gamma_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt,gamma),cext)

    plt.show()
    exit()
    ext = cext + 'CMD_{}_{}_{}_{}_nbeads_{}_dt_{}_gamma_{}'.format(enskey,'OTOC',potkey,Tkey,nbeads,dt,gamma)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.3,3.3)
    plt.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='m')

    plt.show()

