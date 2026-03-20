import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.potentials import double_well, quartic, morse, mildly_anharmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.utils.nmtrans import FFT

dim = 1

if(1): #Double well potential
    lamda = 2.0
    g = 0.08
    Vb = lamda**4/(64*g)

    Tc = lamda*(0.5/np.pi)
    times = 0.95
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

if(0): #Quartic
    a = 1.0

    pes = quartic(a)

    T = 0.125

    m = 1.0
    N = 1000
    dt_therm = 0.05
    dt = 0.01

    time_therm = 50.0
    time_total = 20.0

    potkey = 'quartic_a_{}'.format(a)
    Tkey = 'T_{}'.format(T)

if(0): #Mildly anharmonic
    omega = 1.0
    a = 0.4#-0.605#0.5#
    b = a**2#0.427#a**2#

    T = 1.0#times*Tc
    beta = 1/T

    m=1.0
    N = 1000
    dt_therm = 0.05
    dt = 0.002

    time_therm = 50.0
    time_total = 30.0

    pes = mildly_anharmonic(m,a,b)

    potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)
    Tkey = 'T_{}'.format(np.around(T,3))

if(0): #Morse_SB
    m=1.0
    delta_anh = 0.05#1
    w_10 = 1.0
    wb = w_10
    wc = w_10 + delta_anh
    alpha = (m*delta_anh)**0.5
    D = m*wc**2/(2*alpha**2)

    pes = morse(D,alpha)
    T = 1.0#TinK*K2au
    beta = 1/T

    potkey = 'Morse_D_{}_alpha_{}'.format(D,alpha)
    Tkey = 'T_{}'.format(T)
    dt = 0.002
    time_total = 20.0

if(0): #Morse
    m=0.5
    D = 9.375
    alpha = 0.382
    pes = morse(D,alpha)

    w_m = (2*D*alpha**2/m)**0.5
    Vb = D/3

    print('alpha, w_m', alpha, Vb/w_m)
    T = 3.18#*0.3
    beta = 1/T
    potkey = 'morse'
    Tkey = 'T_{}'.format(T)

    N = 1000
    dt_therm = 0.05
    dt = 0.02
    time_therm = 50.0
    time_total = 5.0

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
corrkey = 'fd_OTOC'#'qq_TCF'#'R2'#'singcomm' #
syskey = 'Papageno'#'Tosca2'

nbeads = 32
beadkey = 'nbeads_{}_'.format(nbeads)
 
methodkey = 'RPMD'
enskey = 'thermal' #'const_q' #
kwlist = [methodkey,enskey+'_',corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]

seedarr = [1000,1400,1800,2200,2600,3000]#,1400,1800,2200,2600,3000]#[100,200,300,400,500,600,700,800,900,1000]
for seedcount in seedarr:
    tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr,allseeds=False,seedcount=seedcount)
    store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext)

    ext = rpext + 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,'fd_OTOC',potkey,Tkey,nbeads,dt)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.3,4.2)
    plt.plot(tarr,np.log(abs(OTOCarr)),label='$N_{{traj}}$ = {}, $\lambda$={}'.format(seedcount*1000, np.around(slope,2)))
    if(seedcount ==3000):
        plt.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
plt.legend()
plt.title('RPMD OTOCs (finite difference) at T = {}$T_c$'.format(times))
plt.show()

exit()
   
