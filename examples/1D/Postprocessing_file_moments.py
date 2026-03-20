import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.potentials import double_well, quartic, morse, mildly_anharmonic, harmonic1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.utils.nmtrans import FFT

dim = 1

if(0): #Double well potential
    lamda = 2.0
    g = 0.02
    Vb = lamda**4/(64*g)

    Tc = lamda*(0.5/np.pi)
    times = 4.0#0.95
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

    T = 2.5#0.125

    m = 1.0
    N = 1000
    dt_therm = 0.05
    dt = 0.01

    time_therm = 50.0
    time_total = 30.0

    potkey = 'quartic_a_{}'.format(a)
    Tkey = 'T_{}'.format(T)

if(1): #Mildly anharmonic
    a = 0.0
    b = 1.0
    w = 0.0
    n_anh = 4

    T = 10.0 
    beta = 1/T

    m = 1.0
    N = 1000
    dt_therm = 0.05
    dt = 0.01

    time_therm = 50.0
    time_total = 30.0

    pes = mildly_anharmonic(m,a,b,w,n=n_anh)

    potkey = 'mildly_anharmonic_a_{}_b_{}_n_{}_w_{}'.format(a,b,n_anh,w)
    Tkey = 'T_{}'.format(np.around(T,3))

if(0):
    m=1.0
    omega = 1.0
    
    pes = harmonic1D(m,omega)
    potkey = 'harmonic_omega_{}'.format(omega)
    
    T = 1.
    Tkey = 'T_{}'.format(T)

#Path extensions
#path = '/scratch/vgs23/PISC/examples/1D'#
path = os.path.dirname(os.path.abspath(__file__))
rpext = '{}/rpmd/Datafiles/'.format(path)

dt = 0.01
nbeads = 15

#Simulation specifications
methodkey = 'RPMD'
enskey = 'thermal'
corrkey = 'ImTCF_moments'
syskey = 'Selene'
beadkey = 'nbeads_{}_'.format(nbeads)


for T in [10.0,20.0,40.0,100.0]:
    Tkey = 'T_{}'.format(T)
    beta = 1/T
    omegan = nbeads/beta

    n = [0]
    for i in range(1, nbeads // 2 + 1):
        n.append(-i)
        n.append(i)
    if nbeads % 2 == 0:
        n.pop(-2)
    wk = np.sin(np.array(n) * np.pi / nbeads) * (2*omegan)

    kwlist = [methodkey,enskey+'_',corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]
    fname  = seed_finder(kwlist,rpext)[0]

    mom_array = np.loadtxt(rpext+fname)[:,1:]
    mom_num = np.mean(mom_array,axis=0)
    
    x2 = mom_num[1:]*wk[1:]**6
    
    for nmom in [4]:#,4,6,8]:
        xp2 = mom_num[2::2]*wk[2::2]**0#nmom
        xm2 = mom_num[1::2]*wk[1::2]**0#nmom
        xc2 = mom_num[0]

        xm2_odd = xm2[0::2]
        xm2_even = xm2[1::2]
        
        nmom_arr = np.arange(1,len(xp2))
        log_nmom_arr = np.log(nmom_arr)
        log_xp2 = (np.log(xp2)[:-1])
        log_anal = (np.log(T)+np.log(wk[2::2][:-1]**(nmom-2)))

    
        plt.scatter(nmom_arr,xp2[:-1]-1/(m*beta*(wk[2::2][:-1]**2)),label='T={},w={}'.format(T,w))
        #plt.scatter(nmom_arr,(xp2)[:-1],label='T={},w={}'.format(T,w))
        #plt.plot(nmom_arr,1/(m*beta*(wk[2::2]**2))[:-1])

        ### Comparing the moments with the analytical expression
        #plt.scatter(nmom_arr,(log_xp2),label='T={},w={}'.format(T,w))#n_anh))
        #plt.plot(nmom_arr,(log_anal))

        #plt.scatter(log_nmom_arr,abs((log_xp2)-(log_anal)),label='T={},w={}'.format(T,w))

        #plt.plot(log_nmom_arr,(abs(log_xp2-log_anal)),label='T={},w={}'.format(T,w)) #n_anh))

        print('diff, nmom',log_xp2[1], log_anal[1],nmom, log_xp2[1]-log_anal[1])

        #plt.scatter(np.log(nmom_arr),np.log(xp2)[:-1],label='T={},w={}'.format(T,w))#n_anh))
        #plt.plot(np.log(nmom_arr),np.log(xp2)[:-1],label='T={},w={}'.format(T,w))#n_anh))
        #plt.plot(np.log(nmom_arr),np.log(T)+np.log(wk[2::2][:-1]**(nmom-2)))
        
        # Fit log nmom vs log xp2
        p = np.polyfit(np.log(nmom_arr)[:-3],np.log(xp2)[:-4],1)
        #plt.plot(np.log(nmom_arr),np.polyval(p,np.log(nmom_arr)),label='slope={}'.format(p[0]))
        #plt.scatter(np.log(np.arange(1,len(xm2))),np.log(xm2)[:-1],label='T={},n={}'.format(T,n))

plt.xscale('log')
plt.legend()
plt.show()

if(0):
    mom_anal = 1/(m*beta*(wk**2+omega**2))

    for seedcount in [10,20,30,40,50,60,70,80,90,100]:
        mom_array = np.loadtxt(rpext+fname)[:seedcount+1,1:]
        mom_num = np.mean(mom_array,axis=0)

        xp2 = mom_num[2::2]
        xm2 = mom_num[1::2]
        xc2 = mom_num[0]

        xm2_odd = xm2[0::2]
        xm2_even = xm2[1::2]

        mu0_anal = 1/(2*omega*np.sinh(beta*omega/2))
        mu0_num = xc2 - 2*(xm2_odd.sum() - xm2_even.sum())

        #print('mu0',mu0_anal,mu0_num)
        percent_error = np.abs((mu0_num-mu0_anal)/mu0_anal)*100
        #print('seedcount,percent_error',seedcount,percent_error)

#---------------------------------------------------
#print('moments_std',moments_std)

#moments_std = np.std(mom_array,axis=0)
#print('moments',mom_num,mom_anal)
#percent_error = np.abs((mom_num-mom_anal)/mom_anal)*100
#print('percent_error',percent_error)

#plt.hist(mom_array[:,0],bins=50)
#plt.show()

#print('mom_array shape',mom_array.shape,mom_array)

