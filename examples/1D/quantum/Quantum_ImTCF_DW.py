import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well, morse, quartic, asym_double_well, harmonic1D
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from scipy.optimize import curve_fit
import argparse

ngrid = 600

L = 10.0
lb = -L
ub = L
m = 0.5

lamda = 2.0
g = 0.08

def main(lamda,g,ip,lg,times):
    pes = double_well(lamda,g)

    Tc = lamda*(0.5/np.pi)    
    T_au = times*Tc

    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
    Tkey = 'T_{}Tc'.format(times)	

    #print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))
    beta = 1.0/T_au 
    print('T in au, beta',T_au, beta, times) 
    print('ip, local/global',ip,lg)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
    vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

    x_arr = DVR.grid[1:DVR.ngrid]
    basis_N = 50
    n_eigen = 30

    k_arr = np.arange(basis_N) +1
    m_arr = np.arange(basis_N) +1

    if(ip == 'wightman'):
        if (lg=='local'):
            print('Local')
            t_arr = np.linspace(-0.1,0.1,201)
        else:
            print('Global')
            t_arr = np.linspace(-0.5,0.5,1001)
    else:
        t_arr = np.linspace(0,1.0,1001)
    C_arr = np.zeros_like(t_arr) + 0j

    path = os.path.dirname(os.path.abspath(__file__))

    for i in range(len(t_arr)):
        lamda = t_arr[i]
        if(ip != 'wightman'):
            C_arr[i] = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_elts(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,0.0,beta,n_eigen,'qq1',lamda,C_arr[i]) 
        else:
            C_arr[i] = OTOC_f_1D_omp_updated.otoc_tools.wightman_corr_elts(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,0.0,beta,n_eigen,'qq1',lamda,C_arr[i])

    if(ip == 'wightman'):
        if (lg=='local'):
            fname = 'Wightman_local_Im_qqTCF_{}_{}'.format(potkey,Tkey)
        else:
            fname = 'Wightman_Im_qqTCF_{}_{}'.format(potkey,Tkey)
    else:
        fname = 'Im_qqTCF_{}_{}'.format(potkey,Tkey)

    path = os.path.dirname(os.path.abspath(__file__))	
    store_1D_plotdata(t_arr,C_arr,fname,'{}/Datafiles'.format(path))

    #fig,ax = plt.subplots()
    #plt.plot(t_arr,C_arr)
    #fig.savefig('/home/vgs23/Images/Im_qqTCF.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)	
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lamda', '-l',type=float, default=2.0)
    parser.add_argument('--g', '-g', type=float, default=0.08)
    parser.add_argument('--ip', type=str, default='wightman')
    parser.add_argument('--lg', type=str, default=True)
    parser.add_argument('--times', type=float, default=1.0)
    args = parser.parse_args()

    main(args.lamda,args.g,args.ip,args.lg,args.times)

if(0):
    times_arr = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,2.0,2.5,3.0,3.5,4.0]
    #[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
    b1_arr = []
    for times in times_arr:
    #for times in [1.0]:#[0.5,1.0,2.0,3.0,4.0]: #,5.0,6.0,7.0,8.0]: #[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]: #
        Tkey = 'T_{}Tc'.format(times)
        T = times*Tc

        data = read_1D_plotdata('{}/Datafiles/Wightman_Im_qqTCF_{}_{}_{}.txt'.format(path,potkey,enskey,Tkey))
        #data = read_1D_plotdata('{}/Datafiles/Im_qqTCF_{}_{}_{}.txt'.format(path,potkey,enskey,Tkey))
        
        t_arr = data[:,0]
        C_arr = data[:,1]

        #Plot first derivative
        der = T*np.gradient(C_arr,t_arr[1]-t_arr[0])
        derder = np.gradient(der,t_arr[1]-t_arr[0])

        #Fit derivative to linear function
        popt,pcov = curve_fit(lambda x,l: l*x,t_arr,der) 

        b1_arr.append(popt[0])

:q
:q
:q!
        #plt.scatter(times,abs(der[0]))
        #plt.plot(t_arr,der,label='T = {}Tc, der'.format(times))
        plt.plot(t_arr,C_arr,label='T = {}Tc'.format(times))
    
    plt.legend()
    plt.show()

    plt.scatter(times_arr,b1_arr)
    plt.plot(times_arr,b1_arr)
    plt.plot(times_arr,2*np.pi*np.array(times_arr)*Tc)
    plt.show()


