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


ngrid = 1000

L=10 #4*np.sqrt(1/(4+np.pi))#10
lb=0
ub=L

lbc=0
ubc=L

m=0.5

print('L',L)

potkey = '1D_Box_m_{}_L_{}'.format(m,np.around(L,2))

T_au = 1.0
beta = 1.0/T_au
Tkey = 'T_{}'.format(T_au)

basis_N = 40
n_eigen = 35

#----------------------------------------------------------------------
def potential(x):
    if(x<lbc or x>ubc):
        return 1e12
    else:
        return 0.0

neigs = 100
potential = np.vectorize(potential)

DVR = DVR1D(ngrid, lb, ub,m, potential)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

#----------------------------------------------------------------------
vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:]/L)

vecs = vecs_anal
vals = vals_anal

#----------------------------------------------------------------------
x_arr = DVR.grid[1:DVR.ngrid]

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0.0,1.0,101) #(-0.1,0.1,201)
C_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'qq_TCF'#'OTOC'#
enskey = 'Kubo'#

corrcode = {'OTOC':'xxC','qq_TCF':'qq1','qp_TCF':'qp1'}
enscode = {'Kubo':'kubo','Standard':'stan'}	

if(0):
    for i in range(len(t_arr)):
        lamda = t_arr[i]
        C_arr[i] = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_elts(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,0.0,beta,n_eigen,corrcode[corrkey],lamda,C_arr[i]) 
        #C_arr[i] = OTOC_f_1D_omp_updated.otoc_tools.wightman_corr_elts(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,0.0,beta,n_eigen,corrcode[corrkey],lamda,C_arr[i])

    #fname = 'Wightman_Im_qqTCF_{}_{}_{}'.format(potkey,enskey,Tkey)
    fname = 'Im_qqTCF_{}_{}_{}'.format(potkey,enskey,Tkey)
    path = os.path.dirname(os.path.abspath(__file__))	
    store_1D_plotdata(t_arr,C_arr,fname,'{}/Datafiles'.format(path))

    fig,ax = plt.subplots()
    plt.plot(t_arr,C_arr)

    #fig.savefig('/home/vgs23/Images/Im_qqTCF.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)	
    plt.show()
    exit()

if(1):
    times_arr = [0.05,0.1,0.2,0.4,0.6,1.0]#,1.0,2.0,4.0,10.0]
    #[0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]#,2.0,2.5,3.0,3.5,4.0]
    #[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
    b1_arr = []
    for times in times_arr:
    #for times in [1.0]:#[0.5,1.0,2.0,3.0,4.0]: #,5.0,6.0,7.0,8.0]: #[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]: #
        Tkey = 'T_{}'.format(times)
        T = times

        #data = read_1D_plotdata('{}/Datafiles/Wightman_Im_qqTCF_{}_{}_{}.txt'.format(path,potkey,enskey,Tkey))
        data = read_1D_plotdata('{}/Datafiles/Im_qqTCF_{}_{}_{}.txt'.format(path,potkey,enskey,Tkey))
        
        t_arr = data[:,0]
        C_arr = data[:,1]

        #Plot first derivative
        der = T*np.gradient(C_arr,t_arr[1]-t_arr[0])
        derder = np.gradient(der,t_arr[1]-t_arr[0])

        #Fit derivative to linear function
        popt,pcov = curve_fit(lambda x,l: l*x,t_arr,der) 

        b1_arr.append(popt[0])

        #plt.scatter(times,abs(der[0]))
        #plt.plot(t_arr,der,label='T = {}Tc, der'.format(times))
        plt.plot(t_arr,C_arr,label='T = {}Tc'.format(times))
    
    plt.legend()
    plt.show()
    exit()

    popt,perr = curve_fit(lambda x,l,c: l*x+c,times_arr,b1_arr)

    plt.scatter(times_arr,b1_arr)
    plt.plot(times_arr,b1_arr)
    plt.plot(times_arr,popt[0]*np.array(times_arr) + popt[1])
    plt.plot(times_arr,np.pi*np.array(times_arr)*Tc)
    
    print('popt',popt,np.pi*Tc)
    plt.show()


