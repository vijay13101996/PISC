###DVR functions
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from mylib.oneD import OTOC1D
#from mylib.oneD import Potentials_and_DVR as PnDVR
from mylib.oneD import DVR1D_mod
import time

OTOC1D.otoc_1d.which_version()#Old check to see if I'm using updated code
#OTOC1D.otoc_1d.test_openmpi()
time_0= time.perf_counter()

##########DVR##########

hbar=1 #like in OTOC paper
m=1/2 #like in OTOC paper
pot_name='DW'#PiB, HO, DW
print('Potential: %s' % pot_name)

H, grid, t, log_OTOC_mic, log_OTOC_T= DVR1D_mod.DVR_initialize(pot_name=pot_name, m=m,hbar=hbar)
w,v = LA.eigh(H) #scipy.space.linalg.eigsh is slower 

#####For non-default parameters:##### 
#grid= np.linspace(-3,3,1500,endpoint=False)[1:]
#t=np.linspace(0,7,500)
#w,v, log_OTOC_mic,log_OTOC_T=  DVR1D_mod.DVR_all_inclusive(pot_name,grid,t,m,hbar)

time_1=time.perf_counter()
print('Diagonalization (%i x %i) in t = %f s ' % (len(grid),len(grid),time_1-time_0))

##########Parameters for OTOC##########

N_trunc=150 #100 in OTOC paper
N_trunc +=1################Quickfix 1
X_matrix= np.zeros((N_trunc,N_trunc))
EV=v[:,:N_trunc]#eigenvector bzw wavefkt
Energies= w[:N_trunc]
N_grdpts=len(grid)
N_tsteps=len(t)
B_matrix= np.zeros((N_trunc,N_trunc,N_tsteps),dtype=complex)
Cn_Fortran = np.zeros((N_trunc, N_tsteps))
print('N_grd: %i, N_trunc: %i, timesteps: %i ' % (N_grdpts, N_trunc-1,len(t)))
labda=2
Tc=0.5*labda/np.pi

##########Calculation of OTOC##########

X_matrix=OTOC1D.otoc_1d.get_x_nm(X_matrix,EV,grid, N_grdpts,N_trunc)
time_2=time.perf_counter()
print('X matrix in t = %f s ' % (time_2-time_1))

#B_matrix = OTOC1D.otoc_1d.get_b_nm_dc(B_matrix,X_matrix, w[:N_trunc],t,N_trunc)
B_matrix = OTOC1D.otoc_1d.get_b_nm_omp(B_matrix,X_matrix, w[:N_trunc],t,N_trunc,len(t))
B_matrix = B_matrix[:-1,:-1,:]###########Quickfix 2
time_3=time.perf_counter()
print('B matrix in t = %f s' % (time_3-time_2))

#Cn_Fortran = OTOC1D.otoc_1d.get_c_n(Cn_Fortran, B_matrix,N_trunc, N_tsteps)
########Quickfix3
Cn_Fortran = OTOC1D.otoc_1d.get_c_n(Cn_Fortran[:-1,:], B_matrix,N_trunc-1, N_tsteps)
time_4=time.perf_counter()
print('C matrix in t = %f s ' % (time_4-time_3))

##########Plotting##########
from mylib.oneD import plot_my_OTOC as plt_OTOC
MC_other_parameters=()
#MC_other_parameters=(10,11,50,100,110,120,130,140)
#MC_other_parameters =  range(len(Cn_Fortran[:,1])) #(un)comment if other are (un)desired
plt_OTOC.plot_MC_OTOC(Cn_Fortran, pot_name, t=t,log_OTOC_mic=log_OTOC_mic,other_parameters=MC_other_parameters)
plt_OTOC.plot_MC_OTOC(Cn_Fortran, pot_name, t=t,log_OTOC_mic=False,other_parameters=MC_other_parameters)
plt_OTOC.plot_Thermal_OTOC(Cn_Fortran, pot_name, t=t,Energies=Energies,log_OTOC_T=log_OTOC_T,other_parameters=(0.8*Tc,0.9*Tc,1.0*Tc,10*Tc))#,other_parameters=(1,5,9,30,50,100,150))
plt.show(block=True)

####HOW Thermal OTOC is Calculated, just as a reminder what plot_Thermal_OTOC does
def C_T(C,E,beta=1): #C is microcanonical OTOC
    Z=0 
    sum=np.zeros_like(C[1,:])#get Thermal OTOC for each time t
    for n in range(len(C[:,1])):
        Z+=np.exp(-beta *E[n]) #partition sum
    for n in range(len(C[:,1])):
        for time in range(len(C[1,:])):
            sum[time] += np.exp(-beta *E[n])*C[n,time] #wights of all microcanonical OTOCS
    return sum/Z
