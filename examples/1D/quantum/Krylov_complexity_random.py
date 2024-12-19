import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import argparse
import matplotlib as mpl

neigs = 800
L = 40
omega = 1.0

neigs_arr = np.arange(1,neigs+1)

harm_vals = omega*neigs_arr - omega/2
box_vals = np.pi**2*neigs_arr**2/(2*L**2)
pow_vals = 0.05*neigs_arr**1.5

vals = harm_vals
print('vals',vals[:5])

O = np.zeros((neigs,neigs))

k_diag = neigs
for i in range(neigs):
    for j in range(i,neigs):
        #if(abs(i-j)%2==0): 
            if(abs(i-j)<=k_diag):
                O[i,j] =  1.0 #np.random.uniform(0,10)
                #O[i,j] = 1.0 #+ 0.2*np.random.normal(0,1)
                O[j,i] = O[i,j]

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

#O = O_anal

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)
L2O = np.zeros((neigs,neigs))
L2O = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,LO,L2O)

b1 = 0.0
b1 = Krylov_complexity.krylov_complexity.compute_ip(LO,LO,1.0,vals,0.5,b1,'asm')
b1 = b1**0.5
O2 = L2O/b1 - O*b1

plt.imshow(np.log(abs(O2)))
plt.show()

plt.plot(np.diag(abs(L2O/b1),6))
plt.plot(np.diag(abs(O2),6),label='k={}'.format(6))
plt.plot(np.diag(abs(O*b1),6))

plt.legend()
plt.show()

T_arr = [2.,4.,6.]#0.25,0.5,1.0,2.0]#np.arange(0.1,1.1,0.1)
mun_arr = []
mu0_harm_arr = []
mu_all_arr = []
bnarr = []

nmoments = 100
ncoeff = 200

On = np.zeros((neigs,neigs))
nmat = 10 

eBh = np.diag(np.exp(-0.5*vals/T_arr[0]))
Z = np.sum(np.exp(-vals/T_arr[0]))

ip = 'asm'

lamda = 0.0

x = 5.

def lanczos_coeffs(O, L, vals, beta, lamda, ip, ncoeff):
    bnarr = []
    mun_arr = []
    mu0_harm_arr = []
    mu_all_arr = []
    bnarr = []
    
    On = np.zeros((neigs,neigs))

    for T_au in T_arr:
        print('T',T_au)
        Tkey = 'T_{}'.format(T_au)

        beta = 1.0/T_au 

        moments = np.zeros(nmoments+1)
        moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, moments)
        even_moments = moments[0::2]

        barr = np.zeros(ncoeff)
        barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, lamda, ip)
        bnarr.append(barr)
        
        if(1):
            for nmat in range(1,11,1):#range(100):
                fig, ax = plt.subplots(1,2)
                barr_mat = np.zeros(ncoeff)
                barr_mat, On = Krylov_complexity.krylov_complexity.compute_on_matrix(O, L, barr, beta, vals, lamda, ip, On, nmat+1) 
            
                On2 = np.matmul(eBh,np.matmul(On.T,np.matmul(eBh,On)))/Z   
                #On2 = np.matmul(eBh**2,np.matmul(On.T,On))/Z
                bval = 0.0
                bval = Krylov_complexity.krylov_complexity.compute_ip(On,On,beta,vals,lamda,bval,ip)
               
                print('bn',barr[nmat],bval**0.5,np.trace(On2)**0.5) 
                
                logOn = np.abs(np.log(On))
                #print('logOn',logOn[0:5,0:5])

                #plt.imshow(np.abs(np.log(On2)))#vmax=1e4)
                ax[0].imshow((np.log(abs(On))))#vmax=1e4)
                #plt.imshow(np.real(On),vmax=20,vmin=-10)#,cmap=cmap)
                #plt.plot(np.diag(On))
                #ax[0].colorbar()
                #ax[0].title(r'$O_{:d}$'.format(nmat))
                k=50
                ax[0].hlines(k,xmin=0,xmax=neigs-10)
                ax[1].plot(np.log(abs(On[k,:])),label='nmat={}'.format(nmat))


                #plt.plot((np.diag(On2)),label='nmat={}'.format(nmat))
                #plt.hlines(np.log(sum(np.diag(On2))),xmin=0,xmax=neigs)
                plt.show()
                
                #print('On',On[0:5,0:5])
                #plt.plot(np.log(np.diag(abs(On),0)),label='nmat={}'.format(nmat))
                #trace = np.trace(On2)
                #plt.scatter(nmat,np.log(trace))
                #plt.title(r'$O_{:d}$'.format(nmat))
            #plt.legend()
            #plt.show()
            exit()

    mun_arr.append(even_moments)
    mu_all_arr.append(moments)


    mun_arr = np.array(mun_arr)
    bnarr = np.array(bnarr)

    print('bnarr',bnarr.shape,bnarr[0,:12])

    for i in range(len(T_arr)):
        plt.scatter(np.arange(1,ncoeff+1),bnarr[i,:],label='T={}'.format(T_arr[i]),s=5)
        plt.plot(np.arange(1,ncoeff+1), x*np.pi*T_arr[i]*np.arange(0,ncoeff))
    #plt.hlines(vals[k_diag-1]//2,xmin=0,xmax=ncoeff)

lanczos_coeffs(O, L, vals, 1.0/T_arr[0], 0.5/x, 'asm', ncoeff)

plt.ylim([0,vals[-1]*0.6])
plt.legend()
plt.show()
exit()

plt.scatter(np.arange(0,nmoments//2+1),np.log(mun_arr[0,:]))
plt.show()
