import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.Quartic import quartic
from PISC.potentials.Morse_1D import morse
from PISC.potentials import harmonic1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
#import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope

ngrid = 400

L = 8
lb = -L
ub = L

m = 1.0
omega1 = 1.0
xw = 7
omega2 = xw*omega1

E = 50.0

#Find all multiples of omega2 that are less than E
Nx = 0
while True:
    Nx += 1
    if (Nx+1)*omega2 > E:
        break

Nxlist = np.arange(0,Nx+1)
Nylist = int(E) - 7*Nxlist-1

print('Ny',Nylist, Nxlist)

pes1 = harmonic1D(m,omega1)
pes2 = harmonic1D(m,omega2)
potkey = 'harmonic'


DVR1 = DVR1D(ngrid,lb,ub,m,pes1.potential)
vals1,vecs1 = DVR1.Diagonalize() 

DVR2 = DVR1D(ngrid,lb,ub,m,pes2.potential)
vals2,vecs2 = DVR2.Diagonalize()

x_arr1 = DVR1.grid
x_arr2 = DVR2.grid

T_au = 5
beta = 1.0/T_au 
print('T in au, beta',T_au, beta)

Psi_MES = np.zeros((ngrid,ngrid))
for nx,ny in zip(Nxlist,Nylist):
    psi_x = vecs1[:,nx]
    psi_y = vecs2[:,ny]
    
    #psi = np.zeros((ngrid,ngrid))
    #for i in range(ngrid):
    #    for j in range(ngrid):
    #        psi[i,j] = psi_x[i]*psi_y[j]

    Psi_MES += np.outer(psi_x,psi_y)


norm = np.sqrt(np.sum(np.abs(Psi_MES)**2*DVR1.dx*DVR2.dx))
Psi_MES /= norm
#print('norm',norm)

rho = np.abs(Psi_MES)**2

#plt.imshow(abs(Psi_MES)**2,origin='lower',extent=[lb,ub,lb,ub])
#plt.show()
#exit()

#Plot 1D section of Psi_MES at x = x0
x0 = 0
ix0 = np.argmin(np.abs(x_arr1-x0))
#plt.plot(x_arr2,abs(Psi_MES[ix0,:])**2)

#plt.show()

#---------------------------------------------------------------

# Compute environment density matrix (along y axis)
if(1):
    rho_env = np.zeros((ngrid,ngrid))
    #rho_env_temp = np.zeros((ngrid,ngrid))

    nvals = 20
    #print('nvals',vals1[nvals])
    vals_env = E-vals1[:nvals]
    vals_env = np.sort(vals_env) #- vals_env[0]

    vecs_env = np.zeros((ngrid,nvals))
    vecs_env = vecs1[:,:nvals]
    vecs_env = vecs_env[:,::-1]

    for n in range(nvals):
        for i in range(DVR1.ngrid):
            for j in range(DVR1.ngrid):
                rho_env[i,j] += vecs_env[i,n]*vecs_env[j,n]*np.exp(-beta*(vals_env[n]))
                #rho_env_temp[i,j] += vecs_env[i,n]*vecs_env[j,n]*np.exp(beta*(vals_env[n]))

    plt.imshow(rho_env,extent=[lb,ub,lb,ub])
    plt.show()

    exit()

#---------------------------------------------------------------
# Compute system density matrix (along x axis) 

if(1):
    nvals = 20
    #print('nvals',vals2[nvals])
    vals_sys = vals2[:nvals]

    vecs_sys = np.zeros((ngrid,nvals))
    vecs_sys = vecs2[:,:nvals]

    rho_sys = np.zeros((ngrid,ngrid))
    rho_sys_temp = np.zeros((ngrid,ngrid))

    for n in range(nvals):
        for i in range(DVR2.ngrid):
            for j in range(DVR2.ngrid):
                rho_sys[i,j] += vecs_sys[i,n]*vecs_sys[j,n]*np.exp(-beta*(vals_sys[n]))
                rho_sys_temp[i,j] += vecs_sys[i,n]*vecs_sys[j,n]*np.exp(beta*(vals_sys[n]))

    Znum = np.trace(rho_sys)*DVR2.dx
    Z = 1/(2*np.sinh(beta*omega2/2))

    #rho_sys/=Znum

    Z_mb_num = np.trace(rho_sys_temp)*DVR2.dx
    Z_mb = 1/(2*np.sinh(-beta*omega2/2))

    #rho_sys_temp/=Z_mb_num

    print('Z,trace',Z,Znum)
    print('Z_mb,trace',Z_mb,Z_mb_num)
    #print('Z_mb,trace',Z_mb,np.trace(rho_sys_temp)*DVR2.dx)


    rho = np.matmul(rho_sys,rho_sys_temp)

    ###### Start by understanding why the diagonal elements of rho are not identical.

    #plt.plot(x_arr1,np.diag(rho),color='red')
    #plt.show()

    print('trace rho',np.trace(rho)*DVR1.dx*DVR2.dx)

    rho_id = np.zeros((ngrid,ngrid))
    for n in range(nvals):
        for i in range(DVR2.ngrid):
            for j in range(DVR2.ngrid):
                rho_id[i,j] += vecs_sys[i,n]*vecs_sys[j,n]


    print('trace id',np.trace(rho_id)*DVR1.dx)
    
    plt.imshow(rho,extent=[lb,ub,lb,ub])
    plt.show()

    plt.imshow(rho_id,extent=[lb,ub,lb,ub])
    plt.show()

    exit()

#---------------------------------------------------------------
# Compute system density matrix (along x axis) using MES
rho_sys_rel = np.zeros((ngrid,ngrid))

for i in range(ngrid):
    for j in range(ngrid):
        rhoij = np.matmul(np.matmul((Psi_MES[:,i]),rho_env),Psi_MES[:,j])*DVR1.dx*DVR1.dx
        rho_sys_rel[i,j] = rhoij
        
        #print('i,j,rhoij',i,j,rhoij)

print('trace rho_sys_rel',np.trace(rho_sys_rel)*DVR2.dx)
#plt.plot(x_arr1,np.diag(rho_sys_rel))
#plt.show()


plt.imshow(rho_sys_rel,extent=[lb,ub,lb,ub])
plt.show()







