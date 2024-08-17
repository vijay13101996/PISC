import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt

ngrid = 1000

L = 12
lb = -L
ub = L

m=1.0
a=0.0#4
b=0.1#16
omega=0.0
n_anharm=4

pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)

potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 100
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

plt.plot(x_arr,vecs[:,-1])
plt.show()

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
O = (pos_mat)

print('O',np.around(O[:5,:5],3),'vals',vals[-1])
#exit()

mom_mat = np.zeros((neigs,neigs))
mom_mat = Krylov_complexity.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dx, mom_mat)
P = mom_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

T_arr = np.arange(1.,13.05,0.5)#[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
mun_arr = []
mu0_harm_arr = []
bnarr = []
nmoments = 80
ncoeff = 50

for T_au in T_arr:
    
    Tkey = 'T_{}'.format(T_au)

    beta = 1.0/T_au 

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')
    bnarr.append(barr)

    #print('moments',np.around(moments[-1],5))
    mun_arr.append(even_moments)

    mu0_harm_arr.append(1.0/(2*m*omega*np.sinh(0.5*beta*omega)))
    #print('mu0_harm',mu0_harm_arr[-1])

mun_arr = np.array(mun_arr)
bnarr = np.array(bnarr)
print('mun_arr',mun_arr.shape)
print('bnarr',bnarr.shape)

store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))
exit()

if(1):
    for i in [0,2,4]:#,6,8,10,12,14,16]:
        plt.scatter((np.arange(1,nmoments//2+1)),np.log(mun_arr[i,1:]),label='T={}'.format(np.around(T_arr[i],2)))
    
    #plt.xlim([10,nmoments//2])
    
    plt.title(r'$neigs={}$'.format(neigs))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mu_{2n}$')
    plt.ylim([0.0,2000])
    plt.legend()    
    plt.show()
    exit()

if(0):
    for i in [0,2,4]:#,6,8,10,12,14,16]:
        plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={}'.format(T_arr[i]))
    
    #plt.xlim([10,nmoments//2])
    
    plt.title(r'$neigs={}$'.format(neigs))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$b_n$')
    plt.ylim([0.0,2000])
    plt.legend()    
    plt.show()
    exit()

slope_arr = []
mom_list = range(0,nmoments//2+1)
for i in mom_list:#,5,6,7,8,9,10]:#,3,4]:#range(0,ncoeff,2):
    #plt.scatter(T_arr,bnarr[:,i],label='n={}'.format(i))
    
    #plt.scatter((T_arr),(mun_arr[:,i]),label='n={}'.format(2*i))
    plt.scatter(np.log(T_arr),np.log(mun_arr[:,i]),label='n={}'.format(2*i))

    lT_arr = np.log(T_arr)
    lmun_arr = np.log(mun_arr[:,i])

    p = np.polyfit(lT_arr,lmun_arr,1)

    slope_arr.append(p[0])
    
    plt.plot(lT_arr,p[0]*lT_arr+p[1],label='n={}'.format(2*i))


    # Fit mun_arr[:,i] to a T_arr**(i/2)
    #p = np.polyfit(T_arr,mun_arr[:,i],i/2)
    #print('p',p)
    #plt.plot(T_arr,p[0]*T_arr**(i/2),label='n={}'.format(2*i))

#plt.scatter(T_arr,np.array(mu0_harm_arr),label='n=0, harm',color='black')
plt.xlabel(r'$log(T)$')
plt.ylabel(r'$log(\mu_{2n})$')
plt.legend()
plt.show()

exit()

plt.scatter(mom_list,np.array(slope_arr))
plt.xlabel(r'$n$')
plt.ylabel(r'$slope$')
plt.show()

if(0):
    for i in [1,2,3]:#range(1):#nmoments//2+1):
        # Fit times_arr vs mun_arr[:,i] to a line
        p = np.polyfit(times_arr,mun_arr[:,i],1)
        print('p',p)
        plt.plot(times_arr,p[0]*times_arr+p[1],label='n={}'.format(2*i))
        plt.scatter(times_arr,mun_arr[:,i],label='n={}'.format(2*i))
    #plt.scatter(times_arr,mun_arr)
    plt.legend()
    plt.show()
    

#ncoeffs = 20
#barr = np.zeros(ncoeffs)
#barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')

#b0 = np.sqrt(1/(2*m*w*np.sinh(0.5*beta*w)))

#print('barr',barr,b0)

#plt.scatter(np.arange(ncoeffs),barr)
#plt.show()


