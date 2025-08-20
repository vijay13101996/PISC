import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
import argparse
import matplotlib as mpl

m=1
neigs = 800
L = 40
omega = 1.0

neigs_arr = np.arange(1,neigs+1)

harm_vals = omega*neigs_arr - omega/2

vals = harm_vals
print('vals',vals[:5])

O = np.zeros((neigs,neigs)) + 1j*0.0

def comp_pos_mat(i,j):
            if(i==j):
                return 0.0
            elif(i-j==1):
                return np.sqrt(1/(2*m*omega))*np.sqrt(j+1)
            elif(j-i==1):
                return np.sqrt(1/(2*m*omega))*np.sqrt(i+1)
            else:
                return 0.0

pos_mat_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        pos_mat_anal[i,j] = comp_pos_mat(i,j)

O = pos_mat_anal
for i in range(3):
    O = np.matmul(O,O)


liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

nmoments = 8
ncoeff = nmoments//2+1
moments = np.zeros(nmoments+1)

T_arr = [2.0]#,4.0,6.0]

b0 = 0.0
beta = 1.0/T_arr[0]
b0 = Krylov_complexity.krylov_complexity.compute_ip(O,O,beta,vals,0.5,b0,'asm')

O/=np.sqrt(b0)

print('O ip',Krylov_complexity.krylov_complexity.compute_ip(O,O,beta,vals,0.5,0.0,'asm'))


def compute_moments(O, vals, beta, lamda, ip, nmoments):
    mun_arr = []
    bnarr = []
    On = np.zeros((neigs,neigs))

    Z = 0.0
    for i in range(neigs):
        Z += np.exp(-beta*vals[i])
    

    for T_au in T_arr:
        print('T',T_au)
        Tkey = 'T_{}'.format(T_au)

        beta = 1.0/T_au 

        moments = np.zeros(nmoments+1)
        moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, ip, lamda, moments)
        even_moments = moments[0::2]

        barr = np.zeros(ncoeff)
        barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, lamda, ip)
        bnarr.append(barr)
        
        print('moments',moments[::2])
        mun_arr.append(moments)
   
    mun_arr = np.array(mun_arr)
    bnarr = np.array(bnarr)
    
    for i in range(len(T_arr)):
        #plt.scatter(np.arange(0,nmoments+1),np.log(mun_arr[i,:]),label='T={}'.format(T_arr[i]),s=10)
        plt.scatter(np.arange(0,ncoeff),(bnarr[i,:]),label='T={}'.format(T_arr[i]),s=10)

    return mun_arr, bnarr, Z

def mom_to_bn(even_moments):
    ncoeff = len(even_moments) 
    bnarr = np.zeros(ncoeff)
    Marr = np.zeros((ncoeff,ncoeff)) # index l is the row, index j is the column
    Marr[:,0] = even_moments # The M matrix is to be filled from left to the diagonal, first column is the moments

    bnarr[0] = 1.0 # We assume that the operator is normalized

    for l in range(1,ncoeff): # b0 and M[0,0] are already set, and the first row is filled until the diagonal.
        for j in range(1,l+1): # First column is already filled, so we start from j=1
            if (j==1): # Fill the first column
                Marr[l,j] = Marr[l,j-1]/bnarr[j-1]**2 # M[:,-1] is set to zero by default
            else:
                Marr[l,j] = Marr[l,j-1]/bnarr[j-1]**2 - Marr[l-1,j-2]/bnarr[j-2]**2
        
        bnarr[l] = np.sqrt(Marr[l,l]) # The diagonal element is the next b

    return bnarr
        
mun_arr, bnarr, Z = compute_moments(O, vals, 1.0/T_arr[0], 0.5, 'wgm', nmoments)

bn_mom = mom_to_bn(mun_arr[0,0::2])

print('bnarr',bnarr[0,:5])
print('bn_mom',bn_mom[:5])

exit()

hankel = np.zeros((ncoeff,ncoeff))
for i in range(ncoeff):
    for j in range(ncoeff):
        if((i+j)%2==0):
            hankel[i,j] = mun_arr[0,i+j]
        else:
            hankel[i,j] = 0.0

print('hankel',hankel[:5,:5], hankel.shape,bnarr.shape)

b0sq = bnarr[0,0]**2
b1sq = bnarr[0,1]**2

print('bsq',bnarr[0]**2)

bprod = np.prod((bnarr[0,:]**2))
print('bprod',bprod)
print('hankel det',np.linalg.det(hankel))




#plt.ylim([0,vals[-1]*0.6])
plt.legend()
plt.show()
exit()

if(0):
        if (l==1): #Computing b1
            Marr[l,j] = Marr[l,j-1]/bnarr[0]**2 # m^{-1} is set to zero
            
        else:
            for j in range(1,l+1):
                bjm1 = bnarr[j-1]
                if(j==1):
                    bjm2 = 1.0
                else:
                    bjm2 = bnarr[j-2]
                print('j',j,'bj-1, bj-2',j-1,j-2,bjm1,bjm2,'mm1,mm2',mm1,mm2)
                mcurr = mm1/bjm1**2 - mm2/bjm2**2
                mm2 = mm1
                mm1 = mcurr
                print('mcurr',mcurr)
        #print('l',l,'mcurr',mcurr)
        bnarr[l] = np.sqrt(mcurr)


