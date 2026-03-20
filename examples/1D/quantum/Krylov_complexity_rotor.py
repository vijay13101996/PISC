import numpy as np
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
import scipy
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import pickle

def K_complexity(O,vals,beta,ncoeff=50):
    # Compute the Krylov complexity and moments

    neigs = O.shape[0]

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    
    return barr


neigs = 401
I = 1.0 # Moment of inertia
hbar = 1.0 # Planck's constant
beta = 0.1

# Eigenstates are |n>, n=0,+-1,+-2,... and we arrange in that order
nrange = [0]
for i in range(1,neigs//2+1):
    nrange.append(i)
    nrange.append(-i)

nrange = np.array(nrange)
print(nrange.shape)

vals = (hbar**2/(2.0*I))*nrange**2 # Energy eigenvalues for quantum rotor

theta = np.zeros((neigs,neigs),dtype=complex) # Theta operator in energy basis
for i in range(neigs):
    for j in range(neigs):
        n = nrange[i]
        m = nrange[j]
        print(n,m)
        if (n==m):
            theta[i,j] = np.pi
        else:
            theta[i,j] = 1/(1j*(n-m))


k=1
#for k in range(1,6):
if True:
    cosktheta = np.zeros((neigs,neigs),dtype=complex) # Cos(k*theta) operator in energy basis

    for i in range(neigs):
        for j in range(m,neigs):
            n = nrange[i]
            m = nrange[j]
            if(0):
                if (k+n-m)%2==0 and (k+n-m)>=0 and (k+n-m)<=2*k:
                    #print('|n-m|=',abs(n-m))
                    cosktheta[m,n] = 1/2**k*scipy.special.comb(k,(k+n-m)//2)
                else:
                    cosktheta[m,n] = 0
            if(abs(n-m)==1):
                cosktheta[m,n] = 0.5
            

            cosktheta[n,m] = np.conj(cosktheta[m,n])


    print(np.real(cosktheta)[:,1])
    if k>1:
        cosktheta_old = np.copy(cosktheta)
        for n in range(k-1):
            cosktheta = np.dot(cosktheta,cosktheta_old)

    barr_theta = K_complexity(theta,vals,beta,ncoeff=50)
    barr_cosktheta = K_complexity(cosktheta,vals,beta,ncoeff=50)
    narr = np.arange(0,len(barr_theta))

    #plt.scatter(narr,barr_theta,label=r'$\hat{{O}} = \theta$',s=5)
    plt.scatter(narr,barr_cosktheta,label=r'$\hat{{O}} = \cos^k(\theta), k={}$'.format(k),s=5)
    #plt.plot(narr, np.pi*narr/beta, '--', label=r'$\alpha = {\pi}/{\beta}$', color='r')
plt.xlabel('$n$')
plt.ylabel('$b_n$')
plt.legend()
plt.show()





