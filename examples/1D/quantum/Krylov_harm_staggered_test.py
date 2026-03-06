import numpy as np
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D

def K_complexity(O,vals,beta,ncoeff=50):
    # Compute the Krylov complexity and moments

    neigs = O.shape[0]
    vals = vals[:neigs]  # Ensure vals is the same length as O

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    
    return barr

lb = -20.0
ub = 20.0
m = 1.0
ngrid = 1000
neigs = 100

def V_hq(x):
    if(x<0):
        return x**2
    else:
        return x**4
if(0):
    V_hq_vec = np.vectorize(V_hq)

    DVR = DVR1D(ngrid,lb,ub,m,V_hq_vec)
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

vals = np.arange(neigs) + 0.5
O1= np.zeros((neigs, neigs))  # Example operator (all ones)
O2 = np.zeros((neigs, neigs))  # Example operator (all ones)
O3 = np.zeros((neigs, neigs))  # Example operator (all ones)
for i in range(neigs):
    for j in range(neigs):
        O1[i,j] = 1.0
        if (abs(i-j)%2)==0:
            O2[i,j] = 1.0
        elif(abs(i-j)%2)==1:
            O2[i,j] = 2.0
        if (abs(i-j)%3)==0:
            O3[i,j] = 1.0
    if(i==j):
        O1[i,j] = 0.0
        O2[i,j] = 0.0
        O3[i,j] = 0.0

beta = 3
ncoeff = 50
b1arr = K_complexity(O1, vals, beta, ncoeff)
b2arr = K_complexity(O2, vals, beta, ncoeff)
b3arr = K_complexity(O3, vals, beta, ncoeff)

narr = np.arange(ncoeff)
plt.scatter(narr, b1arr, color='blue', label='Krylov Coefficients')
#plt.scatter(narr, b2arr, color='orange', label='Krylov Coefficients (Staggered 2)')
#plt.scatter(narr, b3arr, color='green', label='Krylov Coefficients (Staggered 3)')
plt.xlabel('n')
plt.ylabel('b_n')
plt.title('Krylov Coefficients vs n')
plt.legend()
plt.grid()
plt.show()

