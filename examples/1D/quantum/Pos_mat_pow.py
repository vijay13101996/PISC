import numpy as np
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well

ngrid = 1000

L = 20
lb = -L
ub = L
m = 1.0

w=1.0
pes = harmonic1D(m,w)
potkey = 'harmonic_w_{}'.format(w)

T_au = 1.
beta = 1.0/T_au 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 500
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
O = (pos_mat)

def comp_bn(O, ncoeffs, beta, vals):
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)
    #print('liou_mat',liou_mat[0:5,0:5],vals[0:5])

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeffs)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5,'wgm')

    return barr


O2 = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        if(abs(i-j)==1):
            O2[i,j] = 1.0

Oexp = np.zeros((neigs,neigs))
gamma = 1.
for i in range(neigs):
    for j in range(neigs):
            Oexp[i,j] = np.exp(-gamma*abs(i-j)**0.5)

ncoeffs = 40
narr = np.arange(ncoeffs)
plt.plot(narr[1:], np.pi*T_au*narr[1:], marker='x', linestyle='--', color='r', label='Expected')

barr = comp_bn(Oexp, ncoeffs, beta, vals)
#fit barr and narr to a line
fit = np.polyfit(narr[1:], barr[1:], 1)
print('alpha', fit[0], np.pi/(beta+4*gamma))
plt.plot(narr[1:], fit[0]*narr[1:] + fit[1], linestyle='--', color='g', label='Fit: {:.2f}n + {:.2f}'.format(fit[0], fit[1]))
plt.scatter(narr[1:], barr[1:], s=3, label='Exponential')

if(0):
    for k in [100]:#,100,200,300]:
        Ok = O2.copy()
        for n in range(k):
            Ok = np.dot(Ok, O2)

        #Set diagonal elements to zero
        for i in range(neigs):
            Ok[i,i] = 0.0

        print('Ok', Ok)

        #Compute Lanczos coefficients
        O = Ok.copy()

        barr = comp_bn(O, ncoeffs, beta, vals)

        plt.plot(narr[1:], barr[1:], marker='o', linestyle='-')
plt.title('Lanczos Coefficients')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.grid()
plt.show()

exit()
Oklog = np.log(np.abs(Ok) + 1e-10)  # Avoid log(0)

plt.imshow(Oklog, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Position Matrix Power (k={})'.format(k))
plt.xlabel('State Index')
plt.ylabel('State Index')
#plt.xticks(np.arange(0, neigs, 5))
#plt.yticks(np.arange(0, neigs, 5))
plt.show()

