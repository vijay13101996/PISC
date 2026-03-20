import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt

ngrid = 1000

L = 8
lb = -L
ub = L

n_anharm=6

neigs = 350

k=10
nvals = np.arange(1,neigs+1)[k+1::2]
def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

O_anal = np.zeros((neigs,neigs))
O_dum = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)
        if(i!=j):
            O_dum[i,j] = L*(1/(i+j)**3)#*(1-(-1)**(i+j+2))/np.pi**2

Omat_1db = abs(O_anal[k,k+1::2])
Omat_dum = abs(O_dum[k,k+1::2])
print('nvals',len(nvals),len(Omat_1db),len(Omat_dum))
#plt.plot((nvals[:20]), np.log(Omat_1db[:20]), 'o', label='analytical')
#plt.plot(np.log(nvals[:200]), np.log(Omat_dum[:200]), 'o', label='dum')
#fit log nvals to log Omat

#coeff = np.polyfit(np.log(nvals[20:200]), np.log(Omat_1db[20:200]), 1)
#print('coeff',coeff)
#plt.plot(np.log(nvals[20:200]), coeff[0]*np.log(nvals[20:200])+coeff[1], label='fit')


#plt.show()
#exit()

n_anharm_lst = [4,6,8,10,12,14]
L_lst = [10,8,6,4,2,2]

n_anharm_lst = [12,14,16,18]
L_lst = [2,2,2,2]

n_anharm_lst = [6,12,18]
L_lst = [8,2,2]

for n_anharm, L in zip(n_anharm_lst, L_lst):
    lb = -L
    ub = L

    m=1.0
    a=0.0#4
    b=L/2#16
    omega=0.0

    pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)
    
    potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,n_anharm)
    neigs = 350

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    x_arr = DVR.grid[1:ngrid]
    plt.plot(x_arr,pes.potential_func(x_arr), label='{}:{}'.format(n_anharm,L))

    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
    O = (pos_mat)

    Omat = abs(pos_mat[k,k+1::2])

    plt.plot((nvals[:20]), np.log(Omat[:20]), 'o', label='p={}'.format(n_anharm))
    plt.plot((nvals[:20]), np.log(Omat[:20]), label='p={}'.format(n_anharm))


plt.title('Position matrix')
plt.xlabel('j')
plt.ylabel('i')
plt.legend()
plt.show()
exit()

#Plot 1D box
pot = np.zeros_like(x_arr)
for i in range(len(x_arr)):
    if(x_arr[i]<-L/2 or x_arr[i]>L/2):
        pot[i] = 1e6
    else:
        pot[i] = 0
plt.plot(x_arr, pot, label='1D box')

plt.show()

exit()

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

diagO = np.diag(abs(O),1)
LnO = np.zeros((neigs,neigs))
LnO = O
n = 12
eigarr = np.arange(1,neigs+1)
for i in range(n):
    #diag1 = np.diag(abs(LnO),1)
    #diag2 = np.diag(abs(L*LnO),1)
    #diag4 = np.diag(abs(LnO),3)
    LnO = L*LnO
    fl_O = np.fliplr(abs(LnO))
    plt.scatter(eigarr, np.diag(fl_O),s=10)
    plt.plot(eigarr, np.diag(fl_O),label='n={}'.format(i))
    #plt.plot((diag1/diag2))#,alpha=0.5)
    #plt.plot(diag4)

#plt.xscale('log')
#plt.yscale('log')
#plt.show()

#Plot the anti-diagonal of the position matrix
#plt.imshow(O,origin='lower')
#plt.colorbar()

#plt.plot(np.diag(fl_O),'o')
plt.show()

exit()

#LO = np.zeros((neigs,neigs))
#LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

#LLO = np.zeros((neigs,neigs))
#LLO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,LO,LLO)

diagO1 = np.diag(abs(O),1)
diagLO1 = np.diag(abs(LO),1)
diagLLO1 = np.diag(abs(LLO),1)

#plt.plot(diagO1, label='O1')
#plt.plot(diagLO1, label='LO1')
plt.plot(diagO1/diagLO1,'o', label='O1/LO1')
plt.plot(diagLO1/diagLLO1)
plt.legend()
plt.show()

