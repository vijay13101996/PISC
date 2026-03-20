import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt

ngrid = 1000

L=10 
lb=0
ub=L

lbc=0
ubc=L

m=0.5

print('L',L)

potkey = '1D_Box_m_{}_L_{}'.format(m,np.around(L,2))

anal = True
#anal = False

def potential(x):
    if(x<lbc or x>ubc):
        return 1e12
    else:
        #print('x',x)
        return 0.0#x**4

neigs = 1000
potential = np.vectorize(potential)

#xgrid = np.linspace(lb,ub,ngrid)
#plt.plot(xgrid,potential(xgrid))
#plt.ylim([0,1000])
#plt.show()

#----------------------------------------------------------------------

DVR = DVR1D(ngrid, lb, ub,m, potential)
if(not anal):
    vals,vecs = DVR.Diagonalize(neig_total=neigs)


x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

#----------------------------------------------------------------------

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:]/L)

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

#----------------------------------------------------------------------

print('vals',vals_anal[-1])

if(not anal):
    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

if(anal):
    print('Using analytical pos_mat, vals')
    vals = vals_anal
    O = O_anal
else:
    print('Using numerical pos_mat, vals')
    vals = vals
    O = pos_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

LL0 = np.zeros((neigs,neigs))
LL0 = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,LO,LL0)

diagO = np.diag(abs(O),1)
LnO = np.zeros((neigs,neigs))
LnO = O
n = 50
eigarr = np.arange(1,neigs+1)

maxarr = []

for i in range(n):
    LnO = L*LnO
    fl_O = np.fliplr(abs(LnO))
    diag =  np.diag(fl_O)
    plt.scatter(eigarr, diag,s=10)
    plt.plot(eigarr, diag,label='n={}'.format(i))
    
    maxind = np.argmax(diag)
    print('maxind',eigarr[maxind])
    plt.scatter(eigarr[maxind], diag[maxind],s=20,color='k')
    
    maxarr.append(eigarr[maxind])

plt.yscale('log')
plt.show()

plt.plot((maxarr))
plt.xscale('log')
plt.yscale('log')
plt.show()

exit()


fl_O = np.fliplr(abs(O))
plt.plot(np.diag(fl_O),'o')
plt.show()
exit()

diagO1 = np.diag(abs(O),1)
diagLO1 = np.diag(abs(LO),1)
diagLL01 = np.diag(abs(LL0),1)

#plt.plot(diagO1, label='O1')
#plt.plot(diagLO1, label='LO1')
plt.plot(diagO1/diagLO1,'o', label='O1/LO1')
plt.plot(diagLO1/diagLL01)
plt.legend()
plt.show()




