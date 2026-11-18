import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.engine import Krylov_complexity

# Box from 0 to L
m = 1.0
L = 2.0
ngrid = 2**9 
neigs = ngrid-1

lb0 = 0
ub0 = L

def pot_0(x):
    if(x>L or x<0):
        return 1e10
    else:
        return 0.0

DVR0 = DVR1D(ngrid, lb0, ub0,m, pot_0)
vals0,vecs0 = DVR0.Diagonalize(neig_total=neigs)

x_arr0 = DVR0.grid[1:ngrid]
dx0 = x_arr0[1]-x_arr0[0]

pos_mat_0 = np.zeros((neigs,neigs)) 
pos_mat_0 = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs0, x_arr0, dx0, dx0, pos_mat_0)

# Box from -L/2 to L/2

def pot(x):
    if(x>L/2 or x<-L/2):
        return 1e10
    else:
        return 0.0

lb = -L/2
ub = L/2

DVR = DVR1D(ngrid, lb, ub,m, pot)
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

pos_mat = np.zeros((neigs,neigs))
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

print('pos_mat_0 \n',np.around(pos_mat_0[0:10,0:10],3),'\n')
print('pos_mat \n',np.around(pos_mat[0:10,0:10],3),'\n')















