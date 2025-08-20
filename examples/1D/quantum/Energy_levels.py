import numpy as np
from PISC.dvr.dvr import DVR1D
from matplotlib import pyplot as plt
from PISC.potentials import mildly_anharmonic
from PISC.engine import Krylov_complexity
L=20


lb = -L
ub = L
ngrid = 1000
m = 1.0

n_anharm = 4

pes = mildly_anharmonic(m,0.0,1.0,w=0.0,n=n_anharm)


DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 100
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]
pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

O = np.log(np.abs(pos_mat))

if(0):
    diag = np.diag(O,k=1)
    narr = np.arange(1,diag.size+1)
    log_narr = np.log(narr)

    #fit a line to the log-log plot
    fit = np.polyfit(log_narr[10:], diag[10:], 1)
    print('Slope:', fit[0])


    plt.plot(log_narr, diag, 'o', label='DVR')
    plt.plot(log_narr, fit[0]*log_narr + fit[1], label='Fit')
    plt.xlabel('log(n)')
    plt.ylabel('log(O)')
    plt.title('log(O) vs log(n)')
    plt.legend()
    plt.show()
    exit()

fig, ax = plt.subplots(2,1)
for i in range(0,neigs,40):
    ax[0].plot(O[i,i+1::2], label='n = %d' % i)

    j=i+1
    while j < neigs:
        ax[1].scatter(i,j)
        j += 2
plt.show()

exit()
narr = np.arange(1,neigs+1)
log_narr = np.log(narr)[20:]
log_vals = np.log(vals)[20:]

#Fit a line to the log-log plot
coeffs = np.polyfit(log_narr, log_vals, 1)
print('Slope:', coeffs[0])
plt.plot(np.log(narr), np.log(vals), 'o', label='DVR')
plt.plot(log_narr, coeffs[0]*log_narr + coeffs[1], label='Fit')
plt.xlabel('n')
plt.ylabel('E_n')
plt.title('Energy levels of a mildly anharmonic oscillator')
plt.show()
