import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity

if(1): # 1D Box
    m = 1.0
    L = 2.0
    ngrid = 2**9 
    neigs = ngrid-1

    lb = 0
    ub = L

    def pot(x):
        if(x>L or x<0):
            return 1e10
        else:
            return 0.0

    potkey = 'L_{}_1D_box_0'.format(L)
    DVR = DVR1D(ngrid, lb, ub,m, pot)
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    x_arr = DVR.grid[1:ngrid]
    dx = x_arr[1]-x_arr[0]
    
    def pos_mat_anal(i,j,neigs):
        if(i==j):
            return L/2
        else:
            return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

    vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
    vecs_anal = np.zeros((neigs,ngrid-1))
    for i in range(neigs):
        vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:ngrid]/L)

    vals_box = vals_anal
    vecs_box = vecs_anal

    O_anal = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(neigs):
            O_anal[i,j] = pos_mat_anal(i,j,neigs)
    
    pos_mat_box = O_anal - L/2*np.diag(np.ones(neigs))

if(1): # Quartic oscillator
    m = 1.0
    L = 10.0
    n_anharm = 4
    omega = 0.0
    a = 0.0
    b = 1.0
    ngrid = 2**9 

    lb = -L
    ub = L
    pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

    potkey = 'L_{}_quartic'.format(L)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 200 # :ngrid-1
    vals_quart,vecs_quart = DVR.Diagonalize(neig_total=neigs)

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]
    
    pos_mat_quart = np.zeros((neigs,neigs))
    pos_mat_quart = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs_quart, x_arr, dx, dx, pos_mat_quart)

pot = 'box' # 'box' or 'quartic'
if(pot=='box'):
    vals = vals_box
    O = pos_mat_box
    potkey = '1D_box'
elif(pot=='quartic'):
    vals = vals_quart
    O = pos_mat_quart
    potkey = 'quartic'

w_F_arr = np.arange(0.5, 30.01, 0.25)

"""
For each frequency w_F, we count how many pairs of energy levels (E_i, E_j) are within w_F, 2w_F... of each other upto epsilon
"""

epsilon = 0.01
nph_arr= np.arange(1,50)

if(0):
    for w_F in w_F_arr:
        count_arr = np.zeros(len(nph_arr))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                delta_E = abs(vals[i]-vals[j])
                for nph in nph_arr:
                    if abs(delta_E - nph*w_F) < epsilon:
                        count_arr[nph-1] += 1
        plt.plot(nph_arr, count_arr, label='w_F={:.2f}'.format(w_F))
        plt.show()

if(0):
    for w_F in w_F_arr:
        count_arr = np.zeros(len(nph_arr))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                delta_E = abs(vals[i]-vals[j])
                for nph in nph_arr:
                    if abs(delta_E - nph*w_F) < epsilon:
                        count_arr[nph-1] += 1
        plt.scatter(w_F, max(count_arr), label='w_F={:.2f}'.format(w_F), color='blue')

    plt.xlabel('Driving Frequency w_F')
    plt.ylabel('Max Count of Level Pairs')
    plt.title('{}'.format(potkey))
    plt.show()

print('O \n', O[:10,:10])

if(0):
    absorp_arr = np.zeros(len(w_F_arr))
    for w_F in w_F_arr:
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                delta_E = abs(vals[i]-vals[j])
                for nph in nph_arr:
                    if abs(delta_E - nph*w_F) < epsilon:
                        absorp_arr[w_F_arr==w_F] += abs(O[i,j])**2*nph

    plt.plot(w_F_arr, absorp_arr)
    plt.xlabel('Driving Frequency w_F')
    plt.ylabel('Absorption Strength')
    plt.title('{}'.format(potkey))
    plt.show()

if(1):
    """
    At each w_F, see what nph is possible for each pair of levels.
    """
    for w_F in w_F_arr:
        nph_possible = np.zeros(len(nph_arr))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                delta_E = abs(vals[i]-vals[j])
                for nph in nph_arr:
                    if abs(delta_E - nph*w_F) < epsilon:
                        nph_possible[nph-1] += 1

        # Plot all nph_possible for this w_F
        plt.scatter(w_F*np.ones(len(nph_arr)), nph_possible, label='w_F={:.2f}'.format(w_F), color='blue')

    plt.xlabel('Driving Frequency w_F')
    plt.ylabel('Count of Level Pairs for each nph')
    plt.title('{}'.format(potkey))
    plt.show()



