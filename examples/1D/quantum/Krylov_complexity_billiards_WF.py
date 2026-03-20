import numpy as np
from PISC.engine import Krylov_complexity_2D as Krylov_complexity
from PISC.dvr.dvr import DVR2D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from Krylov_WF_tools import coh_st_coeff_2D, wf_t, comp_Ot, Cn, corr_func, av_O, fix_vecs, compute_Wn, verify_Ct
import time
import matplotlib
import os

#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

neigs = 500 #Number of eigenstates to be calculated

R = np.sqrt(1/(4+np.pi))
a = R #0.0

if(1): #Stadium billiards
    def potential_xy(x,y):
        if( (x+a)**2 + y**2 < R**2 or (x-a)**2 + y**2 < R**2):
            return 0.0
        elif(x>-a and x<a and y>-R and y<R):
            return 0.0 
        else:
            return 1e6
        
    potential_xy = np.vectorize(potential_xy)   
    potkey = 'MANU_stadium_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

if(1): #Rectangular box
    def potential_xy(x,y):
        if (x>(a+R) or x<(-a-R) or y>R or y<-R):
            return 1e6
        else:
            return 0.0

    potential_xy = np.vectorize(potential_xy)

    potkey = 'MANU_rectangular_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

#System parameters
m = 0.5
L = (a+R)*1.5
lbx = -L
ubx = L
lby = -2*R
uby = 2*R
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

start_time = time.time()
print('R',R, 4*R)

if(0): #Plot the potential
    xg = np.linspace(lbx,ubx,ngridx)
    yg = np.linspace(lby,uby,ngridy)
    xgr,ygr = np.meshgrid(xg,yg)
    plt.contour(xgr,ygr,potential_xy(xgr,ygr),levels=np.arange(-1,30,1.0))
    #plt.imshow(potential_xy(xgr,ygr),origin='lower')
    plt.show()    
    #exit()

x = np.linspace(lbx,ubx,ngridx+1)
y = np.linspace(lby,uby,ngridy+1)


fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential_xy)
print('potential',potkey)   

#Diagonalization
param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':neigs}
with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
    f.write('\n'+str(param_dict))

if(0): #Diagonalize the Hamiltonian
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    store_arr(vecs[:,:neigs],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    store_arr(vals[:neigs],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
    print('Time taken:',time.time()-start_time)
    
    #plt.plot(vals[:neigs])
    #plt.show()

    #exit()

if(1): #Read eigenvalues and eigenvectors and test whether the Wavefunctions look correct
    vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    
    n=99
    print('vals[n]', vals[-1])
    
    #plt.imshow(DVR.eigenstate(vecs[:,-1])**2,origin='lower')
    #plt.show()

vecs = fix_vecs(vecs, tol=1e-2)

x_arr = DVR.pos_mat(0)
y_arr = DVR.pos_mat(1)
#-------------------------------------------------------
pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
O = pos_mat

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

if(0):  # TEST!!
    tarr = np.arange(0.0,10.01,0.1)
    Ot = np.zeros((neigs,neigs), dtype=complex)

    Ct_arr = np.zeros(len(tarr), dtype=complex)
    Ct_arr_exact = np.zeros(len(tarr), dtype=complex)

    for i in range(len(tarr)):
        t = tarr[i]
        Ot = comp_Ot(O, L, t)
        ip = 0.0
        ip = Krylov_complexity.krylov_complexity.compute_ip_wf(Ot, O, coeff_wf, ip)
        print('t={}, ip={}'.format(t, ip))
        Ct_arr_exact[i] = ip

        Cn_t = Cn(t, vals, O, n_wf)
        print('t={}, Cn_t={}'.format(t, Cn_t))
        Ct_arr[i] = Cn_t

    plt.plot(tarr, Ct_arr.real, label='Cn_t (sum over states)')
    plt.plot(tarr, Ct_arr_exact.real, '--', label='Ct_arr_exact (Krylov)')

    plt.plot(tarr, Ct_arr.imag, label='Cn_t imag (sum over states)')
    plt.plot(tarr, Ct_arr_exact.imag, '--', label='Ct_arr_exact imag (Krylov)')
    plt.legend()

    plt.xlabel('Time')
    plt.ylabel('Correlation Function')
    plt.title('Comparison of Correlation Functions')
    plt.show()

    exit()

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

ncoeff = 100
barr = np.zeros(ncoeff)

x0 = 0.0
p0 = 0.0
sigma_x = 1.0 #0.25

coeff_wf, wf = coh_st_coeff_2D(x0, 0.0, p0, 0.0, sigma_x, sigma_x, x, y, vecs, neigs, DVR)

if(1):
    #coeff_wf[:] = 0.0
    #coeff_wf[:10] = 1.0/np.sqrt(10)

    T = 10.0
    beta = 1.0/T
    #Compute Lanczos coefficients
    ip = 0.0
    ip = Krylov_complexity.krylov_complexity.compute_ip(O, O, beta, vals, 0.5, ip, 'wgm')
    print('ip(O,O)', ip)
    O/=np.sqrt(ip)
    ip = Krylov_complexity.krylov_complexity.compute_ip(O, O, beta, vals, 0.5, 0.0, 'wgm')
    print('ip(O,O) after normalization', ip)

    barr[:] = 0.0
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5,'wgm')
    plt.plot(np.arange(ncoeff), barr, 'o-')
    plt.xlabel('n')
    plt.ylabel('$b_n$')
    plt.title('Lanczos Coefficients for {}'.format(potkey))
    plt.show()
    verify_Ct(O, vals, barr, beta, neigs)
    exit() 


    Wn = compute_Wn(L, coeff_wf, vals, barr)
    plt.plot(np.arange(len(Wn)), np.log(abs(Wn)), 'o-')
    plt.xlabel('n')
    plt.ylabel('log|Wn|')
    plt.title('Log of Wn for Lanczos Coefficients for {}'.format(potkey))
    plt.show()

    exit()

tarr = np.arange(0.0,10.01,0.1)

if(0):
    wf_tarr, avg_x2 = wf_t(wf, vecs, vals, tarr, N=10)

    for i in range(5):
        plt.plot(tarr, avg_x2[2*i+1].real, label='⟨x^{}⟩'.format(i+1))
    plt.xlabel('Time')
    plt.ylabel('Expectation Values')
    plt.title('Time Evolution of Position Moments')
    plt.legend()
    plt.show()

    exit()

if(0):
    plt.xlabel('x')
    plt.ylabel('|ψ(x,t)|²')
    plt.xlim([-10,10])
    for i in range(0, len(tarr), 20):
        plt.plot(x_arr, np.abs(wf_tarr[:,i])**2, label='t={}'.format(np.around(tarr[i],2)))
        #plt.scatter(avg_x2[i], 0.0, color='red', marker='x', s=100, label='⟨x²⟩ at t={}'.format(np.around(tarr[i],2)))
        plt.pause(0.25)
        plt.clf()
    plt.title('Time Evolution of Coherent State Wavefunction')
    plt.legend()
    plt.show()

    exit()

p0_arr = np.arange(-4.0,4.01,2.)
sigma_arr = np.arange(0.1,2.01,0.5)
x0_arr = np.arange(1,9.01,1.0)

slope_arr = []

change_var = 'sigma_x'  # 'p0' or 'sigma_x' or 'x0'

if(change_var=='p0'):
    var_arr = p0_arr
elif(change_var=='sigma_x'):
    var_arr = sigma_arr
elif(change_var=='x0'):
    var_arr = x0_arr

for var in var_arr:
    if(change_var=='p0'):
        p0 = var
    elif(change_var=='sigma_x'):
        sigma_x = var
    elif(change_var=='x0'):
        x0 = var

    print('Computing for sigma_x={}, p0={}, x0={}'.format(sigma_x, p0, x0))

    coeff_wf, wf = coh_st_coeff_2D(x0, 0.0, p0, 0.0, sigma_x, sigma_x, x, y, vecs, neigs, DVR)
    #plt.plot(coeff_wf[1:].real)

    barr[:] = 0.0
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs_wf(O, L, barr, coeff_wf)
    
    if(change_var=='p0'):
        plt.scatter(np.arange(ncoeff),barr,label='$p_0$={}'.format(p0))
    elif(change_var=='sigma_x'):
        plt.scatter(np.arange(ncoeff),barr,label='$\sigma_x$={}'.format(np.around(sigma_x,2)))
    elif(change_var=='x0'):
        plt.scatter(np.arange(ncoeff),barr,label='$x_0$={}'.format(x0))

    #FIT Lanczos coeffs to a line
    p = np.polyfit(np.arange(ncoeff), barr, 1)
    plt.plot(np.arange(ncoeff), p[0]*np.arange(ncoeff)+p[1], '--')
    print('slope for sigma_x, p0, x0={},{},{}: {}'.format(sigma_x, p0, x0, p[0]))
    slope_arr.append(p[0])

    #av_x = av_O(O, tarr, coeff_wf, vals)
    #plt.plot(tarr, av_x.real, label='$\sigma_x$={}'.format(sigma_x))
    
    #Ct_arr = corr_func(tarr, vals, O, L, coeff_wf)
    #plt.plot(tarr, Ct_arr.real, label='$p_0$={}'.format(p0))
    #plt.plot(tarr, Ct_arr.real, label='$\sigma_x$={}'.format(sigma_x))

plt.xlabel(r'$n$')
plt.ylabel('$b_n$')
plt.title('Lanczos Coefficients for {}'.format(potkey))
plt.legend()
plt.show()

if(change_var=='p0'):
    plt.plot(p0_arr, slope_arr, 'o-')
    plt.xlabel('$p_0$ of Coherent State')
    plt.title('Slope vs Initial Momentum of Coherent State')
elif(change_var=='x0'):
    plt.plot(x0_arr, slope_arr, 'o-')
    plt.xlabel('$x_0$ of Coherent State')
    plt.title('Slope vs Initial Position of Coherent State')
elif(change_var=='sigma_x'):
    plt.plot(sigma_arr, slope_arr, 'o-')
    plt.xlabel('$\sigma_x$ of Coherent State')
    plt.title('Slope vs Width of Coherent State')


plt.ylabel('Slope of Lanczos Coefficients')
plt.grid()
plt.show()


