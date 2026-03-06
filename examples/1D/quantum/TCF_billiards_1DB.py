import numpy as np
from PISC.engine import Krylov_complexity_2D
from PISC.dvr.dvr import DVR2D
from PISC.potentials import coupled_harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
import time
import os
import matplotlib
from matplotlib.patches import Rectangle
from Krylov_WF_tools import fix_vecs, comp_Ot, Cn, av_O, avg_O, av_O_therm, find_coeff_wf, verify_Ct

#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

R = 0.37  #np.sqrt(1/(4+np.pi))
a = R  #0.0

neigs = 500 #Number of eigenstates to be calculated

bill = 1
diag = 1

path = os.path.dirname(os.path.abspath(__file__))

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
print('R',R)

if(0): #Plot the potential
    xg = np.linspace(lbx,ubx,ngridx)
    yg = np.linspace(lby,uby,ngridy)
    xgr,ygr = np.meshgrid(xg,yg)
    plt.contour(xgr,ygr,potential_xy(xgr,ygr),levels=np.arange(-1,30,1.0))
    #plt.imshow(potential_xy(xgr,ygr),origin='lower')
    plt.show()    
    #exit()

def TCF_O(vals, beta, neigs, O, t_arr):
    
    n_arr = np.arange(neigs)
    m_arr = np.arange(neigs)

    C_arr = np.zeros_like(t_arr) + 0j

    for n in n_arr:
        for m in m_arr:
            C_arr += np.exp(-beta*(vals[n]+vals[m])/2) * np.exp(1j*(vals[n]-vals[m])*t_arr) * np.abs(O[n,m])**2

    Z = np.sum(np.exp(-beta*vals))
    C_arr /= Z

    return C_arr

def Krylov_O(vals, beta, neigs, O, ncoeff):
    L = np.zeros((neigs,neigs))
    L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,L)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity_2D.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff) 
    barr = Krylov_complexity_2D.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 0.5, 'wgm')
    return barr

def compute_O2_avg(O, vals, T, neigs):
    beta = 1.0/T
    Z = np.sum(np.exp(-beta*vals))
    O2_avg = 0.0
    for n in range(neigs):
        for m in range(neigs):
            O2_avg += np.exp(-0.5*beta*(vals[n]+vals[m])) * np.abs(O[n,m])**2
    O2_avg /= Z
    return O2_avg

def potential_stadium(x,y):
    if( (x+a)**2 + y**2 < R**2 or (x-a)**2 + y**2 < R**2):
        return 0.0
    elif(x>-a and x<a and y>-R and y<R):
        return 0.0 
    else:
        return 1e6
potential_stadium = np.vectorize(potential_stadium)

def potential_box(x,y):
    if (x>(a+R) or x<(-a-R) or y>R or y<-R):
        return 1e6
    else:
        return 0.0
potential_box = np.vectorize(potential_box)

def bookkeeping(pot):
    if pot=='box':
        potential_xy = potential_box
        potkey = 'MANU_rectangular_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))
    elif pot=='stadium':
        potential_xy = potential_stadium
        potkey = 'MANU_stadium_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

    x = np.linspace(lbx,ubx,ngridx+1)
    #print('Vs',pes.potential_xy(0,0))
    fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  

    DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential_xy)
    print('potential',potkey)   

    #Diagonalization
    param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':neigs}
    with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
        f.write('\n'+str(param_dict))

    if(diag): #Diagonalize the Hamiltonian
        vals,vecs = DVR.Diagonalize(neig_total=neigs)

        store_arr(vecs[:,:neigs],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        store_arr(vals[:neigs],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
        print('Time taken:',time.time()-start_time)
        
        #plt.plot(vals[:neigs])
        #plt.show()

        #exit()

    #Read eigenvalues and eigenvectors and test whether the Wavefunctions look correct
    vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

    n=99
    print('vals[n]', vals[:10])

    #plt.imshow(DVR.eigenstate(vecs[:,0])**2,origin='lower')
    #plt.show()

    return DVR, vals, vecs, potkey

tarr = np.linspace(-300,300,10000)
T = 0.25
beta = 1.0/T

for pot in ['box','stadium']:
    DVR, vals, vecs, potkey = bookkeeping(pot)
    try:
        # Read data if it exists
        fname = 'Spectra_TCF_{}_T_{}'.format(potkey, np.around(T,2))
        data = np.loadtxt('{}/Datafiles/{}'.format(path,fname), skiprows=1)
        
        fname = 'TCF_{}_T_{}'.format(potkey, np.around(T,2))
        data_tcf = np.loadtxt('{}/Datafiles/{}'.format(path,fname), skiprows=1)

        fname = 'Lanczos_coeffs_{}_T_{}'.format(potkey, np.around(T,2))
        data_lanczos = np.loadtxt('{}/Datafiles/{}'.format(path,fname), skiprows=1)

        print('Data for {} already exists, skipping computation.'.format(pot))

        if(1): #Plot Lanczos coeffs
            plt.scatter(data_lanczos[:,0], data_lanczos[:,1], label='Lanczos coeffs {}'.format(pot))
            plt.xlabel('n', fontsize=xl_fs)
            plt.ylabel('Lanczos b_n', fontsize=yl_fs)
            plt.legend(fontsize=le_fs)
   
        if(0): #Plot TCF FT
            plt.plot(data[:,0], data[:,1], label='TCF FT {}'.format(pot))
            plt.xlabel('Frequency ω', fontsize=xl_fs)
            plt.ylabel('Log TCF FT log|C(ω)|', fontsize=yl_fs)
            for d in np.diff(np.sort(vals)):
                plt.axvline(x=d, color='r', linestyle='--', alpha=0.5)
            plt.legend(fontsize=le_fs)
    
        if(0): #Plot TCF
            plt.plot(data_tcf[:,0], np.log(abs(data_tcf[:,1])), label='Re C(t) {}'.format(pot))
            plt.xlabel('Time t', fontsize=xl_fs)
            plt.ylabel('TCF C(t)', fontsize=yl_fs)
            plt.legend(fontsize=le_fs)

    except IOError:
        print('Computing data for {}...'.format(pot))

        x_arr = DVR.pos_mat(0)
        pos_mat = np.zeros((neigs,neigs)) 
        pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
        O = pos_mat

        O2_avg = compute_O2_avg(O, vals, T=10.0, neigs=neigs)
        print('<O^2>_T=', O2_avg)
        diffs = np.diff(np.sort(vals))

        C_tcf = TCF_O(vals, beta, neigs, O, tarr)

        freqs = np.fft.fftfreq(len(tarr), d=(tarr[1]-tarr[0]))*2*np.pi
        C_tcf_ft = np.fft.fft(C_tcf)
        #Plot real part of C_tcf
        C_tcf_ft = np.fft.fftshift(C_tcf_ft)
        freqs = np.fft.fftshift(freqs)

        #Save freqs and C_tcf_ft in column format
        fname = 'Spectra_TCF_{}_T_{}'.format(potkey, np.around(T,2))
        np.savetxt('{}/Datafiles/{}'.format(path,fname), np.column_stack((freqs, np.log(np.abs(C_tcf_ft)))), header='Frequency Log_TCF_FT', comments='')

        fname = 'TCF_{}_T_{}'.format(potkey, np.around(T,2))
        np.savetxt('{}/Datafiles/{}'.format(path,fname), np.column_stack((tarr, C_tcf.real, C_tcf.imag)), header='Time Re_C_t Im_C_t', comments='')

        coeffs = 200
        barr = Krylov_O(vals, beta, neigs, O, coeffs)
        coeff_arr = np.arange(coeffs)

        fname = 'Lanczos_coeffs_{}_T_{}'.format(potkey, np.around(T,2))
        np.savetxt('{}/Datafiles/{}'.format(path,fname), np.column_stack((coeff_arr, barr)), header='n b_n', comments='')
        
        verify_Ct(vals, beta, neigs, O, barr, tarr, C_tcf)

plt.show()

if(0): #Plot TCF
    plt.plot(tarr, C_tcf.real, label='Re C(t)')
    plt.plot(tarr, C_tcf.imag, label='Im C(t)')
    plt.xlabel('Time t', fontsize=xl_fs)
    plt.ylabel('TCF C(t)', fontsize=yl_fs)
    plt.show()

if(0): #Plot TCF FT
    plt.plot(freqs, np.log(np.abs(C_tcf_ft)), label='|C(ω)|')
    plt.xlabel('Frequency ω', fontsize=xl_fs)
    plt.ylabel('Log TCF FT log|C(ω)|', fontsize=yl_fs)
    
    for d in diffs:
        plt.axvline(x=d, color='r', linestyle='--', alpha=0.5)

    plt.show()

if(0):
    #Compare lanczos coeffs of rectangular and stadium billiards
    fname_bil = 'Lanczos_coeffs_MANU_lorenz_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))
    fname_box = 'Lanczos_coeffs_MANU_square_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))
    data_bil = np.loadtxt('{}/Datafiles/{}'.format(path,fname_bil), skiprows=1)
    data_box = np.loadtxt('{}/Datafiles/{}'.format(path,fname_box), skiprows=1)
    n_bil = data_bil[:,0]
    b_bil = data_bil[:,1]
    n_box = data_box[:,0]
    b_box = data_box[:,1]   

    plt.scatter(n_bil, b_bil, label='Lorenz billiards')
    plt.scatter(n_box, b_box, label='Square box', ls='--')
    plt.xlabel('n', fontsize=xl_fs)
    plt.ylabel('Lanczos b_n', fontsize=yl_fs)
    plt.legend(fontsize=le_fs)
    plt.show()

if(0):
    #Compare rectangular and stadium billiards spectra
    fname_rect = 'Spectra_TCF_MANU_square_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))
    fname_stad = 'Spectra_TCF_MANU_lorenz_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))
    data_rect = np.loadtxt('{}/Datafiles/{}'.format(path,fname_rect), skiprows=1)
    data_stad = np.loadtxt('{}/Datafiles/{}'.format(path,fname_stad), skiprows=1)
    freqs_rect = data_rect[:,0]
    Ctcf_ft_rect = data_rect[:,1]
    freqs_stad = data_stad[:,0]
    Ctcf_ft_stad = data_stad[:,1]

    plt.plot(freqs_rect, Ctcf_ft_rect, label='Square box')
    plt.plot(freqs_stad, Ctcf_ft_stad, label='Lorenz billiards')
    for d in diffs:
        plt.axvline(x=d, color='r', linestyle='--', alpha=0.5)

    plt.xlabel('Frequency ω', fontsize=xl_fs)
    plt.ylabel('Log TCF FT log|C(ω)|', fontsize=yl_fs)
    plt.legend(fontsize=le_fs)
    plt.show()


