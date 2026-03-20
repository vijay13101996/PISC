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


plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
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
a = R#0.0

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

stad = 1
if(stad): #A 'quarter' of a Stadium billiards
    def potential_xy(x,y):
        if(x>0 and y>0):
            if( (x+a)**2 + y**2 < R**2 or (x-a)**2 + y**2 < R**2):
                return 0.0
            elif( x<a and y<R):
                return 0.0 
            else:
                return 1e6
        else:
            return 1e6
        
    potential_xy = np.vectorize(potential_xy)   
    potkey = 'MANU_quartstadium_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

if(not stad): #Rectangular box
    def potential_xy(x,y):
        if (x>(a+R) or x<(-a-R) or y>R or y<-R):
            return 1e6
        else:
            return 0.0

    potential_xy = np.vectorize(potential_xy)

    potkey = 'MANU_rectangular_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))

#System parameters
m = 0.5
L = (a+R)*1.25
if(stad):
    lbx = -0.1
    lby = -0.1
else:
    lbx = -L
    lby = -1.2*R
ubx = L
uby = 1.2*R
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

x = np.linspace(lbx,ubx,ngridx+1)
#print('Vs',pes.potential_xy(0,0))
fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential_xy)

print('potential',potkey)   

#Diagonalization
param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':neigs}
with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
    f.write('\n'+str(param_dict))

if(1): #Diagonalize the Hamiltonian
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


print('vals', vals[:10])

x_arr = DVR.pos_mat(0)
#-------------------------------------------------------
pos_mat = np.zeros((neigs,neigs)) 
pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
O = pos_mat

diffs = np.diff(np.sort(vals))
if(1):
    #Remove degenerate levels
    print('vals', np.around(vals,2))
    print('Original diffs len:', len(diffs))
    diffs = diffs[diffs>1e-2]
    print('Filtered diffs len:', len(diffs))

    diffs/=np.mean(diffs)
    plt.hist(diffs, bins=20, density=True)
    plt.show()
    exit()

tarr = np.linspace(-50,50,10000)
T = 10.0
beta = 1.0/T
C_tcf = TCF_O(vals, beta, neigs, O, tarr)

freqs = np.fft.fftfreq(len(tarr), d=(tarr[1]-tarr[0]))*2*np.pi
C_tcf_ft = np.fft.fft(C_tcf)
#Plot real part of C_tcf
C_tcf_ft = np.fft.fftshift(C_tcf_ft)
freqs = np.fft.fftshift(freqs)

#Save freqs and C_tcf_ft in column format
fname = 'Spectra_TCF_{}'.format(potkey)
np.savetxt('{}/Datafiles/{}'.format(path,fname), np.column_stack((freqs, np.log(np.abs(C_tcf_ft)))), header='Frequency Log_TCF_FT', comments='')



if(1): #Plot TCF
    plt.plot(tarr, C_tcf.real, label='Re C(t)')
    plt.plot(tarr, C_tcf.imag, label='Im C(t)')
    plt.xlabel('Time t', fontsize=xl_fs)
    plt.ylabel('TCF C(t)', fontsize=yl_fs)

if(0): #Plot TCF FT
    plt.plot(freqs, np.log(np.abs(C_tcf_ft)), label='|C(ω)|')
    plt.xlabel('Frequency ω', fontsize=xl_fs)
    plt.ylabel('Log TCF FT log|C(ω)|', fontsize=yl_fs)
    
    for d in diffs:
        plt.axvline(x=d, color='r', linestyle='--', alpha=0.5)

plt.show()


#Compare rectangular and stadium billiards spectra
fname_rect = 'Spectra_TCF_MANU_rectangular_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))
fname_stad = 'Spectra_TCF_MANU_quartstadium_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))
data_rect = np.loadtxt('{}/Datafiles/{}'.format(path,fname_rect), skiprows=1)
data_stad = np.loadtxt('{}/Datafiles/{}'.format(path,fname_stad), skiprows=1)
freqs_rect = data_rect[:,0]
Ctcf_ft_rect = data_rect[:,1]
freqs_stad = data_stad[:,0]
Ctcf_ft_stad = data_stad[:,1]

plt.plot(freqs_rect, Ctcf_ft_rect, label='Rectangular box')
plt.plot(freqs_stad, Ctcf_ft_stad, label='Quart Stadium billiards')
plt.xlabel('Frequency ω', fontsize=xl_fs)
plt.ylabel('Log TCF FT log|C(ω)|', fontsize=yl_fs)
plt.legend(fontsize=le_fs)
plt.show()


