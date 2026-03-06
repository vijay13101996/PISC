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

R = 1.0#0.37  #np.sqrt(1/(4+np.pi))
a = R  #0.0

neigs = 500 #Number of eigenstates to be calculated

bill = 1
diag = 0

path = os.path.dirname(os.path.abspath(__file__))

#System parameters
m = 0.5
mul = 1.05
L = (a+R)*mul#25
lbx = -L
ubx = L
lby = -mul*R
uby = mul*R
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

start_time = time.time()
print('R',R)

def plot_potential(potential_xy):
    xg = np.linspace(lbx,ubx,ngridx)
    yg = np.linspace(lby,uby,ngridy)
    xgr,ygr = np.meshgrid(xg,yg)
    plt.contour(xgr,ygr,potential_xy(xgr,ygr),levels=np.arange(-1,30,1.0))
    #plt.imshow(potential_xy(xgr,ygr),origin='lower')
    plt.show()    
    exit()

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

    #plot_potential(potential_xy)

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
    L_x = 2*(a+R)
    L_y = 2*R
    vals_x = (np.pi**2 * (np.arange(1,neigs+1))**2 )/(2*m*L_x**2)
    vals_y = (np.pi**2 * (np.arange(1,neigs+1))**2 )/(2*m*L_y**2)

    # Combine vals_x and vals_y to get 2D box eigenvalues
    vals_box_2D = []
    for nx in range(1,neigs+1):
        for ny in range(1,neigs+1):
            vals_box_2D.append(vals_x[nx-1] + vals_y[ny-1])
    vals_box_2D = np.array(sorted(vals_box_2D))[:neigs]
    print('Analytical 2D box vals:', vals_box_2D[:10])
    print('diff', (vals[:20]-vals_box_2D[:20])/vals_box_2D[:20]*100)

    #plt.imshow(DVR.eigenstate(vecs[:,0])**2,origin='lower')
    #plt.show()

    return DVR, vals, vecs, potkey

tarr = np.linspace(-200,200,10000)
T = 5.0
beta = 1.0/T

#fig,ax = plt.subplots(figsize=(5,5))

for pot in ['stadium','box']:#,'stadium']:
    for n in [20]: #np.arange(0,40,2):#,20]:
        DVR, vals, vecs, potkey = bookkeeping(pot)
        print('Computing data for {}...'.format(pot))

        x_arr = DVR.pos_mat(0)
        pos_mat = np.zeros((neigs,neigs)) 
        pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
        O = pos_mat
    
        #O[:] = 1.0
        ip = 0.0
        ip = Krylov_complexity_2D.krylov_complexity.compute_ip(O, O, beta, vals, 0.5, ip, 'wgm')
        print('ip(O,O)', ip)
        O/=np.sqrt(ip)
        
        #O[:] = 1.0
        L = np.zeros((neigs,neigs))
        L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,L)
        ncoeff = 50
        print('O.shape', O.shape, neigs, L.shape)

        LnO = O.copy()
        for i in range(n-1):
            LnO = L*LnO

        On = np.zeros((neigs, neigs), dtype=complex)
        bnarr = np.zeros(ncoeff)

        bnarr, On = Krylov_complexity_2D.krylov_complexity.compute_on_matrix(O, L, bnarr, beta, vals, 0.5, 'wgm', On, n+1)
   
        #On = On- LnO
        #Compute total variation of On
        TV = 0.0
        for j in range(neigs):
            for k in range(neigs-1):
                TV += np.abs(On[j,k+1]-On[j,k])

        if(0):
            if pot=='box':
                if(n==0):
                    plt.scatter(n, np.log(TV), label=pot, color='blue')
                else:
                    plt.scatter(n, np.log(TV), color='blue')
            else:
                if(n==0):
                    plt.scatter(n, np.log(TV), label=pot, color='orange')
                else:                
                    plt.scatter(n, np.log(TV), color='orange')

        if(1):
            fig, ax = plt.subplots(2,1, figsize=(5,10))
            #On = On - LnO
            ax[0].imshow(np.log(np.abs(On)))
            ax[0].set_xlabel(r'$k$', fontsize=xl_fs)
            ax[0].set_ylabel(r'$j$', fontsize=yl_fs)
            ax[0].axhline(y=4, color='r', linestyle='--')
            ax[0].set_title(r'$\log|O_n|$ matrix for {}, n={}'.format(pot,n), fontsize=tp_fs)
            ax[1].plot(np.log(np.abs(On[:,4])), label=pot+' n={}'.format(n))
            ax[1].set_xlabel('k', fontsize=xl_fs)
            ax[1].set_ylabel(r'$\log |O_{jk}|$ at j=4', fontsize=yl_fs)
            plt.tight_layout()
            #plt.savefig('On_matrix_and_cross_{}_n_{}_u.png'.format(pot,n), dpi=300)
            plt.show()
        
        if(0):
            On_cross = On[:,4]#int(neigs/2)]

            ax.plot(np.log(np.abs(On_cross)), label=pot+' n={}'.format(n), alpha=0.5)
            ax.set_xlabel('k', fontsize=xl_fs)
            ax.set_ylabel('$\log |O_{jk}|$ at j=4', fontsize=yl_fs)
            #plt.legend(fontsize=le_fs)

plt.legend(fontsize=le_fs)
plt.xlabel('n', fontsize=xl_fs)
plt.ylabel('Total Variation of $O_n$', fontsize=yl_fs)
plt.title('Total Variation of $O_n$ vs n for Stadium and Box Billiards', fontsize=tp_fs)
plt.tight_layout()
plt.show()
exit()
if(0):
    for pot in ['stadium','box']:
        DVR, vals, vecs, potkey = bookkeeping(pot)
        print('Computing data for {}...'.format(pot))

        x_arr = DVR.pos_mat(0)
        pos_mat = np.zeros((neigs,neigs)) 
        pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
        O = pos_mat

        O[:] = 1.0
        ip = 0.0
        ip = Krylov_complexity_2D.krylov_complexity.compute_ip(O, O, beta, vals, 0.5, ip, 'wgm')
        print('ip(O,O)', ip)
        O/=np.sqrt(ip)
        
        L = np.zeros((neigs,neigs))
        L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,L)
        ncoeff = 50
        print('O.shape', O.shape, neigs, L.shape)
        On = np.zeros((neigs, neigs), dtype=complex)
        bnarr = np.zeros(ncoeff)

        bnarr, On = Krylov_complexity_2D.krylov_complexity.compute_on_matrix(O, L, bnarr, beta, vals, 0.5, 'wgm', On, n+1)

        #plt.imshow(np.log(np.abs(On)))
        #plt.show()
        
        if(1):
            On_cross = On[:,4]#int(neigs/2)]
            ax.plot(np.log(np.abs(On_cross)), label=pot+r' n=20, $\hat{O}=\hat{u}$', lw=2)

ax.legend(fontsize=le_fs-2)
plt.tight_layout()
plt.savefig('On_cross_comparison_stadium_box_u_operator.png', dpi=300)
plt.show()

exit()

for pot in ['box','stadium']:
    DVR, vals, vecs, potkey = bookkeeping(pot)
    print('Computing data for {}...'.format(pot))

    x_arr = DVR.pos_mat(0)
    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
    O = pos_mat
            
    ip = 0.0
    ip = Krylov_complexity_2D.krylov_complexity.compute_ip(O, O, beta, vals, 0.5, ip, 'wgm')
    print('ip(O,O)', ip)
    O/=np.sqrt(ip)

    #O2_avg = compute_O2_avg(O, vals, T=10.0, neigs=neigs)
    #print('<O^2>_T=', O2_avg)
    #diffs = np.diff(np.sort(vals))
    
    tarr_short = np.linspace(-50,50,10000)

    if(1):
        L = np.zeros((neigs,neigs))
        L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,L)
        n=20
        m=20
        ncoeff = 50
        print('O.shape', O.shape, neigs, L.shape)
        On = np.zeros((neigs, neigs), dtype=complex)
        Om = np.zeros((neigs, neigs), dtype=complex)
        #O[:] = 1.0
        bnarr = np.zeros(ncoeff)
        bmarr = np.zeros(ncoeff)
        

        bnarr, On = Krylov_complexity_2D.krylov_complexity.compute_on_matrix(O, L, bnarr, beta, vals, 0.5, 'wgm', On, n+1)
        bmarr, Om = Krylov_complexity_2D.krylov_complexity.compute_on_matrix(O, L, bmarr, beta, vals, 0.5, 'wgm', Om, m+1)

        ip_nm = 0.0
        ip_nm = Krylov_complexity_2D.krylov_complexity.compute_ip(On/bnarr[n], Om/bmarr[m], beta, vals, 0.5, ip_nm, 'wgm')
        print('Inner product <O_n|O_m> with n={} m={} : {}'.format(n, m, ip_nm))

        # Plot cross section of On at index, say k=100
        On_cross = On[:,int(neigs/2)]
    
        # Plot antidiagonal cross section
        On_antidiag = np.array([On[i,neigs-1-i] for i in range(neigs)]) 
        plt.plot(np.log(np.abs(On_cross)), label=pot+' n={}'.format(n))
        #plt.plot(np.log(np.abs(On_antidiag)), label=pot+' n={} antidiag'.format(n))    
        plt.xlabel('Index', fontsize=xl_fs)
        plt.ylabel('|O_n| cross section', fontsize=yl_fs)
        plt.legend(fontsize=le_fs)
        #plt.show() 


        #plt.imshow(np.log(np.abs(Om)))
        #plt.axhline(y=int(neigs/2), color='r', linestyle='--')
        #plt.axvline(x=int(neigs/2), color='r', linestyle='--')
        #plt.colorbar()
        #plt.title('|O_n| with n={}'.format(n))
        #plt.show()

    if(0):
        C_tcf_short = TCF_O(vals, beta, neigs, O, tarr_short)
        C_tcf = TCF_O(vals, beta, neigs, O, tarr)

        freqs = np.fft.fftfreq(len(tarr), d=(tarr[1]-tarr[0]))*2*np.pi
        C_tcf_ft = np.fft.fft(C_tcf)
        #Plot real part of C_tcf
        C_tcf_ft = np.fft.fftshift(C_tcf_ft)
        freqs = np.fft.fftshift(freqs)

        freqs_short = np.fft.fftfreq(len(tarr_short), d=(tarr_short[1]-tarr_short[0]))*2*np.pi
        C_tcf_ft_short = np.fft.fft(C_tcf_short)
        C_tcf_ft_short = np.fft.fftshift(C_tcf_ft_short)
        freqs_short = np.fft.fftshift(freqs_short)
    if(1):
        coeffs = 200
        tarr_pos = np.linspace(0,100,5000)
        barr = Krylov_O(vals, beta, neigs, O, coeffs)
        coeff_arr = np.arange(coeffs) 
    if(0):
        plt.plot(coeff_arr, barr, label=pot)
        plt.xlabel('n', fontsize=xl_fs)
        plt.ylabel('Lanczos b_n', fontsize=yl_fs)
        plt.legend(fontsize=le_fs)
    if(0):
        plt.plot(tarr_short, C_tcf_short.real, label='TCF Short t')
        #plt.plot(tarr, C_tcf.real, label='TCF Full t',alpha=0.5)
        plt.xlabel('Time t', fontsize=xl_fs)
        plt.ylabel('C(t)', fontsize=yl_fs)
        plt.legend(fontsize=le_fs)
        #plt.show()

    if(0):
        plt.plot(freqs_short, np.log(np.abs(C_tcf_ft_short)), label='TCF Short t')
        plt.plot(freqs, np.log(np.abs(C_tcf_ft)), label='TCF Full t',alpha=0.5)
        #plt.xlim(-5,5)
        plt.xlabel('Frequency ω', fontsize=xl_fs)
        plt.ylabel('|C(ω)|', fontsize=yl_fs)
        plt.legend(fontsize=le_fs)
        plt.show()

    #verify_Ct(O, vals, barr, beta, neigs, tarr=tarr_pos)
plt.show()
