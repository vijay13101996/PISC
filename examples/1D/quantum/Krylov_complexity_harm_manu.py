import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import argparse
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

ngrid = 1000

L = 30


"""
Tasks to complete:
    1. Compute the moments and Lanczos coefficients assuming random values for Oij and then
    compare with the results from that of a 1D box.
    2. Understand how random values of Oij can yield the same moments as that of a 1D box.
    3. Truncate the O matrix to different subdiagonals and see how the moments and coefficients change.
    4. Understand the striations in the On matrices
    5. Understand why the On's curve downward for sublinear energy scaling and upward for superlinear energy scaling.
    6. Understand the thermodynamic root of the structure of On matrices
"""


def main(m,a,b,omega,n_anharm,L):
    print('b,omega,L',b,omega,L)

    lb = -L
    ub = L
    pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

    potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(L,omega,a,b,n_anharm)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 200
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    print('vals', vals[-1])

    x_arr = DVR.grid[1:ngrid]
    dx = x_arr[1]-x_arr[0]

    #plt.plot(x_arr,vecs[:,-1])
    #plt.show()

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
    O = (pos_mat)
    
    if(0):
        #print machine epsilon
        eps = np.finfo(float).eps
        print('eps',eps)
        for i in range(len(O)):
            for j in range(len(O)):
                if (i+j)%2 == 0:
                    O[i,j] = 0.0

        plt.plot(np.diag(abs(O),k=39))
        #plt.imshow(np.abs(O),vmin=0,vmax=1)
        plt.show()
        exit()

    if(1): #Random O
        rng = np.random.default_rng()
        O = np.zeros((neigs,neigs))
        for i in range(neigs):
            for j in range(i,neigs):
                O[i,j] = 1e-13#*rng.normal(0,1)#*1e-13
                O[j,i] = O[i,j]


    mom_mat = np.zeros((neigs,neigs))
    mom_mat = Krylov_complexity.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dx, mom_mat)
    P = mom_mat

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    T_arr = np.array([1,2,3,4,5])#,5,10,20])
    mun_arr = []
    mu0_harm_arr = []
    mu_all_arr = []
    bnarr = []
    nmoments = 60
    ncoeff = 200

    for T_au in T_arr: 
        Tkey = 'T_{}'.format(T_au)

        beta = 1.0/T_au 

        moments = np.zeros(nmoments+1)
        moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, 'asm', 0.5, moments)
        even_moments = moments[0::2]

        barr = np.zeros(ncoeff)
        barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
        bnarr.append(barr)

        mun_arr.append(even_moments)
        mu_all_arr.append(moments)

        mu0 = 1.0/(2*m*omega*np.sinh(0.5*beta*omega))
        print('mu0',mu0, even_moments[0] )
        mu0_harm_arr.append(mu0*omega**(2*np.arange(0,nmoments//2+1)))
    
    mun_arr = np.array(mun_arr)
    mu0_harm_arr = np.array(mu0_harm_arr)
    mu_all_arr = np.array(mu_all_arr)
    bnarr = np.array(bnarr)

    if(1): #Plot bn (assuming O is random)
        fig,ax = plt.subplots(1,2)
        plt.subplots_adjust(wspace=0.35,hspace=0.35)

        for i in range(len(T_arr)):
            ax[0].scatter(np.arange(0,ncoeff),bnarr[i,:],label=r'$T={}$'.format(T_arr[i]),s=10)
            ax[1].scatter(np.arange(0,nmoments//2+1),np.log(mun_arr[i,:]),s=10)
            ax[1].plot(np.arange(0,nmoments//2+1),np.log(mun_arr[i,:]))

        ax[0].set_xlabel(r'$n$',fontsize=xl_fs)
        ax[0].set_ylabel(r'$b_n$',fontsize=xl_fs)
        ax[0].set_xticks([0,25,50,75,100])#,150,200])
        ax[0].set_xlim([-1,101])
        ax[0].set_ylim([0.0,vals[-1]*0.6])
        ax[0].tick_params(axis='both',labelsize=ti_fs)
        ax[0].annotate(r'$(a)$', xy=(0.05, 0.9), xytext=(0.05, 0.9), textcoords='axes fraction', fontsize=xl_fs)


        ax[1].set_xlabel(r'$n$',fontsize=xl_fs)
        ax[1].set_ylabel(r'$log \: \mu_{2n}$',fontsize=xl_fs)
        ax[1].set_xticks([0,10,20,30])#,40])
        ax[1].set_ylim([min(np.log(mun_arr[-1,:]))*1.2,310])
        ax[1].tick_params(axis='both',labelsize=ti_fs)
        ax[1].annotate(r'$(b)$', xy=(0.05, 0.9), xytext=(0.05, 0.9), textcoords='axes fraction', fontsize=xl_fs)

        fig.legend(fontsize=le_fs-2,loc=(0.11,0.91),ncol=5)
        fig.set_size_inches(7,3.5)	
        fig.savefig('/home/vgs23/Images/bn_vs_n_harm_unit.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)
        plt.show()

    if(0): #Plot moments

        for i in range(neigs):
            for j in range(neigs):
                if abs(i-j)>1:
                    O[i,j] = O[i,j]#1e-13

        #Compute moments
        
        Z = np.sum(np.exp(-beta*vals))

        mu2n = np.zeros(nmoments//2+1)
        maxval  = 0.0
        for i in range(neigs):
            for j in range(neigs):
                mu2n += abs(O[i,j])**2*np.exp(-0.5*beta*(vals[i]+vals[j]))*(vals[i]-vals[j])**(2*np.arange(0,nmoments//2+1))
                val = np.exp(-0.5*beta*(vals[i]+vals[j]))*(vals[i]-vals[j])**(2*np.arange(0,nmoments//2+1))
    

                #mu2n += (vals[i]-vals[j])**(2*np.arange(0,nmoments//2+1))*np.exp(-0.5*beta*(vals[i]+vals[j]))#*abs(O[i,j])**2
                #mu2n = (vals[80]-vals[0])**(2*np.arange(0,nmoments//2+1))*np.exp(-0.5*beta*(vals[80]+vals[0]))
                
        print('maxval',maxval)
        print('error', (vals[-1]-vals[0])**(2*14),np.exp(-0.5*beta*(vals[-1]+vals[0]))*abs(O[0,-1])**2)

        mu2n = mu2n/Z

        fig,ax = plt.subplots(1,1)

        for i in range(len(T_arr)):
            ax.scatter(np.arange(0,nmoments//2+1),np.log(mun_arr[i,:]),label=r'$Numerical$')
            ax.plot(np.arange(0,nmoments//2+1),np.log(mu0_harm_arr[i,:]),label=r'$Analytical$',color='k')
            ax.plot(np.arange(0,nmoments//2+1),np.log(mu2n),label=r'$Exact$',color='r')
        ax.set_xlabel(r'$n$',fontsize=xl_fs)
        ax.set_ylabel(r'$\mu_{2n}$',fontsize=xl_fs)
        ax.legend(fontsize=le_fs)
        ax.tick_params(axis='both',labelsize=ti_fs)


        fig.set_size_inches(4,4)	
        #fig.savefig('/home/vgs23/Images/mun_vs_n_harm.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)
        plt.show()



    store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
    store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
    store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))
    exit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute moments for a mildly anharmonic potential')
    parser.add_argument('--m', type=float, default=1.0, help='mass')
    parser.add_argument('--a', type=float, default=0.0, help='Cubic anharmonicity')
    parser.add_argument('--b', type=float, default=0.0, help='quartic anharmonicity')
    parser.add_argument('--omega', type=float, default=2.0, help='harmonic frequency')
    parser.add_argument('--n_anharm', type=int, default=4, help='number of anharmonic terms')
    parser.add_argument('--L', type=float, default=L, help='grid length')

    args = parser.parse_args()
    main(args.m,args.a,args.b,args.omega,args.n_anharm,args.L)

exit()
