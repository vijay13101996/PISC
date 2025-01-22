import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import argparse

ngrid = 1000

L = 40

#m=1.0
#a=0.0#4
#b=0.0#1e-9
#omega=2.0
#n_anharm=4


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
    neigs = 400
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    print('vals', vals[-1])

    x_arr = DVR.grid[1:ngrid]
    dx = x_arr[1]-x_arr[0]

    #plt.plot(x_arr,vecs[:,-1])
    #plt.show()

    if(0):
        def comp_pos_mat(i,j):
            if(i==j):
                return 0.0
            elif(i-j==1):
                return np.sqrt(1/(2*m*omega))*np.sqrt(j+1)
            elif(j-i==1):
                return np.sqrt(1/(2*m*omega))*np.sqrt(i+1)
            else:
                return 0.0

        pos_mat_anal = np.zeros((neigs,neigs))
        for i in range(neigs):
            for j in range(neigs):
                pos_mat_anal[i,j] = comp_pos_mat(i,j)

        vals_anal = np.zeros(neigs)
        for i in range(neigs):
            vals_anal[i] = omega*(i+0.5)


    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
    O = (pos_mat)
    #O[abs(O)<1e-12] = 0.0 
    if(0):

        O[:] = 0.0
        for i in range(neigs):
            for j in range(i,neigs):
                if(abs(i-j)%2==0 and abs(i-j)<=200):
                    O[i,j] = np.random.normal(0,1)*1e-12
                    O[j,i] = O[i,j]


    #O[abs(O)<1e-12] = 0.0
    #O = np.matmul(O,O) 

    if(0):
        for k in [5,7]:#,5,7]:#np.arange(0,10,2):
            plt.plot((abs(np.diag(O,k))[:]),label='k={}'.format(k))
        plt.legend()
        plt.show()
        #exit()

    mom_mat = np.zeros((neigs,neigs))
    mom_mat = Krylov_complexity.krylov_complexity.compute_mom_matrix(vecs, vals, x_arr, m, dx, dx, mom_mat)
    P = mom_mat

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    T_arr = [1.0,2.0]#2.0,4.0]#0.5,0.5,1.0,2.0]
    mun_arr = []
    mu0_harm_arr = []
    mu_all_arr = []
    bnarr = []
    nmoments = 100
    ncoeff = 200

    if(0):
        n_even = nmoments//2
       
        start = 0
        even_moments = np.zeros(n_even+1-start)
        sum_anal = np.zeros_like(even_moments)
        sum_num = np.zeros_like(even_moments)
        for i in range(len(vals)):
            for j in range(len(vals)):
                if(abs(i-j)<=1):
                    sum_num += np.exp(-0.5*(vals[i]+vals[j])/T_arr[0])*(vals[i]-vals[j])**(2*np.arange(start,n_even+1))*abs(pos_mat[i,j])**2
                    rand = np.random.normal(0,1)#*1e-14
                    sum_anal += np.exp(-0.5*(vals_anal[i]+vals_anal[j])/T_arr[0])*(vals_anal[i]-vals_anal[j])**(2*np.arange(start,n_even+1))*abs(pos_mat_anal[i,j])**2
                    #if(abs(pos_mat[i,j])>1e-6):
                    #    print('i,j',i,j,abs(abs(pos_mat_anal[i,j])-abs(pos_mat[i,j])),pos_mat_anal[i,j],pos_mat[i,j])

        plt.plot(np.arange(start,n_even+1),np.log(sum_anal))
        plt.scatter(np.arange(start,n_even+1),np.log(sum_num))
        plt.show()
        exit()

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

    #print('mun_arr',mun_arr.shape,mu0_harm_arr.shape)
    #print('mun_arr',mun_arr[0,:5],mu0_harm_arr[0,:5])
    print('bnarr',bnarr.shape,bnarr[0,:12])

    if(1):
        #plt.scatter(np.arange(ncoeff),bnarr[0,:]/(np.pi*T_arr[0]),label='T={},neigs={},b={}'.format(T_arr[0],neigs,b))
        #plt.plot(np.arange(ncoeff),np.arange(ncoeff))
        #plt.scatter(np.arange(nmoments//2+1),np.log(mun_arr[0,:]),label='T={},neigs={},b={}'.format(T_arr[0],neigs,b))
        plt.scatter(np.arange(0,nmoments+1),np.log(mu_all_arr[0,:]),label='T={},neigs={},b={}'.format(T_arr[0],neigs,b))
        plt.xlabel(r'$n$')
        plt.ylabel(r'$b_{n}$')
        plt.legend()
        plt.show()

    store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
    store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
    store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))
    exit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute moments for a mildly anharmonic potential')
    parser.add_argument('--m', type=float, default=1.0, help='mass')
    parser.add_argument('--a', type=float, default=0.0, help='quartic anharmonicity')
    parser.add_argument('--b', type=float, default=0.0, help='quartic anharmonicity')
    parser.add_argument('--omega', type=float, default=4.0, help='harmonic frequency')
    parser.add_argument('--n_anharm', type=int, default=4, help='number of anharmonic terms')
    parser.add_argument('--L', type=float, default=L, help='grid length')

    args = parser.parse_args()
    main(args.m,args.a,args.b,args.omega,args.n_anharm,args.L)

exit()

if(0):
    for i in [0]:#,1,2,3]:#,6,8,10,12,14,16]:
        plt.scatter((np.arange(1,nmoments//2+1)),np.log(mun_arr[i,1:]),label='T={}'.format(np.around(T_arr[i],2)))
        #plt.scatter((np.arange(1,nmoments//2+1)),np.log(mu0_harm_arr[i,1:]),label='T={}'.format(np.around(T_arr[i],2)),color='black',s=2) 
        
    #plt.xlim([10,nmoments//2])
    
    plt.title(r'$neigs={}$'.format(neigs))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mu_{2n}$')
    #plt.ylim([0.0,2000])
    plt.legend()    
    plt.show()
    exit()

if(0):
    for i in [0]:#,1,2,3,4,5,6]:#,6,8,10,12,14,16]:
        plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={}'.format(T_arr[i]))
    
    #plt.xlim([10,nmoments//2])
    
    #plt.title(r'$neigs={},L={}$'.format(neigs,ub))
    
    plt.title(r'$\omega={},a={},b={},n={}$'.format(omega,a,b,n_anharm))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$b_n$')
    #plt.ylim([0.0,2000])
    plt.legend()    
    plt.show()
    exit()

slope_arr = []
mom_list = range(0,nmoments//2+1)
for i in mom_list:#,5,6,7,8,9,10]:#,3,4]:#range(0,ncoeff,2):
    #plt.scatter(T_arr,bnarr[:,i],label='n={}'.format(i))
    
    #plt.scatter((T_arr),(mun_arr[:,i]),label='n={}'.format(2*i))
    plt.scatter(np.log(T_arr),np.log(mun_arr[:,i]),label='n={}'.format(2*i))
    plt.scatter(np.log(T_arr),np.log(mu0_harm_arr[:,i]),label='n={}'.format(2*i),color='black')

    lT_arr = np.log(T_arr)
    lmun_arr = np.log(mun_arr[:,i])

    p = np.polyfit(lT_arr,lmun_arr,1)

    slope_arr.append(p[0])
    
    plt.plot(lT_arr,p[0]*lT_arr+p[1],label='n={}'.format(2*i))


    # Fit mun_arr[:,i] to a T_arr**(i/2)
    #p = np.polyfit(T_arr,mun_arr[:,i],i/2)
    #print('p',p)
    #plt.plot(T_arr,p[0]*T_arr**(i/2),label='n={}'.format(2*i))

#plt.scatter(T_arr,np.array(mu0_harm_arr),label='n=0, harm',color='black')
plt.xlabel(r'$log(T)$')
plt.ylabel(r'$log(\mu_{2n})$')
plt.legend()
plt.show()

exit()

plt.scatter(mom_list,np.array(slope_arr))
plt.xlabel(r'$n$')
plt.ylabel(r'$slope$')
plt.show()

if(0):
    for i in [1,2,3]:#range(1):#nmoments//2+1):
        # Fit times_arr vs mun_arr[:,i] to a line
        p = np.polyfit(times_arr,mun_arr[:,i],1)
        print('p',p)
        plt.plot(times_arr,p[0]*times_arr+p[1],label='n={}'.format(2*i))
        plt.scatter(times_arr,mun_arr[:,i],label='n={}'.format(2*i))
    #plt.scatter(times_arr,mun_arr)
    plt.legend()
    plt.show()
    

#ncoeffs = 20
#barr = np.zeros(ncoeffs)
#barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')

#b0 = np.sqrt(1/(2*m*w*np.sinh(0.5*beta*w)))

#print('barr',barr,b0)

#plt.scatter(np.arange(ncoeffs),barr)
#plt.show()


