import numpy as np
from PISC.utils.readwrite import read_arr, store_arr
from matplotlib import pyplot as plt

n_anharm = 4

if(1): #Plot bnarr, mun_arr vs n at different T for Billiards/2D Box

    m=0.5
    R=np.sqrt(1/(4+np.pi))#10
    a=R 

    neigs = 100
    potkey_list = ['lowT_stadium_billiards_a_{}_R_{}'.format(np.around(a,2),np.around(R,2)),
                   'lowT_rectangular_box_a_{}_R_{}'.format(np.around(a,2),np.around(R,2))]

    for potkey in potkey_list: 
        if('stadium' in potkey):
            if a==0.0:
                potkey_trunc = 'CB'
            else:
                potkey_trunc = 'SB' 
        elif('rectangular' in potkey):
            if a==0.0:
                potkey_trunc = 'Sq. Box'
            else:
                potkey_trunc = 'Rec. Box'


        fname = 'mun_arr_{}_neigs_{}'.format(potkey,neigs)
        mun_arr = read_arr(fname)

        fname = 'bnarr_{}_neigs_{}'.format(potkey,neigs)
        bnarr = read_arr(fname)

        T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))
        
        ncoeff = bnarr.shape[1]
        nmoments = (mun_arr.shape[1]-1)*2

        i=0
        for i in [2]:#,4,8,12,16]:
            plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={},neigs={},{}'.format(T_arr[i],neigs,potkey_trunc))

    plt.xlabel(r'$n$')
    plt.ylabel(r'$b_{n}$')
    plt.legend()
    plt.show()

if(0): # Plot bnarr, mun_arr vs n at different T for MAH
    neigs = 550
    a=0.0
    w=0.0
    n_anharm = 4
    b=1.
    n_anharm = 4
    #for n_anharm in [4,6,8,10]:
    for b in [0.1,0.5,1.0]:
    #for w in [1.0,0.0]:
        potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(w,a,b,n_anharm)

        fname = 'mun_arr_{}_neigs_{}'.format(potkey,neigs)
        mun_arr = read_arr(fname)

        fname = 'bnarr_{}_neigs_{}'.format(potkey,neigs)
        bnarr = read_arr(fname)

        T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))

        ncoeff = bnarr.shape[1]
        nmoments = (mun_arr.shape[1]-1)*2

        i=0
        for i in [8,]:#,4,8,12,16]:
            plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={},neigs={},n_anharm={},b={}'.format(T_arr[i],neigs,n_anharm,b))

    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mu_{2n}$')

    plt.legend()
    plt.show()


if(0): # Plot bnarr, mun_arr vs n at different T
    #for neigs in [350,450,550,50]:
    neigs=550
    b=1.
    for n_anharm in [4,6,8]:
    #n_anharm = 6
    #for b in [1.0,0.1]:
        fname = 'mun_arr_MAH_b_{}_order_{}_neigs_{}'.format(b,n_anharm,neigs)
        mun_arr = read_arr(fname)

        fname = 'bnarr_MAH_b_{}_order_{}_neigs_{}'.format(b,n_anharm,neigs)
        bnarr = read_arr(fname)

        T_arr = read_arr('T_arr_MAH_b_{}_order_{}_neigs_{}'.format(b,n_anharm,neigs))

        ncoeff = bnarr.shape[1]
        nmoments = (mun_arr.shape[1]-1)*2

        i=10
        #plt.scatter((np.arange(1,nmoments//2+1)),np.log(mun_arr[i,1:]),label='T={},neigs={}'.format(np.around(T_arr[i],2),neigs))
        for i in [2,]:#,4,8,12,16]:
            plt.scatter(np.arange(40),bnarr[i,:40],label='T={},neigs={},n_anharm={},b={}'.format(T_arr[i],neigs,n_anharm,b))

    plt.xlabel(r'$n$')
    plt.ylabel(r'$b_{n}$')
    #plt.ylim([0.0,500])
    plt.legend()    
    plt.show()

if(0): # Plot bnarr, mun_arr vs T at different n
    #for neigs in [550]:#,450,550]:
    
    neigs = 550
    #for n_anharm in [4,6]:
    for n_anharm in [6]:
        fname = 'mun_arr_MAH_order_{}_neigs_{}'.format(n_anharm,neigs)
        mun_arr = read_arr(fname)

        fname = 'bnarr_MAH_order_{}_neigs_{}'.format(n_anharm,neigs)
        bnarr = read_arr(fname)

        T_arr = read_arr('T_arr_MAH_order_{}_neigs_{}'.format(n_anharm,neigs))

        ncoeff = bnarr.shape[1]
        nmoments = (mun_arr.shape[1]-1)*2

        for i in range(30,40,3):#ncoeff):
            plt.plot(T_arr,bnarr[:,i],label='n={},neigs={}'.format(i,neigs))
           
        #for i in range(15,27,3):#nmoments//2,3):
        #    plt.plot(np.log(T_arr),np.log(mun_arr[:,i]),label='n={},n_nanharm={}'.format(2*i,n_anharm))
    
    plt.xlabel(r'$T$')
    plt.ylabel(r'$\mu_n$')
    #plt.ylim([0.0,500])
    #plt.legend()
    plt.show()
