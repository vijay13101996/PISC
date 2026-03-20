import numpy as np
from PISC.utils.readwrite import read_arr, store_arr
from matplotlib import pyplot as plt

n_anharm = 4

if(1):
    omega = 2.0
    b = 0.0
    neigs = 400
    a = 0.0
    m = 1.0
    n_anharm = 4
    L = 20.0
    #for L in [5.0,10.0,20.0,40.0,80.0,120.0,200.0]:
    
    for omega in [1.0]:#,2.0,3.0,4.0,5.0]:
        potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(L,omega,a,b,n_anharm)

        fname = 'mun_arr_{}_neigs_{}'.format(potkey,neigs)
        mun_arr = read_arr(fname)

        fname = 'bnarr_{}_neigs_{}'.format(potkey,neigs)
        bnarr = read_arr(fname)

        T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))

        ncoeff = bnarr.shape[1]
        nmoments = (mun_arr.shape[1])

        for i in [0,1,2,3]:#range(len(T_arr)):
            logdata = np.log(mun_arr[i,:])
            diff = np.diff(logdata)
            plt.scatter(np.arange(nmoments),np.log(mun_arr[i,:]),label='T={}, MAH, L={}'.format(T_arr[i],L))
            
            #plt.scatter(np.arange(1,nmoments),diff,label='T={}, MAH, L={}'.format(T_arr[i],L))

    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mu_{2n}$')
    plt.legend()
    plt.show()

    exit()

if(0): 
    #mom = False
    mom = True
    indlist = [2]

    m=1.0

    neigs = 400
    L = 40
    potkey = 'TEMP_1D_Box_m_{}_L_{}'.format(m,np.around(L,2))
    
    fname = 'mun_arr_{}_neigs_{}'.format(potkey,neigs)
    mun_arr = read_arr(fname)

    fname = 'bnarr_{}_neigs_{}'.format(potkey,neigs)
    bnarr = read_arr(fname)

    T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))

    ncoeff = bnarr.shape[1]
    nmoments = (mun_arr.shape[1])

    for i in indlist:#range(len(T_arr)):
        if(mom):
            xarr = np.arange(nmoments)
            yarr = np.log(mun_arr[i,:])
            plt.scatter(xarr,yarr,label='T={}, 1D Box'.format(T_arr[i],neigs))
            
            #Fit xarr,yarr to a line
            xarr = xarr[25:47]
            yarr = yarr[25:47]
            p = np.polyfit(xarr,yarr,1)
            slope = p[0]
            off = p[1]
            plt.plot(xarr,slope*xarr+off,label='T={}, 1D Box, slope={}'.format(T_arr[i],np.around(slope,2)),color='black',lw=3)
            print('slope',slope)
        else:
            plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={}, 1D Box'.format(T_arr[i],neigs),s=3)

    omega = 2.0
    a = 0.0
    b = 1e-4
    n_anharm = 4
    potkey = 'TEMP_Pert_wcomp_MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,n_anharm)
    
    fname = 'mun_arr_{}_neigs_{}'.format(potkey,neigs)
    mun_arr = read_arr(fname)

    fname = 'bnarr_{}_neigs_{}'.format(potkey,neigs)
    bnarr = read_arr(fname)

    T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))

    ncoeff = bnarr.shape[1]
    nmoments = (mun_arr.shape[1])#*2

    for i in indlist:#range(len(T_arr)):
        if(mom):
            plt.scatter(np.arange(nmoments),np.log(mun_arr[i,:]),label='T={}, MAH, b={}'.format(T_arr[i],b))

            #Fit xarr,yarr to a line
            xarr = np.arange(nmoments)
            yarr = np.log(mun_arr[i,:])
            xarr = xarr[25:47]
            yarr = yarr[25:47]
            p = np.polyfit(xarr,yarr,1)
            slope = p[0]
            off = p[1]
            plt.plot(xarr,slope*xarr+off,label='T={}, MAH, slope={}'.format(T_arr[i],np.around(slope,2)),color='black',lw=3)
            print('slope',slope)

        else:
            plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={}, MAH, b={}'.format(T_arr[i],b),s=3)

    plt.xlabel(r'$n$')
    if(mom):
        plt.ylabel(r'$\mu_{2n}$')
    else:
        plt.ylabel(r'$b_{n}$')

    plt.legend()
    plt.show()
    exit()


if(0): #Plot mun_arr vs n for different temperatures
    m=1.0
    neigs = 450
    a=0.0
    b=0.0
    w=4.0

    potkey = 'Pert_TEMP_MAH_w_{}_a_{}_b_{}_n_{}'.format(w,a,b,n_anharm)
    
    fname = 'mun_arr_{}_neigs_{}'.format(potkey,neigs)
    mun_arr = read_arr(fname)

    fname = 'bnarr_{}_neigs_{}'.format(potkey,neigs)
    bnarr = read_arr(fname)

    T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))

    ncoeff = bnarr.shape[1]
    nmoments = (mun_arr.shape[1])#*2

    print(nmoments,np.shape(mun_arr[0,:]))

    for i in range(len(T_arr)):

        #plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={},neigs={},b={}'.format(T_arr[i],neigs,b))
        plt.scatter(np.arange(nmoments),np.log(mun_arr[i,:]),label='T={},neigs={},b={}'.format(T_arr[i],neigs,b))

    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mu_{n}$')
    plt.legend()
    plt.show()
    exit()

if(0): #Plot bnarr, mun_arr vs n at different perturbations for MAH
    m=1.0
    neigs = 150
    a=0.0
    w=4.0
    b=0.1
    for b in [0.0, 1e-4, 1e-6, 1e-8, 1e-9]:#,1.0,0.1,0.01,0.001]:#,0.001,0.0001,0.00001]:
    #for b in [0.0,0.01,0.1,0.5]:
    #for w in [2.0,1.0,0.5,0.0]:
        potkey = 'Pert_wcomp_MAH_w_{}_a_{}_b_{}_n_{}'.format(w,a,b,n_anharm)
        
        fname = 'mun_arr_{}_neigs_{}'.format(potkey,neigs)
        mun_arr = read_arr(fname)

        fname = 'bnarr_{}_neigs_{}'.format(potkey,neigs)
        bnarr = read_arr(fname)

        T_arr = read_arr('T_arr_{}_neigs_{}'.format(potkey,neigs))

        ncoeff = bnarr.shape[1]
        nmoments = (mun_arr.shape[1])#*2

        print(nmoments,np.shape(mun_arr[0,:]))
        i=0
        plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={},neigs={},b={}'.format(T_arr[i],neigs,b))
        #plt.scatter(np.arange(nmoments),np.log(mun_arr[i,:]),label='T={},neigs={},b={},w={}'.format(T_arr[i],neigs,b,w))


    plt.xlabel(r'$n$')
    plt.ylabel(r'$b_{n}$')
    plt.legend()
    plt.show()
    exit()

