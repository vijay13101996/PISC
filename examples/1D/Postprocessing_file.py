import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.potentials import double_well, quartic, morse, mildly_anharmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.utils.nmtrans import FFT

dim = 1

if(1): #Double well potential
    lamda = 2.0
    g = 0.02
    Vb = lamda**4/(64*g)

    Tc = lamda*(0.5/np.pi)
    times = 4.0#0.95
    T = times*Tc
    beta=1/T
    print('T',T)

    m = 0.5
    N = 1000
    dt = 0.005

    time_total = 5.0#

    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
    pes = double_well(lamda,g)

    Tkey = 'T_{}Tc'.format(times)

if(1): #Quartic
    a = 1.0

    pes = quartic(a)

    T = 2.5#0.125

    m = 1.0
    N = 1000
    dt_therm = 0.05
    dt = 0.01

    time_therm = 50.0
    time_total = 30.0

    potkey = 'quartic_a_{}'.format(a)
    Tkey = 'T_{}'.format(T)

if(0): #Mildly anharmonic
    omega = 1.0
    a = 0.0#4     #0.4#-0.605#0.5#
    b = 0.0#16    #a**2#0.427#a**2#

    T = 2.0 #times*Tc
    beta = 1/T

    m = 1.0
    N = 1000
    dt_therm = 0.05
    dt = 0.01

    time_therm = 50.0
    time_total = 30.0

    pes = mildly_anharmonic(m,a,b)

    potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)
    Tkey = 'T_{}'.format(np.around(T,3))

if(0): #Morse_SB
    m=1.0
    delta_anh = 0.05#1
    w_10 = 1.0
    wb = w_10
    wc = w_10 + delta_anh
    alpha = (m*delta_anh)**0.5
    D = m*wc**2/(2*alpha**2)

    pes = morse(D,alpha)
    T = 1.0#TinK*K2au
    beta = 1/T

    potkey = 'Morse_D_{}_alpha_{}'.format(D,alpha)
    Tkey = 'T_{}'.format(T)
    dt = 0.002
    time_total = 20.0

if(0): #Morse
    m=0.5
    D = 9.375
    alpha = 0.382
    pes = morse(D,alpha)

    w_m = (2*D*alpha**2/m)**0.5
    Vb = D/3

    print('alpha, w_m', alpha, Vb/w_m)
    T = 3.18#*0.3
    beta = 1/T
    potkey = 'morse'
    Tkey = 'T_{}'.format(T)

    N = 1000
    dt_therm = 0.05
    dt = 0.02
    time_therm = 50.0
    time_total = 5.0

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
#path = '/scratch/vgs23/PISC/examples/1D'#
path = os.path.dirname(os.path.abspath(__file__))
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'Im_qq_TCF'#'OTOC'#'R2'#'singcomm' #
syskey = 'Papageno'#'Tosca2'

if(1):#RPMD
    nbeads = 32
    tarr = np.arange(0,nbeads)
    OTOCarr = np.zeros_like(tarr) +0j
    #potkey = 'harmonic_omega_{}'.format(1.0)
    #Tkey = 'T_{}'.format(4.0)
    beadkey = 'nbeads_{}_'.format(nbeads)
    
    if(1): ##Collect files of thermal ensembles
        methodkey = 'RPMD'
        enskey = 'thermal' #'const_q' #

        kwlist = [methodkey,enskey+'_',corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]

        tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr,allseeds=False,seedcount=400)
        if(corrkey!='OTOC'):
            OTOCarr/=nbeads
            plt.plot(tarr,OTOCarr)
        else:
            plt.plot(tarr,np.log(abs(OTOCarr)))
        store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext)
        plt.show()
        exit()
        ext = rpext + 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,'OTOC',potkey,Tkey,nbeads,dt)
        slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2,3)#3.4,4.4)#1.8,3.8)#3.2,4.2)
        plt.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='m')
        plt.show()

        exit()
        

    if(0): ##Collect files of microcanonical ensembles
        methodkey = 'RPMD'
        enskey  = 'mc'
        E = 1.3
        Ekey = 'E_{}'.format(E)
        kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey,beadkey,Ekey]

        tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

        if(corrkey!='OTOC'):
            OTOCarr/=nbeads
        #plt.plot(tarr,OTOCarr)
        plt.plot(tarr,np.log(abs(OTOCarr)))
        plt.show()
        store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt,Ekey),rpext)

    if(0): ##Histograms of thermal ensembles
        kwqlist = ['Thermalized_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        kwplist = ['Thermalized_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]

        kwqlist = ['Const_q_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        kwplist = ['Const_q_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        
        fqlist = seed_finder(kwqlist,rpext,dropext=True)
        fplist = seed_finder(kwplist,rpext,dropext=True)

        E=[]
        V=[]
        K=[]
        Q=[]

        for qfile,pfile in zip(fqlist,fplist):
            qcart = read_arr(qfile,rpext)
            pcart = read_arr(pfile,rpext)

            fft = FFT(1,nbeads)
            q = fft.cart2mats(qcart)
            p = fft.cart2mats(pcart)

            #print('qfile,pfile', qfile,pfile)
            omegan = nbeads/beta
            potsys = np.sum(pes.potential(qcart),axis=1)
            potspr = np.sum(0.5*m*omegan**2*(qcart-np.roll(qcart,1,axis=-1))**2,axis=2)[:,0]
            pot = potsys+potspr
            kin = np.sum(np.sum(pcart**2/(2*m),axis=1),axis=1)
            #pot = pes.potential(q[:,0,0]/nbeads**0.5)
            kin = p[:,0,0]**2/(2*m)

            Etot = pot+kin
            E.extend(Etot)
            K.extend(kin)
            V.extend(pot)
            Q.extend(q[:,0,1])

        E=np.array(E)
        V=np.array(V)
        K=np.array(K)
        E/=nbeads
        V/=nbeads
        K/=nbeads

        #plt.hist(Q,bins=100)
        #plt.show()
        #exit()

        bins = np.linspace(0.0,0.25,200)
        dE = bins[1]-bins[0]
        #countsV, bin_edgeV = np.histogram(V,bins=200)
        #countsK, bin_edgeK = np.histogram(K,bins=200)

        #print('counts V', countsV[:80], bin_edgeV[80])
        #print('counts K', countsK[:80], bin_edgeK[80])

        #Ehist = plt.hist(x=E, bins=bins,density=True,color='r')
        #Vhist = plt.hist(x=V, bins=bins,density=True,color='g',alpha=0.5)
        Khist = plt.hist(x=K, bins=bins,density=True,color='b',alpha=0.5)

        plt.axvline(x=T/2,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)
        #plt.axvline(x=nbeads*T/2,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)
        #plt.axvline(x=nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)
        #plt.axvline(x=V.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='g')
        plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')
        #plt.axvline(x=E.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='r')
        #plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='m')
        plt.show()
        exit()

    if(0): ##Histograms of microcanonical ensembles
        kwqlist = ['Microcanonical_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        kwplist = ['Microcanonical_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        
        kwqlist = ['Const_q_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        kwplist = ['Const_q_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]

        fqlist = seed_finder(kwqlist,rpext,dropext=True)
        fplist = seed_finder(kwplist,rpext,dropext=True)

        RG = []
        bins = np.linspace(0.0,5,200)
        for qfile,pfile in zip(fqlist,fplist):
            qcart = read_arr(qfile,rpext)
            pcart = read_arr(pfile,rpext)

            fft = FFT(1,nbeads)
            q = (fft.cart2mats(qcart)[:,0,0])/nbeads**0.5
            #p = fft.cart2mats(pcart)
            rg = (np.mean((qcart[:,0,:]-q[:,None])**2,axis=1))**0.5
            RG.extend(rg)

        RGhist = plt.hist(x=RG, bins=bins,density=True)
        plt.show()
        exit()

    if(0):#RPMD static averages
        methodkey = 'RPMD'
        enskey = 'thermal'
        corrkey = 'stat_avg'
        sigmakey = 'sigma_0.21'

        kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey,sigmakey]

        fname  = seed_finder(kwlist,rpext)
        fname = rpext + fname[0]
        print('fname',fname)

        data = np.loadtxt(fname)[:,1]

        countarr = []
        data_arr = []
        count = 0
        statavg = 0.0
        for i in range(len(data)):
            statavg+=data[i]
            count+=1
            countarr.append(count)
            data_arr.append(statavg/count)

        print('statavg,count',statavg/count,count)
        plt.plot(countarr,data_arr)
        plt.show()

    if(0):
        methodkey = 'RPMD'
        enskey = 'thermal'
        corrkey = 'R2'
        suffix = '_asym'
        kwlist = [methodkey,corrkey,syskey,potkey,Tkey+'_',beadkey,'dt_{}'.format(dt),suffix]#,'qqq']

        X,Y,F = seed_collector_imagedata(kwlist,rpext)#,allseeds=False,seedcount=20)
        X[:,len(X)//2+1:] = X[:,:-len(X)//2:-1]
        Y[len(Y)//2+1:,:] = Y[:-len(Y)//2:-1,:]
        F[:,len(X)//2+1:] = F[:,:-len(X)//2:-1]
        F[len(Y)//2+1:,:] = F[:-len(Y)//2:-1,:]

        X=np.roll(X,len(X)//2,axis=1)
        Y=np.roll(Y,len(Y)//2,axis=0)
        F=np.roll(np.roll(F,len(X)//2,axis=1), len(Y)//2, axis=0)

        #print('Y', Y.shape, F.shape,X)
        #F/=nbeads
        #plt.scatter(0,0,c='r')
        #print('length', X.shape)
        #print(Y[:,300+30],X[:,330])
        #plt.title(r'$\beta={}, N_b={}$'.format(1/T,nbeads))
        #plt.plot(Y[:,330],F[:,330])
        #plt.xlabel('t')
        #plt.ylabel(r'$K_{xxx}^{sym}(t,t\'=3)$')
        fig, ax = plt.subplots()
        pos = ax.imshow(F.T,extent=[X[0].min(),X[0].max(0),Y[:,0].min(),Y[:,0].max()],origin='lower',cmap='bwr')#,vmin=-10,vmax=10)
        #ax.scatter(X.flatten(), Y.flatten(), c=(F.T).flatten())
        #ax.set_xlim([-20,20])
        #ax.set_ylim([-20,20])
        fig.set_size_inches(12, 6)
        fig.colorbar(pos,ax=ax)
        plt.show()
        #potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,np.around(b,2))

        store_2D_imagedata_column(X,Y,F,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt,suffix),rpext,extcol=np.zeros_like(X))

if(1): ##Classical
    sigma = 10.0
    q0 = 0.0
    #potkey = 'FILT_{}_g_{}_sigma_{}_q0_{}'.format(lamda,g,sigma,q0)

    if(1):
        methodkey = 'Classical'
        enskey = 'thermal'#'stable_manifold' # 'const_q'#
        corrkey = 'OTOC_qq'#'qq_TCF'#'singcomm'#'OTOC'

        #E = 4.09#3.125#2.125#
        kwlist = [enskey+'_',methodkey,corrkey,syskey,'cmd_pmf',Tkey]#,'dt_{}'.format(dt)]#,'E_{}_'.format(E)]

        tarr,OTOCarr,stdarr = seed_collector(kwlist,Cext,tarr,OTOCarr,allseeds=False,seedcount=500)

        store_1D_plotdata(tarr,OTOCarr,'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt),Cext)

        ext = Cext + 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt)
        td = 3.5
        tu = 4.5
        slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
        plt.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

        plt.plot(tarr,np.log(abs(OTOCarr)))
        plt.show()


    if(0): # Energy_histogram
        kwqlist = ['Thermalized_rp_qcart', 'beta_{}'.format(beta), potkey]
        kwplist = ['Thermalized_rp_pcart', 'beta_{}'.format(beta), potkey]

        fqlist = seed_finder(kwqlist,Cext,dropext=True)
        fplist = seed_finder(kwplist,Cext,dropext=True)

        E=[]
        V=[]
        K=[]
        for qfile,pfile in zip(fqlist,fplist):
            qcart = read_arr(qfile,Cext)[:,0,0]
            pcart = read_arr(pfile,Cext)[:,0,0]
            #print('qfile,pfile', qfile,pfile)

            pot=pes.potential(qcart)
            kin=pcart**2/(2*m)
            Etot = pot+kin
            E.extend(Etot)
            V.extend(pot)
            K.extend(kin)

        plt.hist(x=E, bins=50,density=True,color='r')
        #plt.hist(x=V, bins=50,color='g')
        #plt.hist(x=K, bins=50,color='b',alpha=0.25)
        plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
        plt.show()

