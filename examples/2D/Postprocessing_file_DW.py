import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,estimate_OTOC_slope,seed_collector,seed_finder
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.potentials import quartic_bistable, Harmonic_oblique#, DW_harm
from PISC.utils.nmtrans import FFT
import scipy
from argparse import ArgumentParser

dim=2
m = 0.5

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

Tc = 0.5*lamda/np.pi

N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 50.0
time_total = 5.0


#Path extensions
path = os.path.dirname(os.path.abspath(__file__))   
path = '/scratch/vgs23/PISC/examples/2D/'
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'fd_OTOC'#'qq_TCF'#'OTOC'#
syskey = 'Papageno'

def main(z,enskey,pot,times,nbeads):
    T = times*Tc
    beta=1/T
    Tkey = 'T_{}Tc'.format(times)
    
    if pot=='dw_qb':# Double well
        alpha = 0.382
        D = 3*Vb

        potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        pes = quartic_bistable(alpha,D,lamda,g,z)

    elif pot=='dw_harm':# Double well with harmonic coupling
        #w = 2.0
        #potkey = 'DW_harm_2D_T_{}Tc_m_{}_w_{}_lamda_{}_g_{}_z_{}'.format(times,m,w,lamda,g,z)
        #pes = DW_harm(m, w, lamda, g, z)

        alpha = 0.382
        D = 3*Vb
        #pes = DW_Morse_harm(alpha,D,lamda,g,z)
        potkey = 'DW_Morse_harm_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        print(potkey)

    beadkey = 'nbeads_{}_'.format(nbeads)
    potkey_ = potkey+'_'

    methodkey = 'RPMD'

    kwlist = [methodkey,enskey+'_',corrkey+'_',syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]#,'1D_const']

    if('OTOC' or 'TCF' in corrkey):
        tarr = np.arange(0.0,time_total,dt)
        OTOCarr = np.zeros_like(tarr) +0j
        tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr)#,seedcount=3000,allseeds=False)
        #plt.plot(tarr,np.log(abs(OTOCarr)),color='r')
        
        store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext,ebar=stdarr)

    elif('stat_avg' in corrkey):
        hesstype='centroid_Hessian'
        kwlist.append(hesstype,'N_{}_'.format(N))
        seedarr, statarr = seed_collector(kwlist,rpext)
        store_1D_plotdata(seedarr,statarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt,hesstype),rpext)

    if(0): # Energy_histogram
        if (enskey=='thermal'):
            print('thermal')
            kwqlist = ['Thermalized_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
            kwplist = ['Thermalized_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        else:
            print('const_q')
            kwqlist = ['Const_q_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
            kwplist = ['Const_q_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        
        fqlist = seed_finder(kwqlist,rpext,dropext=True)
        fplist = seed_finder(kwplist,rpext,dropext=True)
    
        E=[]
        V=[]
        K=[]

        xarr = []
        yarr = [] 
        for qfile,pfile in zip(fqlist,fplist):
            qcart = read_arr(qfile,rpext)
            pcart = read_arr(pfile,rpext)       
            #print('qfile,pfile', qfile,pfile)

            fft = FFT(1,nbeads)
            q = fft.cart2mats(qcart)
            p = fft.cart2mats(pcart)

            x = q[:,0,0]
            y = q[:,1,0]
            xarr.append(x)
            yarr.append(y)      
            #print('qfile,pfile', qfile,pfile)
            omegan = nbeads/beta
            potsys = np.sum(pes.potential(qcart),axis=1)
            potspr = np.sum(np.sum(0.5*m*omegan**2*(qcart-np.roll(qcart,1,axis=-1))**2,axis=2),axis=1)
            pot = potsys+potspr
            kin = np.sum(np.sum(pcart**2/(2*m),axis=1),axis=1)  
            #kin = np.sum(p[...,15]**2/(2*m),axis=1)
            #pot = pes.potential(q[:,0,0]/nbeads**0.5)
            #kin = p[:,0,0]**2/(2*m*nbeads**0.5)
    
            Etot = pot+kin
            E.extend(Etot)
            K.extend(kin)
            V.extend(pot)
                                
        E=np.array(E)
        V=np.array(V)
        K=np.array(K)
        E/=nbeads
        V/=nbeads
        K/=nbeads

        #plt.scatter(xarr,yarr)
        #plt.show()  
    
        bins = np.linspace(0.0,10.0,200)
        dE = bins[1]-bins[0]
        
        #Ehist = plt.hist(x=E, bins=bins,density=True,color='r')
        Vhist = plt.hist(x=V, bins=bins,density=True,color='g',alpha=0.5)
        Khist = plt.hist(x=K, bins=bins,density=True,color='b',alpha=0.5)
       
        #plt.axvline(x=T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=1)
        plt.axvline(x=nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)   
        plt.axvline(x=2*nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)     
        #plt.axvline(x=V.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='g')
        plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')       
        #plt.axvline(x=E.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='r')           
        #plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='m')
        plt.show()
            
    #plt.legend()
    #plt.show()

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-z', type=float, default=0.0)
    argparser.add_argument('--enskey','-e', type=str, default='thermal')
    argparser.add_argument('--pot','-p', type=str, default='dw_qb')
    argparser.add_argument('--times','-t', type=float, default=3.0)
    argparser.add_argument('--nbeads','-nb', type=int, default=1)
    args = argparser.parse_args()
    main(args.z,args.enskey,args.pot,args.times,args.nbeads)

