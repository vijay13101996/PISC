import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,estimate_OTOC_slope,seed_collector,seed_finder
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.potentials import coupled_quartic
from PISC.utils.nmtrans import FFT
import scipy
import argparse

dim=2
m=1.0

g1 = 10
g2 = 0.1
pes = coupled_quartic(g1,g2)
potkey = 'CQ_g1_{}_g2_{}'.format(g1,g2)

N = 1000
dt_therm = 0.05
time_therm = 50.0
time_total = 5.0

path = os.path.dirname(os.path.abspath(__file__))

methodkey = 'RPMD'
syskey = 'Papageno'      
corrkey = 'fd_OTOC'
enskey = 'thermal'

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
#path = '/scratch/vgs23/PISC/examples/2D/'
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)


def main(beta,nbeads,dt):
    print('beta,nbeads,dt',beta,nbeads,dt)
    T = 1.0/beta
    Tkey = 'beta_{}'.format(beta)

    tarr = np.arange(0.0,time_total,dt)
    OTOCarr = np.zeros_like(tarr) +0j

    beadkey = 'nbeads_{}_'.format(nbeads)
    potkey_ = potkey + '_'

    kwlist = [methodkey,enskey+'_',corrkey,syskey,potkey_,Tkey,beadkey,'dt_{}'.format(dt)]

    tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr)
    #plt.plot(tarr,np.log(abs(OTOCarr)))
    store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext,ebar=stdarr)
    #plt.show()

    if(0): # Energy_histogram
        kwqlist = ['Thermalized_rp_qcart','nbeads_{}_'.format(nbeads), 'beta_{}_'.format(beta), potkey]
        kwplist = ['Thermalized_rp_pcart','nbeads_{}_'.format(nbeads), 'beta_{}_'.format(beta), potkey]
        
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

        bins = np.linspace(0.0,40.0,200)
        dE = bins[1]-bins[0]
        
        #Ehist = plt.hist(x=E, bins=bins,density=True,color='r')
        Vhist = plt.hist(x=V, bins=bins,density=True,color='g',alpha=0.5)
        Khist = plt.hist(x=K, bins=bins,density=True,color='b',alpha=0.5)
        
        plt.axvline(x=nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)	
        plt.axvline(x=2*nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)		
        #plt.axvline(x=V.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='g')
        #plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')		
        #plt.axvline(x=E.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='r')			
        #plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='m')
        plt.show()

    if(0): #Radius of gyration histogram 
        kwqlist = ['Microcanonical_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        kwplist = ['Microcanonical_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
        
        fqlist = seed_finder(kwqlist,rpext,dropext=True)
        fplist = seed_finder(kwplist,rpext,dropext=True)

        RG = []
        bins = np.linspace(0.0,0.8,200)	
        for qfile,pfile in zip(fqlist,fplist):
            qcart = read_arr(qfile,rpext)
            pcart = read_arr(pfile,rpext)

            fft = FFT(dim,nbeads)
            q = (fft.cart2mats(qcart)[...,0])/nbeads**0.5
            #p = fft.cart2mats(pcart)
            rg = np.mean(np.sum((qcart-q[:,:,None])**2,axis=1),axis=1)
            RG.extend(rg)

        RGhist = plt.hist(x=RG, bins=bins,density=True)
        plt.show()


    if(0): #Histograms 
        kwqlist = ['Thermalized_rp_qcart', 'beta_{}'.format(beta), potkey]
        kwplist = ['Thermalized_rp_pcart', 'beta_{}'.format(beta), potkey]
        
        #kwqlist = ['Microcanonical_rp_qcart', 'beta_{}'.format(beta), potkey]
        #kwplist = ['Microcanonical_rp_pcart', 'beta_{}'.format(beta), potkey]
        
        fqlist = seed_finder(kwqlist,Cext,dropext=True)
        fplist = seed_finder(kwplist,Cext,dropext=True)

        if(1):
            xarr = []
            pxarr = []
            yarr = [] 
            pyarr = []
            for qfile,pfile in zip(fqlist,fplist):
                qcart = read_arr(qfile,Cext)[:,:,0]
                pcart = read_arr(pfile,Cext)[:,:,0]		
            
                x = qcart[:,0]
                y = qcart[:,1]
                px = pcart[:,0]
                py = pcart[:,1]	

                xarr.extend(x)
                yarr.extend(y)
                pxarr.extend(px)
                pyarr.extend(py)

            plt.scatter(xarr,yarr,s=2)
            plt.show()
            #plt.scatter(xarr,pxarr)
            #plt.show()
            #plt.scatter(yarr,pyarr)
            #plt.show()		

        if(1):	# Energy_histogram
            E=[]
            V=[]
            K=[]
            for qfile,pfile in zip(fqlist,fplist):
                qcart = read_arr(qfile,Cext)[:,:,0]
                pcart = read_arr(pfile,Cext)[:,:,0]		
                #print('qfile,pfile', qfile,pfile)
        
                pot = pes.potential_xy(qcart[:,0],qcart[:,1])
                kin = np.sum(pcart**2/(2*m),axis=1)
                Etot = pot+kin
                E.extend(pot+kin)
                V.extend(pot)
                K.extend(kin)

            K =np.array(K)
            #plt.hist(x=E, bins=100,color='r')
            plt.hist(x=V, bins=100,color='g',alpha=0.5)
            plt.hist(x=K, bins=100,color='b',alpha=0.5)
            #plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
            plt.axvline(x=2*m/beta,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
            plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')			
            plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--beta',type=float,default=1.0)
    argparser.add_argument('--nbeads',type=int,default=1)
    argparser.add_argument('--dt',type=float,default=0.002)
    args = argparser.parse_args()
    main(**vars(args))


