import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,estimate_OTOC_slope,seed_collector,seed_finder
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.potentials import quartic_bistable, Harmonic_oblique
from PISC.utils.nmtrans import FFT
import scipy

dim=2

### Double well
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

z = 1.0#0.5
 
Tc = 0.5*lamda/np.pi
times = 3.0#0.95
T = times*Tc
beta=1/T
Tkey = 'T_{}Tc'.format(times)

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.005
time_therm = 50.0
time_total = 5.0#5.0

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
pes = quartic_bistable(alpha,D,lamda,g,z)

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))   
#path = '/scratch/vgs23/PISC/examples/2D/'
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'fd_OTOC'#'qq_TCF'#'OTOC'#
syskey = 'Papageno'

nbeads=1
beadkey = 'nbeads_{}_'.format(nbeads)
potkey_ = potkey+'_'

methodkey = 'Classical'
enskey= 'thermal'#'const_q'#'thermal'

filt = False
if(filt):
    filtkey = 'filt'
else:
    filtkey = 'nofilt'

if(enskey=='const_q'):
    filtkey = 'const_q'
    filt = False

if(1):
    for z in [1.0]:#,1.0]:#0.0,1.0,2.0]:#0.5,1.0,1.5,2.0]:
        #potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        
        potkey = 'DW_harm_2D_m_{}_w_{}_lamda_{}_g_{}_z_{}'.format(m,2.0,lamda,g,z)
        
        if(1):
            kwlist = [methodkey,enskey+'_',corrkey+'_',syskey,potkey,Tkey,'dt_{}'.format(dt),'_'+filtkey]
               
            if(1):
                tarr,OTOCarr,stdarr = seed_collector(kwlist,Cext,tarr,OTOCarr)#,seedcount=2000,allseeds=False)
                plt.plot(tarr,np.log(abs(OTOCarr)),color='r')
                
                store_1D_plotdata(tarr,OTOCarr,'Classical_thermal_{}_{}_{}_dt_{}_{}'.format(corrkey,potkey,Tkey,dt,filtkey),Cext,ebar=stdarr)

            ext = Cext + 'Classical_thermal_{}_{}_{}_dt_{}_{}'.format('fd_OTOC',potkey,Tkey,dt,filtkey)
            slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.2,3.2)#1.5,2.5)#1.5,2.5)
            plt.plot(t_trunc, slope*t_trunc+ic,color='k',lw=1.5)
            data = np.loadtxt(ext+'.txt',dtype=complex)
            plt.plot(data[:,0],np.log(data[:,1]),label='z={}, $\lambda$={}'.format(z,np.around(slope,2)),lw=1)
        
        if(0): # Energy_histogramz
            if (enskey=='thermal'):
                print('thermal')
                kwqlist = ['Thermalized_rp_qcart','N_{}'.format(N),'nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
                kwplist = ['Thermalized_rp_pcart','N_{}'.format(N),'nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
            else:
                print('const_q')
                kwqlist = ['Const_q_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
                kwplist = ['Const_q_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
            
            fqlist = seed_finder(kwqlist,Cext,dropext=True)
            fplist = seed_finder(kwplist,Cext,dropext=True)
        
            E=[]
            V=[]
            K=[]

            xarr = []
            yarr = [] 
            for qfile,pfile in zip(fqlist,fplist):
                qcart = read_arr(qfile,Cext)
                pcart = read_arr(pfile,Cext)       
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
    plt.show()    

for filtkey in ['filt','nofilt','const_q']:
        print(filtkey)
        ext = Cext + 'Classical_thermal_{}_{}_{}_dt_{}_{}'.format('fd_OTOC',potkey,Tkey,dt,filtkey)
        slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.,3.)#1.5,2.5)#1.5,2.5)
        plt.plot(t_trunc, slope*t_trunc+ic,color='k',lw=1.5)
        data = np.loadtxt(ext+'.txt',dtype=complex)
        plt.plot(data[:,0],np.log(data[:,1]),label='{}'.format(filtkey),lw=1)

plt.legend()
plt.show()
