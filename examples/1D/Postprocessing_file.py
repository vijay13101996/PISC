import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder
from matplotlib import pyplot as plt
import os
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.harmonic_1D import harmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.nmtrans import FFT

dim = 1
lamda = 2.0#
g = 0.08#
Vb = lamda**4/(64*g)

Tc = lamda*(0.5/np.pi)
times = 1.0
T = times*Tc
beta=1/T

m = 0.5
N = 1000
dt = 0.002

nbeads = 16
gamma = 16

time_total = 5.0#

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
pes = double_well(lamda,g)

#potkey='harmonic'
#pes = harmonic(m,1.0)

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'OTOC'#'qq_TCF'#
beadkey = 'nbeads_{}_'.format(nbeads)
Tkey = 'T_{}Tc'.format(times)
syskey = 'Selene'

if(0):#RPMD
	if(1):
		methodkey = 'RPMD'
		enskey = 'thermal'

		kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]
		
		tarr,OTOCarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

		if(corrkey!='OTOC'):
			OTOCarr/=nbeads
		#plt.plot(tarr,OTOCarr)
		plt.plot(tarr,np.log(abs(OTOCarr)))
		plt.show()
		store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext)

	if(0):
		kwqlist = ['Thermalized_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
		kwplist = ['Thermalized_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
		
		fqlist = seed_finder(kwqlist,rpext,dropext=True)
		fplist = seed_finder(kwplist,rpext,dropext=True)
	
		E=[]
		V=[]
		K=[]
				
		for qfile,pfile in zip(fqlist,fplist):
			qcart = read_arr(qfile,rpext)
			pcart = read_arr(pfile,rpext)
	
			fft = FFT(1,nbeads)
			q = fft.cart2mats(qcart)
			p = fft.cart2mats(pcart)
		
			#print('qfile,pfile', qfile,pfile)
			omegan = nbeads/beta
			potsys = np.sum(pes.potential(qcart)[:,0],axis=1)
			potspr = np.sum(0.5*m*omegan**2*(qcart-np.roll(qcart,1,axis=-1))**2,axis=2)[:,0]
			pot = potsys+potspr
			kin = np.sum(np.sum(pcart**2/(2*m),axis=1),axis=1)	
						
			#pot = pes.potential(q[:,0,0]/nbeads**0.5)
			#kin = p[:,0,0]**2/(2*m*nbeads)
	
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
		
		bins = np.linspace(0.0,8.0,200)
		dE = bins[1]-bins[0]
		#countsV, bin_edgeV = np.histogram(V,bins=200)
		#countsK, bin_edgeK = np.histogram(K,bins=200)

		#print('counts V', countsV[:80], bin_edgeV[80])
		#print('counts K', countsK[:80], bin_edgeK[80])

		Ehist = plt.hist(x=E, bins=bins,density=True,color='r')
		Vhist = plt.hist(x=V, bins=bins,density=True,color='g',alpha=0.5)
		Khist = plt.hist(x=K, bins=bins,density=True,color='b',alpha=0.5)
		
		plt.axvline(x=nbeads*T/2,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)	
		plt.axvline(x=nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)		
		plt.axvline(x=V.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='g')
		plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')		
		plt.axvline(x=E.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='r')			
		#plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='m')
		plt.show()

	if(0):
		kwqlist = ['Microcanonical_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
		kwplist = ['Microcanonical_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
		
		fqlist = seed_finder(kwqlist,rpext,dropext=True)
		fplist = seed_finder(kwplist,rpext,dropext=True)
	
		RG = []
		bins = np.linspace(0.0,1.5,200)	
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

if(0):#RPMD/mc
	methodkey = 'RPMD'
	enskey  = 'mc'
	kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey,beadkey]
	
	tarr,OTOCarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

	if(corrkey!='OTOC'):
		OTOCarr/=nbeads
	#plt.plot(tarr,OTOCarr)
	plt.plot(tarr,np.log(abs(OTOCarr)))
	plt.show()
	store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext)

if(0):#CMD
	methodkey = 'CMD'
	gammakey = 'gamma_{}'.format(gamma)

	kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey,gammakey]
	
	tarr,OTOCarr = seed_collector(kwlist,cext,OTOCarr)

	plt.plot(tarr,np.log(abs(OTOCarr)))
	plt.show()
	store_1D_plotdata(tarr,OTOCarr,'CMD_{}_{}_{}_nbeads_{}_dt_{}_gamma_{}'.format(corrkey,potkey,Tkey,nbeads,dt,gamma),cext)

if(1):#Classical
	if(1):
		methodkey = 'Classical'
		enskey = 'mc'#'thermal'

		kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey]
		
		tarr,OTOCarr = seed_collector(kwlist,Cext,tarr,OTOCarr)

		plt.plot(tarr,np.log(abs(OTOCarr)))
		plt.show()
		store_1D_plotdata(tarr,OTOCarr,'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt),Cext)

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
