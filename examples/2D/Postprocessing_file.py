import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.nmtrans import FFT


dim=2

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

z = 0.5
 
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
beta=1/T

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 100.0
time_total = 5.0

nbeads = 8

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
pes = quartic_bistable(alpha,D,lamda,g,z)

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

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

if(1):#RPMD
	if(1):
		methodkey = 'RPMD'
		enskey='thermal'

		kwlist = [methodkey,enskey,corrkey,syskey,potkey,Tkey,beadkey]
		
		tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

		print('stdarr', stdarr[2499])

		if(corrkey!='OTOC'):
			OTOCarr/=nbeads
		#plt.plot(tarr,OTOCarr)
		plt.plot(tarr,np.log(abs(OTOCarr)))
		plt.errorbar(tarr,np.log(abs(OTOCarr)),yerr=stdarr/2,ecolor='m',errorevery=100,capsize=2.0)
		plt.show()
		store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext,ebar=stdarr)

	if(0): # Energy_histogram
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
			#print('qfile,pfile', qfile,pfile)

			fft = FFT(1,nbeads)
			q = fft.cart2mats(qcart)
			p = fft.cart2mats(pcart)
		
			#print('qfile,pfile', qfile,pfile)
			omegan = nbeads/beta
			potsys = np.sum(pes.potential(qcart)+0.6655,axis=1)
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
		
		bins = np.linspace(0.0,8.0,200)
		dE = bins[1]-bins[0]
		
		Ehist = plt.hist(x=E, bins=bins,density=True,color='r')
		Vhist = plt.hist(x=V, bins=bins,density=True,color='g',alpha=0.5)
		Khist = plt.hist(x=K, bins=bins,density=True,color='b',alpha=0.5)
		
		plt.axvline(x=nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)	
		plt.axvline(x=2*nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)		
		plt.axvline(x=V.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='g')
		plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')		
		plt.axvline(x=E.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='r')			
		plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='m')
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

if(0):#RPMD/mc
	methodkey = 'RPMD'
	enskey  = 'mc'
	kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey,beadkey]
	
	tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

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

if(0):#Classical
	if(1):
		methodkey = 'Classical'
		enskey = 'thermal'#'mc'

		kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey]
		
		tarr,OTOCarr,stdarr = seed_collector(kwlist,Cext,tarr,OTOCarr)

		plt.plot(tarr,np.log(abs(OTOCarr)))
		plt.show()
		store_1D_plotdata(tarr,OTOCarr,'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt),Cext)


	if(0): 
		#kwqlist = ['Thermalized_rp_qcart', 'beta_{}'.format(beta), potkey]
		#kwplist = ['Thermalized_rp_pcart', 'beta_{}'.format(beta), potkey]
		
		kwqlist = ['Microcanonical_rp_qcart', 'beta_{}'.format(beta), potkey]
		kwplist = ['Microcanonical_rp_pcart', 'beta_{}'.format(beta), potkey]
		
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

			plt.scatter(xarr,pxarr)
			plt.show()
			plt.scatter(yarr,pyarr)
			plt.show()		

		if(0):	# Energy_histogram
			E=[]
			V=[]
			K=[]
			for qfile,pfile in zip(fqlist,fplist):
				qcart = read_arr(qfile,Cext)[:,:,0]
				pcart = read_arr(pfile,Cext)[:,:,0]		
				#print('qfile,pfile', qfile,pfile)
		
				pot = pes.potential(qcart)
				kin = np.sum(pcart**2/(2*m),axis=1)
				Etot = pot+kin
				E.extend(pot+kin)
				V.extend(pot)
				K.extend(kin)

			plt.hist(x=E, bins=100,color='r')
			plt.hist(x=V, bins=50,color='g',alpha=0.5)
			plt.hist(x=K, bins=50,color='b',alpha=0.5)
			plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
			plt.axvline(x=2*m/beta,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
			plt.show()
