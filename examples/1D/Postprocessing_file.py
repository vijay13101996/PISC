import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.potentials import double_well, quartic, morse, mildly_anharmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.nmtrans import FFT

dim = 1

if(0): #Double well potential
	lamda = 2.0
	g = 0.02#8
	Vb = lamda**4/(64*g)

	Tc = lamda*(0.5/np.pi)
	times = 20.0#0.6
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

if(0): #Quartic
	a = 1.0

	pes = quartic(a)

	T = 1.0
	
	m = 1.0
	N = 1000
	dt_therm = 0.05
	dt = 0.05

	time_therm = 50.0
	time_total = 5.0

	potkey = 'quartic_a_{}'.format(a)
	Tkey = 'T_{}'.format(T)

if(1): #Mildly anharmonic
	omega = 1.0
	a = 1.0/10
	b = 1.0/100
	
	T = 1.0#times*Tc
	
	m=1.0
	N = 1000
	dt_therm = 0.05
	dt = 0.01
	
	time_therm = 50.0
	time_total = 30.0

	pes = mildly_anharmonic(m,a,b)
	
	potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)
	Tkey = 'T_{}'.format(T)

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
path = '/scratch/vgs23/PISC/examples/1D'#os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'pq_TCF'#'singcomm' #
syskey = 'Selene'

if(1):#RPMD
	nbeads = 4
	beadkey = 'nbeads_{}_'.format(nbeads)
	if(0): ##Collect files of thermal ensembles
		methodkey = 'RPMD'
		enskey = 'thermal'

		kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]
		
		tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

		if(corrkey!='OTOC'):
			OTOCarr/=nbeads
			plt.plot(tarr,OTOCarr)
		else:
			plt.plot(tarr,np.log(abs(OTOCarr)))
		plt.show()
		store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext)

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
		
		bins = np.linspace(0.0,10.0,200)
		dE = bins[1]-bins[0]
		#countsV, bin_edgeV = np.histogram(V,bins=200)
		#countsK, bin_edgeK = np.histogram(K,bins=200)

		#print('counts V', countsV[:80], bin_edgeV[80])
		#print('counts K', countsK[:80], bin_edgeK[80])

		#Ehist = plt.hist(x=E, bins=bins,density=True,color='r')
		Vhist = plt.hist(x=V, bins=bins,density=True,color='g',alpha=0.5)
		Khist = plt.hist(x=K, bins=bins,density=True,color='b',alpha=0.5)
		
		plt.axvline(x=nbeads*T/2,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)	
		plt.axvline(x=nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)		
		plt.axvline(x=V.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='g')
		plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')		
		plt.axvline(x=E.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='r')			
		#plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='m')
		plt.show()

	if(0): ##Histograms of microcanonical ensembles
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

	if(1):
		methodkey = 'RPMD'
		enskey = 'thermal'
		corrkey = 'R2'
		suffix = '_sym'
		kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt),suffix]
		
		X,Y,F = seed_collector_imagedata(kwlist,rpext,tarr)
		#F/=nbeads

		#print(Y[:,30],X[:,30])
		plt.title(r'$\beta={}, N_b={}$'.format(1/T,nbeads))
		plt.plot(Y[:,30],F[:,30])
		plt.xlabel('t')
		plt.ylabel(r'$K_{xxx}^{sym}(t,t\'=3)$')
		#plt.imshow(F)
		plt.show()
		#store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext)

if(0): ##Classical
	sigma = 10.0
	q0 = 0.0
	#potkey = 'FILT_{}_g_{}_sigma_{}_q0_{}'.format(lamda,g,sigma,q0)
	
	if(1):
		methodkey = 'Classical'
		enskey = 'thermal'
		corrkey = 'OTOC'#'qq_TCF'#'singcomm'#'OTOC'
	
		#E = 4.09#3.125#2.125#
		kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey]#,'E_{}_'.format(E)]
		
		tarr,OTOCarr,stdarr = seed_collector(kwlist,Cext,tarr,OTOCarr)

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


#--------------------------------------------------------------------------------------------------------

if(0):
	methodkey = 'RPMD'
	enskey = 'thermal'
	corrkey = 'qq_TCF'
	potkey='quartic_a_1.0'
	dt = 0.01
	Tkey='T_0.125'
	
	kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey,'dt_{}'.format(dt)]
	
	tarr = np.arange(0.0,time_total,dt)
	OTOCarr = np.zeros_like(tarr) +0j

	for sc in [1,3,5,10,13,16,20]:
		tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr,allseeds=False,seedcount=sc,logerr=False)
		print('std,C0',stdarr[0],OTOCarr[0]/nbeads)

	if(corrkey!='OTOC'):
		OTOCarr/=nbeads
	#plt.plot(tarr,OTOCarr)
	plt.plot(tarr,OTOCarr)
	plt.errorbar(tarr,OTOCarr,yerr=stdarr/2,ecolor='m',errorevery=100,capsize=2.0)	
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


