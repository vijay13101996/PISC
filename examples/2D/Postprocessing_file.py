import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,estimate_OTOC_slope,seed_collector,seed_finder
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.potentials import quartic_bistable, Harmonic_oblique, Tanimura_SB
from PISC.utils.nmtrans import FFT
import scipy

dim=2

if(0): ### Double well
	lamda = 2.0
	g = 0.08
	Vb = lamda**4/(64*g)

	alpha = 0.382
	D = 3*Vb

	z = 1.0#0.5
	 
	Tc = 0.5*lamda/np.pi
	times = 2.0
	T = times*Tc
	beta=1/T
	Tkey = 'T_{}Tc'.format(times)

	m = 0.5
	N = 1000
	dt_therm = 0.05
	dt = 0.002#05
	time_therm = 50.0
	time_total = 5.0#5.0

	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	pes = quartic_bistable(alpha,D,lamda,g,z)

	#pos = np.zeros((1,2,1))
	#pos[0,:,0] = [1,1]
	#print('PES', pes.potential_xy(1,1))
	#print('grad', pes.dpotential(pos))
	#print('Hess',pes.ddpotential(pos))

if(1): #Tanimura System-Bath
	dim=2
	# Tanimura's system-bath potential
	m = 1.0
	D = 0.0234 
	alpha = 0.00857
	m = 1.0
	mb = 1.0
	wb = 0.0072901 #(= w_10)
	
	m = 1.0
	mb= 1.0
	delta_anh = 0.1
	w_10 = 1.0
	wb = w_10
	wc = w_10 + delta_anh
	alpha = (m*delta_anh)**0.5
	D = m*wc**2/(2*alpha**2)

	VLL = -0.75*wb#0.05*wb
	VSL = 0.75*wb#0.05*wb
	cb = 0.75#0.65*wb#0.75*wb#0.05*wb#0.05*wb

	pes = Tanimura_SB(D,alpha,m,mb,wb,VLL,VSL,cb)
				
	TinK = 300
	K2au = 3.1667e-6
	T = 0.125#TinK*K2au
	beta = 1/T

	potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,cb)
	Tkey = 'T_{}'.format(T)

	N = 1000
	dt_therm = 10.0
	dt = 0.01#2.0
	time_therm = 50.0
	time_total = 100.0#20000.0

if(0):
	m = 1.0
	omega1 = 1.0
	omega2 = 1.0
	trans = np.array([[1.0,0.2],[0.2,1.0]])
	
	pes	= Harmonic_oblique(trans,m,omega1,omega2)	

	T = 1.0
	beta=1/T
	
	N = 1000
	dt_therm = 0.05
	dt = 0.02
	time_therm = 50.0
	time_total = 20.0
	nbeads = 1

	potkey = 'TESTharmonicObl'
	Tkey = 'T_{}'.format(T)

tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
path = '/scratch/vgs23/PISC/examples/2D/'
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'qq_TCF'#'OTOC'#
syskey = 'Selene'

if(1):#RPMD
	nbeads=1
	beadkey = 'nbeads_{}_'.format(nbeads)
	potkey_ = potkey+'_'
	if(0):
		methodkey = 'RPMD'
		enskey='thermal'

		kwlist = [methodkey,enskey,corrkey,syskey,potkey_,Tkey,beadkey,'dt_{}'.format(dt)]
		
		tarr,OTOCarr,stdarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

		#print('stdarr', stdarr[2499])

		if(corrkey!='OTOC'):
			OTOCarr/=nbeads
			plt.plot(tarr,OTOCarr)
		else:
			plt.plot(tarr,np.log(abs(OTOCarr)))
		#plt.errorbar(tarr,np.log(abs(OTOCarr)),yerr=stdarr/2,ecolor='m',errorevery=100,capsize=2.0)
		plt.show()
	
		if(0):
			print('tarr', np.real(tarr) ,dt)
			tau = 20#13
			n_order = 2
			delta = np.power( (np.abs(tarr)) / tau, n_order)
			OTOCarr*=np.exp(-delta)#(1+np.exp(delta))
		
			#tarr = np.arange(0,100.0,0.002)	
			#delta = np.power( (np.abs(tarr)) / tau, n_order)	
			#OTOCarr = np.sin(tarr)
			#OTOCarr*=np.exp(-delta)#(1+np.exp(delta))
		

			FFT = np.fft.fft(np.fft.fftshift(OTOCarr))*dt
			FFT = abs(np.fft.fftshift(FFT))
			freq = np.fft.fftfreq(len(tarr), dt)
			freq *= 2.0 * np.pi
			freq = np.fft.fftshift(freq)
			#print('freq',freq.shape,freq[1]-freq[0]) 
			
			#tarr = np.arange(0,200.01,0.1)
			#sig = np.cos(tarr)
			#FFT = np.abs(np.fft.fft(sig)*0.1)
			#freq = np.fft.fftfreq(len(tarr),0.1)*2*np.pi
			
			plt.plot(freq, FFT)
			plt.xlim([-5,5])
			plt.show()
		store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt),rpext,ebar=stdarr)

	if(0): # Energy_histogram
		kwqlist = ['Thermalized_rp_qcart','nbeads_{}_'.format(nbeads), 'beta_{}'.format(beta), potkey+'_']
		kwplist = ['Thermalized_rp_pcart','nbeads_{}_'.format(nbeads), 'beta_{}'.format(beta), potkey+'_']
	
		print('rpext',rpext)	
		fqlist = seed_finder(kwqlist,rpext,dropext=True)
		fplist = seed_finder(kwplist,rpext,dropext=True)

		#print(fqlist)	
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
			xarr.extend(x)
			yarr.extend(y)		
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
		xarr=np.array(xarr)
		yarr=np.array(yarr)

		#ind, = np.where(V<2.5)
		plt.scatter(xarr,yarr)
		plt.show()	
	
		bins = np.linspace(0.0,10/beta,200)
		dE = bins[1]-bins[0]
		
		#Ehist = plt.hist(x=E, bins=bins,density=True,color='r')
		Vhist = plt.hist(x=V, bins=bins,density=True,color='g',alpha=0.5)
		#Khist = plt.hist(x=K, bins=bins,density=True,color='b',alpha=0.5)
		
		plt.axvline(x=nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)	
		plt.axvline(x=2*nbeads*T,ymin=0.0, ymax = 1.0,linestyle='--',color='k',linewidth=4)		
		plt.axvline(x=V.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='g')
		plt.axvline(x=K.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='b')		
		plt.axvline(x=E.mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='r')			
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

	if(1):
		methodkey = 'RPMD'
		enskey = 'thermal'
		corrkey = 'R2'
		suffix = '_sym'
		kwlist = [methodkey,corrkey,syskey,potkey,Tkey+'_',beadkey,'dt_{}'.format(dt),suffix,'E_filtered']
		
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
	
		store_2D_imagedata_column(X,Y,F,'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}{}_E_filtered'.format(enskey,corrkey,potkey,Tkey,nbeads,dt,suffix),rpext,extcol=np.zeros_like(X))

if(0):#RPMD/mc
	methodkey = 'RPMD'
	enskey  = 'thermal'#'mc'
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
		enskey = 'thermal'#

		kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey,'dt_{}'.format(dt)]
		
		tarr,OTOCarr,stdarr = seed_collector(kwlist,Cext,tarr,OTOCarr,allseeds=True,seedcount=1000)
		#estimate_OTOC_slope(kwlist,Cext,tarr,OTOCarr,2.9,3.9,allseeds=False,seedcount=1000,logerr=True)
	
		#print('stdarr', stdarr[1250])
		#plt.plot(tarr,np.log((OTOCarr)))
		#plt.errorbar(tarr,np.log(abs(OTOCarr)),yerr=stdarr/2,ecolor='m',errorevery=100,capsize=2.0)
		#plt.show()
		store_1D_plotdata(tarr,OTOCarr,'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt),Cext,ebar=stdarr)


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
