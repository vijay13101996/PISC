import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.potentials.Quartic_bistable import quartic_bistable

dim=2

alpha = 0.37
D = 9.375 

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

z = 1.0
 
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc
beta=1/T

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
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
	if(0):
		methodkey = 'RPMD'

		kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey]
		
		tarr,OTOCarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

		if(corrkey!='OTOC'):
			OTOCarr/=nbeads
		#plt.plot(tarr,OTOCarr)
		plt.plot(tarr,np.log(abs(OTOCarr)))
		plt.show()
		store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_nbeads_{}_dt_{}'.format(corrkey,potkey,Tkey,nbeads,dt),rpext)

	if(1): # Energy_histogram
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

			pot = np.sum(pes.potential(qcart),axis=1)
			kin = np.sum(np.sum(pcart**2/(2*m),axis=1),axis=1)	
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
		plt.hist(x=E, bins=50,color='r')
		plt.hist(x=V, bins=50,color='g',alpha=0.5)
		plt.hist(x=K, bins=50,color='b',alpha=0.5)
		plt.axvline(x=2*m/beta,ymin=0.0, ymax = 1.0,linestyle='--',color='k')	
		plt.axvline(x=Vb,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
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

if(0):#Classical
	if(0):
		methodkey = 'Classical'
		enskey = 'mc'

		kwlist = [enskey,methodkey,corrkey,syskey,potkey,Tkey]
		
		tarr,OTOCarr = seed_collector(kwlist,Cext,tarr,OTOCarr)

		plt.plot(tarr,np.log(abs(OTOCarr)))
		plt.show()
		store_1D_plotdata(tarr,OTOCarr,'Classical_{}_{}_{}_dt_{}'.format(corrkey,potkey,Tkey,dt),Cext)

	if(1): # Energy_histogram
		kwqlist = ['Thermalized_rp_qcart', 'beta_{}'.format(beta), potkey]
		kwplist = ['Thermalized_rp_pcart', 'beta_{}'.format(beta), potkey]
		
		fqlist = seed_finder(kwqlist,Cext,dropext=True)
		fplist = seed_finder(kwplist,Cext,dropext=True)
	
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
