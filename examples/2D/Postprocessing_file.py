import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim=2

alpha = 0.363
D = 10.0 

lamda = 2.0
g = 0.08

z = 1.5#1.5
 
Tc = 0.5*lamda/np.pi
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 40.0
time_total = 8.0

nbeads = 4

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
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
	methodkey = 'RPMD'

	kwlist = [methodkey,corrkey,syskey,potkey,Tkey,beadkey]
	
	tarr,OTOCarr = seed_collector(kwlist,rpext,tarr,OTOCarr)

	if(corrkey!='OTOC'):
		OTOCarr/=nbeads
	#plt.plot(tarr,OTOCarr)
	plt.plot(tarr,np.log(abs(OTOCarr)))
	plt.show()
	store_1D_plotdata(tarr,OTOCarr,'RPMD_{}_{}_{}_nbeads_{}_dt_{}'.format(corrkey,potkey,Tkey,nbeads,dt),rpext)

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
	methodkey = 'Classical'

	kwlist = [methodkey,corrkey,syskey,potkey,Tkey]
	
	tarr,OTOCarr = seed_collector(kwlist,Cext,tarr,OTOCarr)

	plt.plot(tarr,np.log(abs(OTOCarr)))
	plt.show()
	store_1D_plotdata(tarr,OTOCarr,'Classical_{}_{}_{}_dt_{}'.format(corrkey,potkey,Tkey,dt),Cext)


