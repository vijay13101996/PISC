from cProfile import label
import numpy as np
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim=2

alpha = 0.38
D = 9.375 

lamda = 2.0
g = 0.08

z = 1.5
 
Tc = 0.5*lamda/np.pi
times = 0.95
T = times*Tc

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 100
time_total = 6.0

nbeads = 16

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

fig,ax = plt.subplots()


if(0):
	corrkey = 'OTOC'
	enskey ='mc'#'thermal'
	Tkey = 'T_{}Tc'.format(times)

	ext = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt)
	ext = Cext+ext
	plot_1D(ax,ext, label=r'$Classical, T=T_c$',color='c', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.,2.)
	print('slope',slope/2)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

if(0):
	for times,c in zip([0.7,0.9,1.0],['r','g','b']):	
		Tkey = 'T_{}Tc'.format(times)
		ext ='RPMD_mc_{}_{}_{}_nbeads_{}_dt_{}'.format(corrkey,potkey,Tkey,nbeads,dt)
		extclass = rpext + ext
		print('fname',extclass)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(extclass,1,2)
		print('2pi/beta', 2*np.pi*T,slope/2)	
		plot_1D(ax,extclass, label=r'$RPMD,T={}Tc,\; N_b={}, \lambda={:.3f}$'.format(times,nbeads,np.real(slope/2)),color=c, log=True,linewidth=1)	
		ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')
	
	plt.title('RPMD microcanonical OTOCs for the 2D double well')	

if(1):
	n_seeds=1000
	#for nbeads,c in zip([8,16],['r','g']):
	for nbeads,c in zip([8],['r']):	
		Tkey = 'T_{}Tc'.format(times)
		x_arr=np.zeros((int(time_total/dt),n_seeds))
		y_arr=np.zeros((int(time_total/dt),n_seeds))
		for seeds in range(n_seeds):
			ext ='RPMD_thermal_{}_{}_{}_{}_N_{}_dt_{}_nbeads_{}_seed_{}'.format(corrkey,syskey,potkey,Tkey,N,dt,nbeads,seeds)
			extclass = rpext + ext
			#print('fname',extclass)
			data = read_1D_plotdata('{}.txt'.format(extclass))
			x_arr[:,seeds]=data[:,0]
			y_arr[:,seeds]=data[:,1]
		x=np.mean(x_arr,axis=1)#mean of t which is always correct but useless
		y=np.mean(y_arr,axis=1)
		y_std=np.std(y_arr,axis=1)
		log_y=np.mean(np.log(y_arr),axis=1)
		log_std= np.std(np.log(y_arr),axis=1)
		#print(np.min(y_std))
		#print(np.min(np.log(y_std)))
		#seems wrong because log(y_std) will give crazy derivations
		#ax.errorbar(x,np.log(y),np.log(y_std),label='N = {}_log_mean'.format(nbeads))
		ax.errorbar(x,log_y,log_std,ecolor='red',label='N = {}_log_mean'.format(nbeads))
		#ax.plot(x,log_y,label='N = {}_mean_log'.format(nbeads))
		ax.set_xlim(0,6.5)
		#ax.set_ylim(0,10)
		#slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(extclass,1,2)
		#print('2pi/beta', 2*np.pi*T,slope/2)	
		#plot_1D(ax,extclass, label=r'$RPMD,T={}Tc,\; N_b={}, \lambda={:.3f}$'.format(times,nbeads,np.real(slope/2)),color=c, log=True,linewidth=1)	
		
		#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')
	
	plt.title('RPMD OTOCs for the 2D double well')	

plt.legend()
plt.show()

