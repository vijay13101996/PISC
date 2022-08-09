import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim = 1
lamda = 2.0
g = 0.08

Tc = lamda*(0.5/np.pi)
times = 1.0
T = times*Tc

m = 0.5
N = 1000
dt = 0.002

nbeads = 16
gamma = 16

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

fig,ax = plt.subplots()

if(1):
	corrkey = 'OTOC'
	enskey = 'mc'#'thermal'
	Tkey = 'T_{}Tc'.format(times)

	ext = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,0.002)#dt)
	ext = Cext+ext
	plot_1D(ax,ext, label=r'$Classical, T=T_c$',color='c', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1,2.)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

if(0):
	corrkey = 'OTOC'
	enskey = 'thermal'
	Tkey = 'T_{}Tc'.format(times)

	ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
	ext = rpext+ext
	plot_1D(ax,ext, label=r'$RPMD, T=T_c$',color='m', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,0.8,1.8)#3.2,4.5)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
	
if(0):
	ext = 'Quantum_OTOC_inv_harmonic_lambda_{}_g_{}_T_{}Tc_basis_140_n_eigen_50'
	l3_t1 = qext + ext.format(1.5,0.035,times)
	#l3_t2 = qext + ext.format(1.5,0.035,0.8)
	#l3_t3 = qext + ext.format(1.5,0.035,0.6) 
	#l3_t4 = qext + ext.format(1.5,0.035,0.4) 
	#l3_t5 = qext + ext.format(1.5,0.035,0.2) 

	#plot_1D(ax,l3_t1, label=r'$Quantum,T=T_c$',color='m', log=True,linewidth=1)
	#plot_1D(ax,l3_t2, label='Quantum,T=0.8Tc',color='k', log=True,linewidth=1)
	#plot_1D(ax,l3_t3, label='Quantum,T=0.6Tc',color='g', log=True,linewidth=1)
	#plot_1D(ax,l3_t4, label='Quantum,T=0.4Tc',color='r', log=True,linewidth=1)
	#plot_1D(ax,l3_t5, label='Quantum,T=0.2Tc',color='b', log=True,linewidth=1)

	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,1.1,1.9)#1.63,2.67)
	#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

if(0):
	for times,c in zip([0.7,0.9,1.0],['r','g','b']):
		ext = 'RPMD_mc_OTOC_{}_T_{}Tc_nbeads_{}_dt_0.002'.format(potkey,times,nbeads)
		l3_t1 = rpext + ext
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,1.,2.)#3.26,4.08)
		ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		print('2pi/beta', 2*np.pi*T,slope/2)
		print('fname',l3_t1)
		plot_1D(ax,l3_t1, label=r'$RPMD,T={}Tc,\; N_b={}, \lambda={:.3f}$'.format(times,nbeads,np.real(slope/2)),color=c, log=True,linewidth=1)
		
	plt.title('RPMD microcanonical OTOCs for the 1D double well')	

if(0):
	ext = 'CMD_OTOC_{}_T_1.0Tc_nbeads_4_gamma_32_dt_0.005'.format(potkey)
	l3_t1 = cext + ext.format(1.5,0.035,1.0)
	plot_1D(ax,l3_t1, label=r'$CMD,\; T=T_c, \gamma=32, N_b=4$',color='b', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,3.26,4.08)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	ext = 'CMD_OTOC_{}_T_1.0Tc_nbeads_16_gamma_16_dt_0.005'.format(potkey)
	l3_t1 = cext + ext.format(1.5,0.035,1.0)
	plot_1D(ax,l3_t1, label=r'$CMD,\; T=T_c, \gamma=16, N_b=16$',color='r', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,3.26,4.08)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	ext = 'Classical_OTOC_{}_T_0.9Tc_dt_0.005'.format(potkey)
	l3_t1 = Cext + ext.format(1.5,0.035,1.0)
	plot_1D(ax,l3_t1, label=r'$Classical, T=T_c$',color='c', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,3.8,4.5)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')


if(0):
	for i,c in zip([1,4],['c','r','k','g']):
		ext = 'RPMD_OTOC_{}_T_{}Tc_nbeads_{}_dt_0.005'.format(potkey,times,i)
		f = rpext + ext#.format(1.5,0.035,1.0,i)	
		plot_1D(ax,f, label=r'$RPMD,T={}Tc,\; N_b={}$'.format(times,i),color=c, log=True,linewidth=1)
		
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(f,2.0,3.0)#1.63,2.67)
		#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

if(0):
	ext = 'Quantum_qq_TCF_{}_T_{}Tc_basis_140_n_eigen_50'.format(potkey,times)
	f = qext + ext.format(1.5,0.035,1.0)	
	plot_1D(ax,f, label=r'$Quantum,T={}Tc$'.format(times),color='g', log=False,linewidth=1)
	
if(0):
	ext = 'Quantum_OTOC_{}_T_{}Tc_basis_140_n_eigen_50'.format(potkey,times)
	f = qext + ext#.format(1.5,0.035,1.0)	
	plot_1D(ax,f, label=r'$Quantum,T={}Tc$'.format(times),color='m', log=True,linewidth=1)

if(0):
	for times,c in zip([2.0,3.0,4.0,10.0], ['c','y','k','g','r','b']):
		ext = 'Quantum_OTOC_{}_T_{}Tc_basis_140_n_eigen_50'.format(potkey,times)
		f = qext + ext#.format(1.5,0.035,1.0)	
		plot_1D(ax,f, label=r'$Quantum,T={}Tc$'.format(times),color=c, log=True,linewidth=1)

if(0):
	ext = 'Quantum_OTOC_{}_T_{}Tc_basis_120_n_eigen_40'.format(potkey,2.0)
	f = qext + ext#.format(1.5,0.035,1.0)	
	plot_1D(ax,f, label=r'$Quantum,T={}Tc$'.format(times),color='k', log=True,linewidth=1)

	ext = 'Quantum_OTOC_{}_T_{}Tc_basis_120_n_eigen_40'.format(potkey,1.0)
	f = qext + ext#.format(1.5,0.035,1.0)	
	plot_1D(ax,f, label=r'$Quantum,T={}Tc$'.format(times),color='y', log=True,linewidth=1)
	
	
	
#plt.suptitle(r'Comparison of OTOCs at $T=T_c$')
plt.legend()
plt.show()



