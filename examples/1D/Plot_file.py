import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim = 1

if(1): #Double Well potential
	lamda = 2.0
	g = 0.08#8

	Tc = lamda*(0.5/np.pi)
	times = 5.0
	T = times*Tc

	m = 0.5
	
	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

if(1): #Mildly anharmonic
	omega = 1.0
	a = 1.0#/10
	b = 1.0#/100
	
	T = 1.0/8#times*Tc
	
	m=1.0
		
	potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)
	Tkey = 'T_{}'.format(T)


N = 1000
dt = 0.01
nbeads = 1
gamma = 16

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

fig,ax = plt.subplots()

if(1):
	for nbeads,sty in zip([1,8],['-','-.']):
		#nbeads = 1
		corrkey = 'singcomm'
		enskey = 'thermal'
		fname_sc = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+fname_sc
		plot_1D(ax,ext, label=r'Commutator, $N_b={}$'.format(nbeads),color='k',linewidth=1.5,style=sty)
			
		corrkey = 'pq_TCF'
		fname_pq = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+fname_pq
		plot_1D(ax,ext, label=r'pq TCF, $N_b={}$'.format(nbeads),color='m',linewidth=1.5,magnify=(1.0/(m*nbeads*T)),style=sty)
		
if(0):
	dt = 0.002
	corrkey = 'OTOC'
	enskey = 'thermal'
	potkey_asym = 'asym_double_well_lambda_{}_g_{}_k_{}'.format(2.0,0.08,0.04)
	potkey_sym  = 'asym_double_well_lambda_{}_g_{}_k_{}'.format(2.0,0.08,0.0)
	
	T_arr = [0.95,1.2,1.4,3.0,10.0]
	bead_arr = 	[16,16,16,8,8]
	color_arr = ['r','g','b','c','k','m']
	td_arr = [3.3,3.0,3.0,2.0,1.0]
	tu_arr = [4.3,4.0,4.0,3.0,2.0]
	for times,beads,c,td,tu in zip(T_arr,bead_arr,color_arr,td_arr,tu_arr):
		Tkey = 'T_{}Tc'.format(times)
		fname_asym = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey_asym,Tkey,beads,dt)
		ext = rpext+fname_asym
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
		ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		plot_1D(ax,ext, label=r'$T={}Tc$, Asymmetric, $\lambda={}$'.format(times,np.around(slope,2)),log=True,color=c,linewidth=1.5)
		
		fname_sym = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey_sym,Tkey,beads,dt)
		ext = rpext+fname_sym
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
		ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		plot_1D(ax,ext, label=r'$T={}Tc$, Symmetric, $\lambda={}$'.format(times,np.around(slope,2)),log=True,color=c,linewidth=1.5,style='--')
		
	
	#plt.show()

if(0):
	dt = 0.005
	corrkey = 'singcomm'#'OTOC'
	enskey = 'thermal'
	Tkey = 'T_{}Tc'.format(times)
	
	fname = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt)
	ext = Cext+fname
	plot_1D(ax,ext, label=r'Unfiltered'.format(times),log=False,color='k',linewidth=2)
		

	for sigma,c in zip([0.5,1.0,5.0,10.0],['r','g','b','c']):
		potkey = 'FILT_{}_g_{}_sigma_{}_q0_{}'.format(lamda,g,sigma,0.0)
		fname = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt)
		ext = Cext+fname
		plot_1D(ax,ext, label=r'$\sigma={}$'.format(sigma),log=False,color=c,linewidth=2)
		
	


if(0):
	dt = 0.02
	corrkey = 'qq_TCF'#'OTOC'
	enskey = 'thermal'
	Tkey = 'T_{}Tc'.format(times)

	fname = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,1,dt)
	ext = rpext+fname
	plot_1D(ax,ext, label=r'$T={}Tc$,classical'.format(times),log=True,color='c',linewidth=2)
		
	fname = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,16,dt)
	ext = rpext+fname
	plot_1D(ax,ext, label=r'$T={}Tc$,RPMD'.format(times),log=True,color='r',linewidth=2)
		
	fname = 'Quantum_{}_{}_{}_{}_basis_{}_n_eigen_{}'.format('Kubo',corrkey,potkey,Tkey,100,70)	
	ext = qext+fname
	plot_1D(ax,ext, label=r'$T={}Tc$,Quantum'.format(times),log=True,color='k',linewidth=2)
		

if(0):
	corrkey = 'qq_TCF'
	enskey = 'thermal'
	dt = 0.02
	for times,c in zip([10.0],['k','c','r','g','b','y','m']):
		Tkey = 'T_{}Tc'.format(times)
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(0.382,9.375,lamda,0.08,1.0)
		ext = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt)
		ext = Cext+ext
		plot_1D(ax,ext, label=r'$T={}Tc$,classical'.format(times),color='c',linewidth=2)
		
		potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
		ext = 'Quantum_qq_TCF_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,100,70)
		ext = qext+ext
		plot_1D(ax,ext, label=r'$T={}Tc$'.format(times),color='k',linewidth=2)
		
if(0):
	corrkey = 'OTOC'
	enskey = 'mc'#'thermal'
	Tkey = 'T_{}Tc'.format(times)

	E = 10.0
	ext = 'Classical_{}_{}_{}_{}_dt_{}_E_{}'.format(enskey,corrkey,potkey,Tkey,0.002,E)
	ext = Cext+ext
	plot_1D(ax,ext, label=r'$Classical, E={}$'.format(E),color='c', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,0.6,1.25)#0.45,1.)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	E=1.3
	ext = 'Quantum_{}_{}_{}_basis_{}_n_eigen_{}_E_{}'.format(enskey,corrkey,potkey,50,20, E)
	ext = qext+ext
	plot_1D(ax,ext, label=r'$Quantum, E={}$'.format(E),color='k', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,0.45,1.0)
	#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}_E_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,0.005,E)
	ext = rpext+ext
	plot_1D(ax,ext, label=r'$RPMD, E={}, T=T_c$'.format(E),color='m', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,0.45,1.0)
	
	plt.title('Microcanonical OTOCs, E={}'.format(E))

if(0):
	corrkey = 'OTOC'
	enskey = 'mc'#'thermal'
	Tkey = 'T_{}Tc'.format(1.0)

	E = 2.75
	td = 0.6
	tu = 1.2
	ext = 'Classical_{}_{}_{}_{}_dt_{}_E_{}'.format(enskey,corrkey,potkey,Tkey,0.002,E)
	ext = Cext+ext
	plot_1D(ax,ext, label=r'Classical mc double commutator, $E={}$'.format(E),color='k', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	corrkey = 'singcomm'
	#enskey = 'thermal'
	#Tkey = 'T_{}Tc'.format(times)

	ext = 'Classical_{}_{}_{}_{}_dt_{}_E_{}'.format(enskey,corrkey,potkey,Tkey,0.002,E)
	ext = Cext+ext
	plot_1D(ax,ext, label=r'Classical mc single commutator, $E={}$'.format(E),color='c', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
	#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

if(0):
	corrkey = 'OTOC'
	enskey = 'thermal'
	Tkey = 'T_{}Tc'.format(times)

	ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
	ext = rpext+ext
	print('f',ext)
	plot_1D(ax,ext, label=r'$RPMD, T=T_c$',color='m', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)
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



