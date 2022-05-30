import numpy as npimport numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

ext = 'Quantum_OTOC_inv_harmonic_lambda_{}_g_{}_T_{}Tc_basis_120_n_eigen_30'
l3_t1 = qext + ext.format(1.5,0.035,1.0)
#l3_t2 = qext + ext.format(1.5,0.035,0.8)
#l3_t3 = qext + ext.format(1.5,0.035,0.6) 
#l3_t4 = qext + ext.format(1.5,0.035,0.4) 
#l3_t5 = qext + ext.format(1.5,0.035,0.2) 

fig,ax = plt.subplots()
plot_1D(ax,l3_t1, label=r'$Quantum,T=T_c$',color='m', log=True,linewidth=1)
#plot_1D(ax,l3_t2, label='Quantum,T=0.8Tc',color='k', log=True,linewidth=1)
#plot_1D(ax,l3_t3, label='Quantum,T=0.6Tc',color='g', log=True,linewidth=1)
#plot_1D(ax,l3_t4, label='Quantum,T=0.4Tc',color='r', log=True,linewidth=1)
#plot_1D(ax,l3_t5, label='Quantum,T=0.2Tc',color='b', log=True,linewidth=1)

slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,1.86,2.68)#1.63,2.67)
#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

if(1):
	ext = 'RPMD_OTOC_{}_T_0.7Tc_nbeads_16_dt_0.005'.format(potkey)
	l3_t1 = rpext + ext.format(1.5,0.035,1.0)
	plot_1D(ax,l3_t1, label=r'$RPMD,T=Tc,\; N_b=32$',color='g', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,8.28,9.83)#3.26,4.08)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

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


plt.suptitle(r'Comparison of OTOCs at $T=T_c$')
plt.legend()
plt.show()


if(0):
	fig,ax = plt.subplots()	

	for c,nbeads in zip(['c','r','g','b','y'],[4,8,16]):
		fname = '{}/cmd/Datafiles/CMD_OTOC_{}_T_{}Tc_nbeads_{}_gamma_{}_dt_{}'.format(path,potkey,times,nbeads,gamma,dt)
		#plot_1D(ax,fname,label=r'$N_b={}$'.format(nbeads),color=c,log=True,linewidth=1)

	#times =1.0
	fname = '{}/classical/Datafiles/Classical_OTOC_{}_T_{}Tc_dt_{}'.format(path,potkey,times,0.01)	
	plot_1D(ax,fname,label=r'Classical'.format(N),color='m',log=True,linewidth=1)

	fname = '{}/quantum/Datafiles/Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(path,potkey,times,120,40)
	plot_1D(ax,fname,label=r'Quantum, T=$T_c$'.format(N),color='k',log=True,linewidth=1)

	#fname2 ='{}/classical/Datafiles/Classical_OTOC_{}_T_{}Tc_dt_{}'.format(path,potkey,0.9,0.01) 
	#'{}/quantum/Datafiles/Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(path,potkey,0.9,120,40)
	#plot_1D(ax,fname2,label=r'Quantum, T=$0.9T_c$',color='k',log=True,linewidth=1)

	plt.suptitle(r'Classical vs Quantum OTOCs for $\lambda={}, g={}$'.format(lamda,g))
	plt.title(r'$T={}T_c$'.format(times))
	plt.legend()
	plt.show()

if(0):
	fname = '{}/cmd/Datafiles/CMD_OTOC_{}_T_{}Tc_nbeads_{}_gamma_{}_dt_{}'.format(path,potkey,times,16,gamma,dt)		
	fname = '{}/quantum/Datafiles/Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(path,potkey,times,120,40)
		
	data = read_1D_plotdata('{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]

	fig,ax = plt.subplots()	
	plot_1D(ax,fname,label='CMD',color='m',log=True,linewidth=1)

	ist = 118
	iend = 244
	slope, ic, t_trunc, OTOC_trunc = find_slope(t_arr,np.log(OTOC_arr),ist,iend)

	print('Bound', 2*np.pi*T)

	plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
	plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
	plt.show()



