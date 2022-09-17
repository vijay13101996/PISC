import numpy as np
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from plt_util import prepare_fig, prepare_fig_ax
dim = 1
lamda = 2.0
g = 0.08

Tc = lamda*(0.5/np.pi)
times = 0.95
T = times*Tc

m = 0.5
N = 1000
dt = 0.002

nbeads = 8
gamma = 16
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
qext = '{}/quantum/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)
sysname='Selene'

fig,ax = prepare_fig_ax(tex=True, dim=1)#plt.subplots()
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\textup{log}\,C_{T}(t)$')
if(1):#high temp classical vs quantum
	basis_N=60
	N_eigen=20
	for times,c in zip([5.0,10.0],['r','g']):
		corrkey = 'OTOC'
		enskey = 'thermal'#'mc'#'thermal'
		Tkey = 'T_{}Tc'.format(times)
		ext = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,0.002)#dt)
		ext = Cext+ext
		plot_1D(ax,ext, label=r'$T=%.1f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)
		if(times>=2.0):
		    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.5,3.5)
		    ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		ext = 'Quantum_OTOC_inv_harmonic_lambda_{}_g_{}_T_{}Tc_basis_{}_n_eigen_{}'
		l3_t1 = qext + ext.format(2.0,0.08,times,basis_N,N_eigen)
		plot_1D(ax,l3_t1, label=r'$Q, T =%.1f T_c$'%times, log=True,linewidth=1)
	file_dpi=600
	plt.legend(fontsize='small',fancybox=True)
	plt.show()
		#fig.savefig('classical/plots/Thermal_OTOC_cl_high_temps.pdf',format='pdf',bbox_inches='tight', dpi=file_dpi)
		#fig.savefig('classical/plots/Thermal_OTOC_cl_high_temps.png',format='png',bbox_inches='tight', dpi=file_dpi)

if(0):#high temp classical OTOC
	for times,c in zip([1.0,2.0,5.0,10.0],['r','g','b','k']):
		corrkey = 'OTOC'
		enskey = 'thermal'#'mc'#'thermal'
		Tkey = 'T_{}Tc'.format(times)
		ext = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,0.002)#dt)
		ext = Cext+ext
		plot_1D(ax,ext, label=r'$T=%.1f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)
		if(times>=2.0):
		    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.5,3.5)
		    ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		#ax.set_xlim([0,5.5])#problematic since not ending before end of plot
		file_dpi=600
		plt.legend(fontsize='small',fancybox=True)
	fig.savefig('classical/plots/Thermal_OTOC_cl_high_temps.pdf',format='pdf',bbox_inches='tight', dpi=file_dpi)
	fig.savefig('classical/plots/Thermal_OTOC_cl_high_temps.png',format='png',bbox_inches='tight', dpi=file_dpi)

if(0):#low temp classical OTOC
	for times,c in zip([0.8,0.9,0.95,1.0],['r','g','b','k']):
		corrkey = 'OTOC'
		enskey = 'thermal'#'mc'#'thermal'
		Tkey = 'T_{}Tc'.format(times)
		ext = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,0.002)#dt)
		ext = Cext+ext
		plot_1D(ax,ext, label=r'$T=%.2f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)
		if(times>=2.0):
		    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.5,3.5)
		    ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		#ax.set_xlim([0,5.5])#problematic since not ending before end of plot
		file_dpi=600
		plt.legend(fontsize='small',fancybox=True)
	fig.savefig('classical/plots/Thermal_OTOC_cl_low_temps.pdf',format='pdf',bbox_inches='tight', dpi=file_dpi)
	fig.savefig('classical/plots/Thermal_OTOC_cl_low_temps.png',format='png',bbox_inches='tight', dpi=file_dpi)


if(0):#thermal RPMD OTOC high T
	for times,c in zip([1.0,2.0,5.0,10.0],['r','g','b','k']):
		corrkey = 'OTOC'
		enskey = 'thermal'
		Tkey = 'T_{}Tc'.format(times)

		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+ext
		plot_1D(ax,ext, label=r'$T=%.1f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)
		if(times>=2.0):
			slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.5,3.5)#3.2,4.5)
			ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		if(times==1.0):
		    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)
		    ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='tab:grey')
		file_dpi=600
		plt.legend(fontsize='small',fancybox=True)
	fig.savefig('rpmd/plots/Thermal_OTOC_RPMD_{}_high_temps.pdf'.format(nbeads),format='pdf',bbox_inches='tight', dpi=file_dpi)
	fig.savefig('rpmd/plots/Thermal_OTOC_RPMD_{}_high_temps.png'.format(nbeads),format='png',bbox_inches='tight', dpi=file_dpi)

if(0):#thermal RPMD OTOC low T
	for times,c in zip([0.8,0.9,0.95,1.0],['r','g','b','tab:grey']):
		corrkey = 'OTOC'
		enskey = 'thermal'
		Tkey = 'T_{}Tc'.format(times)

		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+ext
		plot_1D(ax,ext, label=r'$T=%.2f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)
		if(0.9==times and  0==1):
			slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,5,5.5)#3.2,4.5)
			ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		if(0.95==times and 0==1):
			slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,4,4.5)#3.2,4.5)
			ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
		if(times==1.0):
			slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.2,4.4)
			ax.plot(t_trunc, 0.3+slope*t_trunc+ic,linewidth=1.75,color='k',linestyle='dashed')
			ax.text(t_trunc[10], 0.95+slope*t_trunc[10]+ic,r'$\alpha \exp (2t)$',rotation = 46)
		file_dpi=600
		plt.legend(fontsize='small',fancybox=True)
	fig.savefig('rpmd/plots/Thermal_OTOC_RPMD_{}_low_temps.pdf'.format(nbeads),format='pdf',bbox_inches='tight', dpi=file_dpi)
	fig.savefig('rpmd/plots/Thermal_OTOC_RPMD_{}_low_temps.png'.format(nbeads),format='png',bbox_inches='tight', dpi=file_dpi)



if(0):
	ext = 'Quantum_Kubo_OTOC_inv_harmonic_lambda_{}_g_{}_T_{}Tc_basis_100_n_eigen_20'#140,50
	l3_t1 = qext + ext.format(2.0,0.08,times)
	#l3_t2 = qext + ext.format(1.5,0.035,0.8)
	#l3_t3 = qext + ext.format(1.5,0.035,0.6) 
	#l3_t4 = qext + ext.format(1.5,0.035,0.4) 
	#l3_t5 = qext + ext.format(1.5,0.035,0.2) 

	plot_1D(ax,l3_t1, label=r'$Quantum,T=T_c$',color='m', log=True,linewidth=1)
	#plot_1D(ax,l3_t2, label='Quantum,T=0.8Tc',color='k', log=True,linewidth=1)
	#plot_1D(ax,l3_t3, label='Quantum,T=0.6Tc',color='g', log=True,linewidth=1)
	#plot_1D(ax,l3_t4, label='Quantum,T=0.4Tc',color='r', log=True,linewidth=1)
	#plot_1D(ax,l3_t5, label='Quantum,T=0.2Tc',color='b', log=True,linewidth=1)

	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(l3_t1,1.1,1.9)#1.63,2.67)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
	
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
plt.legend(fontsize='small',fancybox=True)
plt.show()



