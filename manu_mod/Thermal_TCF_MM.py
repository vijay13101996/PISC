import numpy as np
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os
import matplotlib 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope


plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman'})
matplotlib.rcParams['axes.unicode_minus'] = False
path = os.path.dirname(os.path.abspath(__file__))
Cext = '/home/vgs23/PISC/examples/1D/classical/Datafiles/' #
qext = '/home/vgs23/PISC/examples/2D/quantum/Datafiles/' #
rpext = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles/' #

m=0.5

lamda = 2.0
g = 0.08

Vb =lamda**4/(64*g)
Tc = lamda*0.5/np.pi

D = 3*Vb
alpha = 0.382

print('Vb', Vb)
z = 1.0	

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
#potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

rpc = 'olivedrab'
qc = 'orangered'
Cc = 'slateblue'

xl_fs = 10
yl_fs = 10
tp_fs = 10
le_fs = 12
ti_fs = 12

lwd = 1.5

if(1):
	fig,ax = plt.subplots(3,2, gridspec_kw={'width_ratios': [3, 1]})

	xticks = np.arange(0.0,20.01,2)

	for i, times,nb in zip([0,1,2],[3.0,0.95,0.6],[16,32,32]):
		T_au = times*Tc 
		beta = 1.0/T_au 
		Tkey = 'T_{}Tc'.format(times)

		#Quantum
		ext = 'Quantum_Kubo_qp_TCF_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,70,100)
		#ext = 'Quantum_Kubo_qp_TCF_{}_{}_basis_{}_n_eigen_{}'.format(potkey,Tkey,100,70)		
		ext =qext+ext
		data = read_1D_plotdata('{}.txt'.format(ext))
		plot_1D(ax[i,0],ext,label='Quantum',color=qc, log=False,linewidth=lwd)
		plot_1D(ax[i,1],ext,label='Quantum',color=qc,log=False,linewidth=lwd,magnify=5)

		#ext = 'Quantum_Kubo_OTOC_{}_{}_basis_{}_n_eigen_{}'.format(potkey,Tkey,100,70)		
		ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,40,70)
		ext =qext+ext	
		plot_1D(ax[i,1],ext,label='Quantum',color=qc,log=True,linewidth=lwd,style='--')

		#RPMD
		ext = 'RPMD_thermal_pq_TCF_{}_{}_nbeads_{}_dt_{}'.format(potkey,Tkey,nb,0.02)
		ext =rpext+ext
		data = read_1D_plotdata('{}.txt'.format(ext))
		plot_1D(ax[i,0],ext,label='RPMD',color=rpc, log=False,linewidth=lwd)
		plot_1D(ax[i,1],ext,label='RPMD',color=rpc,log=False,linewidth=lwd,magnify=5)

		#Classical
		#ext = 'Classical_thermal_qq_TCF_{}_{}_dt_{}'.format(potkey,Tkey,0.02)
		ext = 'RPMD_thermal_pq_TCF_{}_{}_nbeads_{}_dt_{}'.format(potkey,Tkey,1,0.02)	
		ext =rpext+ext#Cext+ext
		data = read_1D_plotdata('{}.txt'.format(ext))
		plot_1D(ax[i,0],ext,label='Classical',color=Cc, log=False,linewidth=lwd)
		plot_1D(ax[i,1],ext,label='Classical',color=Cc,log=False,linewidth=lwd,magnify=5)

		
	for i in range(3):
		ax[i,0].tick_params(axis='both', which='major', labelsize=tp_fs)
		#ax[i].set_ylim([-0.4,0.5])
		#ax[i,0].set_ylabel(r'$t$', fontsize=xl_fs)
		ax[i,0].set_ylabel(r'$\langle x(t) p_x(0) \rangle$',fontsize=yl_fs)
		ax[i,1].set_xlim([0.0,5.1])
		ax[i,1].set_xticks(np.arange(0.0,5.01))
		ax[i,1].axvline(x=0.7,ls='--',color='gray',lw=1.0,zorder=10)

	ax[0,0].set_xticks([])#xticks)
	ax[0,0].set_title(r'$T=3T_c$', fontsize=ti_fs, x=0.5,y=0.8)
	ax[0,0].set_ylim([-0.4,0.5])
	ax[0,1].set_xticks([])
	ax[0,1].set_ylim([-1.3,2.3])

	ax[1,0].set_xticks([])#xticks)
	ax[1,1].set_xticks([])
	ax[1,0].set_title(r'$T=0.95T_c$',fontsize=ti_fs, x=0.5,y=0.8)

	ax[2,0].set_xlabel(r'$t$',fontsize=xl_fs)
	ax[2,0].set_xticks(xticks)
	ax[2,0].set_title(r'$T=0.6T_c$', fontsize=ti_fs,x=0.5,y=0.8)
	ax[2,1].set_xlabel(r'$t$',fontsize=xl_fs)
	
	fig.set_size_inches(8, 6)
	plt.subplots_adjust(hspace=0.06,wspace=0.08)
	#handles, labels = ax.get_legend_handles_labels()
	fig.legend(loc = (0.25, 0.92),ncol=3, fontsize=le_fs)

	fig.savefig('/home/vgs23/Images/Thermal_qp_TCFs_D2.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	plt.show()

if(1):
	fig, ax = plt.subplots()
	qext1 = '/home/vgs23/PISC/examples/1D/quantum/Datafiles/' #
	potkey1 = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

	times = 0.95
	T = times*Tc
	Tkey = 'T_{}Tc'.format(times)
	beta = 1/T
	lwd = 1.5	
	ext = 'Quantum_Kubo_qq_TCF_{}_{}_basis_{}_n_eigen_{}_LONG'.format(potkey1,Tkey,100,70)			
	ext = qext1+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	datazero = np.abs(np.array(data[:,1]))
	print('datazero',datazero.shape)
	#index = np.argmin(datazero)
	index = np.where(datazero<1e-2)[0]
	print('index',index,data[index,0],abs(data[index,1]))	
	timeperiod = data[index[2],0] - data[index[0],0]
	print('Time period', timeperiod, 2*np.pi/timeperiod )
	
	plot_1D(ax,ext,label='$z = 0.0$',color=qc, log=False,linewidth=lwd,style=':')
	ax.scatter(data[index[2],0], 0.0, color='k')
	ax.scatter(data[index[0],0], 0.0, color='k')
	
	ext = 'Quantum_Kubo_qq_TCF_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,70,100)
	ext = qext+ext
	plot_1D(ax,ext,label='$z = 1.0$',color=qc, log=False,linewidth=lwd,style='-')
	
	
	ax.legend(fontsize=le_fs)
	ax.set_xticks(np.arange(0,200.1,20))	
	ax.axhline(y=0.0,ls='--',color='gray',lw=1.0,zorder=10)
	ax.set_ylabel(r'$\langle x(t) x(0) \rangle$',fontsize=yl_fs)
	ax.set_xlabel(r'$t$',fontsize=yl_fs)
		

	fig.set_size_inches(4, 4)
	#handles, labels = ax.get_legend_handles_labels()
	#fig.legend(loc = (0.27, 0.9),ncol=3, fontsize=le_fs)

	fig.savefig('/home/vgs23/Images/Thermal_qq_TCFs_D2.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	
	plt.show()	
