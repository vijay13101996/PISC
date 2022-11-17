import numpy as np
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope


path = os.path.dirname(os.path.abspath(__file__))
Cext = '/home/vgs23/PISC/examples/2D/classical/Datafiles/'
qext = '/home/vgs23/PISC/examples/2D/quantum/Datafiles/'
rpext = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles/'

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

#fig,ax = plt.subplots(1,3)
fig,ax = plt.subplots(1)

rpc = 'b'
qc = 'r'
Cc = 'k'#'g'

if(0):
	### Temperature: 10 Tc -----------------------------------------------------------
		
	times = 10.0
	T_au = times*Tc 
	beta = 1.0/T_au 
	Tkey = 'T_{}Tc'.format(times)

	#RPMD
	ext = 'RPMD_mc_OTOC_{}_{}_nbeads_{}_dt_{}_E_Vb'.format(potkey,Tkey,8,0.005)
	ext =rpext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	tarr = data[:,0]
	Carr = data[:,1]
	plot_1D(ax[0],ext,label='RPMD',color=rpc, log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax[0].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	#Classical
	ext = 'Classical_mc_OTOC_{}_{}_dt_{}_E_Vb'.format(potkey,Tkey,0.002)
	ext =Cext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	tarr = data[:,0]
	Carr = data[:,1]
	plot_1D(ax[0],ext,label='Classical',color=Cc, log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax[0].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	ax[0].set_xlabel(r'$t$',fontsize=15)
	ax[0].set_ylabel(r'$c_{mc}(t)$',fontsize=15)

	ax[0].set_title(r'$T=10T_c$', fontsize=15)


	### Temperature: 0.95 Tc ----------------------------------------------------------
		
	times = 0.95
	T_au = times*Tc 
	beta = 1.0/T_au 
	Tkey = 'T_{}Tc'.format(times)

	#RPMD
	ext = 'RPMD_mc_OTOC_{}_{}_nbeads_{}_dt_{}_E_Vb'.format(potkey,Tkey,16,0.005)
	ext =rpext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	tarr = data[:,0]
	Carr = data[:,1]
	plot_1D(ax[1],ext,label='RPMD',color=rpc, log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax[1].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	#Classical
	ext = 'Classical_mc_OTOC_{}_{}_dt_{}_E_Vb'.format(potkey,Tkey,0.002)
	ext =Cext+ext
	plot_1D(ax[1],ext,label='Classical',color=Cc, log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax[1].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')


	ax[1].set_xlabel(r'$t$',fontsize=15)
	ax[1].set_title(r'$T=0.95T_c$',fontsize=15)
	### Temperature: 0.7 Tc -----------------------------------------------------------
		
	times = 0.7
	T_au = times*Tc 
	beta = 1.0/T_au 
	Tkey = 'T_{}Tc'.format(times)

	#RPMD
	ext = 'RPMD_mc_OTOC_{}_{}_nbeads_{}_dt_{}_E_Vb'.format(potkey,Tkey,32,0.005)
	ext =rpext+ext
	plot_1D(ax[2],ext,label='RPMD',color=rpc, log=True,linewidth=1)

	#Classical
	ext = 'Classical_mc_OTOC_{}_{}_dt_{}_E_Vb'.format(potkey,Tkey,0.002)
	ext =Cext+ext
	plot_1D(ax[2],ext,label='Classical',color=Cc, log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax[2].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')


	ax[2].set_xlabel(r'$t$',fontsize=15)
	ax[2].set_title(r'$T=0.7T_c$', fontsize=15)


	ax[1].legend()
	ax[0].legend()
	ax[2].legend()

	plt.subplots_adjust(wspace=0.1)
	fig.set_size_inches(20, 5)
	#fig.savefig('/home/vgs23/Images/MC_OTOCs_D1.png'.format(g), dpi=100)
	plt.show()

if(1):
	Tkey = 'T_10.0Tc'
	
	#Classical
	ext = 'Classical_mc_OTOC_{}_{}_dt_{}_E_Vb'.format(potkey,Tkey,0.002)
	ext =Cext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	tarr = data[:,0]
	Carr = data[:,1]
	plot_1D(ax,ext,label='Classical',color=Cc, log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	ext = 'RPMD_mc_OTOC_{}_{}_nbeads_{}_dt_{}_E_Vb'.format(potkey,Tkey,8,0.005)
	ext =rpext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	tarr = data[:,0]
	Carr = data[:,1]
	plot_1D(ax,ext,label='RPMD, $10T_c$',color='r', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	Tkey = 'T_0.95Tc'
	ext = 'RPMD_mc_OTOC_{}_{}_nbeads_{}_dt_{}_E_Vb'.format(potkey,Tkey,16,0.005)
	ext =rpext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	tarr = data[:,0]
	Carr = data[:,1]
	plot_1D(ax,ext,label='RPMD, $0.95T_c$',color='g', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	Tkey = 'T_0.7Tc'
	ext = 'RPMD_mc_OTOC_{}_{}_nbeads_{}_dt_{}_E_Vb'.format(potkey,Tkey,32,0.005)
	ext =rpext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	tarr = data[:,0]
	Carr = data[:,1]
	plot_1D(ax,ext,label='RPMD, $0.7T_c$',color='b', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

	ax.set_xlabel(r'$t$',fontsize=15)
	ax.set_ylabel(r'$c_{mc}(t)$',fontsize=15)


ax.legend()
plt.show()
