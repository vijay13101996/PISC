import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,seed_collector,seed_finder,seed_collector_imagedata
from matplotlib import pyplot as plt
import os
from PISC.potentials import double_well, quartic, morse, mildly_anharmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr, store_2D_imagedata_column
from PISC.utils.nmtrans import FFT
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

tp_fs = 13
xl_fs = 16
yl_fs = 16

le_fs = 9.5
ti_fs = 12

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

Tc = lamda*(0.5/np.pi)
times = 3.0
T = times*Tc
beta=1/T
print('T',T)

m = 0.5

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
pes = double_well(lamda,g)

Tkey = 'T_{}Tc'.format(times)

Tarr = [0.6,0.95,3]#1.4,1.8,2.2,2.6,3.0,5.0]
beadarr =[32,16,8]#16,16,8,8,8,8]

fig,ax = plt.subplots(3)#4,2)


path = '/home/vgs23/PISC/examples/1D'
rpext = '{}/rpmd/Datafiles/'.format(path)

count = 0
for nbeads,times in zip(beadarr,Tarr):
	T = times*Tc
	print('T=', times, 'Tc')
	beta = 1/T
	kwqlist = ['Thermalized_rp_qcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]
	kwplist = ['Thermalized_rp_pcart','nbeads_{}'.format(nbeads), 'beta_{}'.format(beta), potkey]

	fqlist = seed_finder(kwqlist,rpext,dropext=True)
	fplist = seed_finder(kwplist,rpext,dropext=True)

	rg = []
		
	for qfile in fqlist:
		qcart = read_arr(qfile,rpext)
		#pcart = read_arr(pfile,rpext)

		fft = FFT(1,nbeads)
		q = fft.cart2mats(qcart)
		#p = fft.cart2mats(pcart)

		qcent = q[:,0,0]/nbeads**0.5
			
		gyr=np.mean((qcent[:,None]-qcart[:,0,:])**2,axis=1)**0.5		
		
		rg.extend(gyr)
	
	bins = np.linspace(0.0,2.5,101)
	#print('max rg', max(rg))
	q,r = divmod(count,4)
	
	hist, bin_edges = np.histogram(rg, bins)
	hist=np.array(hist)/2000
	weights = 100*np.ones(len(hist))/len(hist)
	#print('shape',hist, bin_edges)
	ax[count].hist(bin_edges[1:], bins=bin_edges[1:], weights=hist,density=False,color='g',alpha=0.5)
	ax[count].axvline(x=np.array(rg).mean(),ymin=0.0, ymax = 1.0,linestyle='--',color='k')
	ax[count].annotate('T={}Tc'.format(times),xy=(0.83,0.22), xytext=(0.7,0.8), xycoords='axes fraction',fontsize=tp_fs)
#,arrowprops=dict(facecolor='black', width=0.03,headlength=5.0,headwidth=5.0))


	#print('N, bin', N[0], bins[0])
	#print('bin_edges', bin_edges)
	#print('sum',hist*bin_edges[1])	
	count+=1

if(1):
	for i in range(3):
		ax[i].set_yticks([0,2,4,6,8])
		ax[i].set_xlim([0.0,1.25])
		ax[i].set_yticklabels=([r'$0.0$',r'$10.0$',r'$20.0$',r'$30.0$'])
		ax[i].set_ylabel(r'$\% \; N_{traj}$', fontsize=yl_fs)
		ax[i].tick_params(axis='both', which='major', labelsize=tp_fs)
		
	ax[0].set_xticks([])
	ax[1].set_xticks([])
	ax[2].set_xticks(np.arange(0,1.21,0.2))
	ax[2].set_xlabel(r'$r_g$', fontsize=xl_fs)

fig.set_size_inches(4, 5)	
fig.savefig('/home/vgs23/Images/rg_hist_1.pdf'.format(g), dpi=400,bbox_inches='tight',pad_inches=0.0)


plt.show()

