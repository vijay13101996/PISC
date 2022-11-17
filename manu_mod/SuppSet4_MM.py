import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials.double_well_potential import double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman'})
matplotlib.rcParams['axes.unicode_minus'] = False

path = os.path.dirname(os.path.abspath(__file__))
Cext = '/home/vgs23/PISC/examples/1D/classical/Datafiles/'
qext = '/home/vgs23/PISC/examples/1D/quantum/Datafiles/'
rpext = '/home/vgs23/PISC/examples/1D/rpmd/Datafiles/'

xl_fs = 18
yl_fs = 18
tp_fs = 16
le_fs = 10
ti_fs = 9

rpc = 'olivedrab'
qc = 'orangered'
Cc = 'slateblue'



ngrid=400
L = 4.0#7.5
lb = -L
ub = L
m = 0.5

lamda = 2.0
g = 0.08#0.02
Tc = lamda*(0.5/np.pi)    
pes = double_well(lamda,g)
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

qgrid = np.linspace(lb,ub,ngrid-1)
potgrid = pes.potential(qgrid)
hessgrid = pes.ddpotential(qgrid)
idx = np.where(hessgrid[:-1] * hessgrid[1:] < 0 )[0] +1
idxl=idx[0]
idxr=idx[1]
Einf = potgrid[idxl]
print('idx', idxl, qgrid[idxl], hessgrid[idxl], hessgrid[idxl-1])
print('E inflection', Einf)	

qgrid_inf = qgrid[idxl:idxr]
potgrid_inf = potgrid[idxl:idxr]

fig,ax = plt.subplots(2,2,gridspec_kw={'width_ratios': [1, 1.5]}) #

narr = [0,2]#[0,2,4,10]#
Earr = [1.3,3.18]#[1.39,4.09,6.64,12.59]#

for i in range(2): #
	ax[i,0].set_ylim([0,7.5])#19])
	ax[i,0].plot(qgrid_inf,potgrid_inf,color='indianred',lw=3)
	qgridl = qgrid[:idxl+1]
	potgridl = potgrid[:idxl+1]
	qgridr = qgrid[idxr-1:]
	potgridr = potgrid[idxr-1:]
	ax[i,0].plot(qgridl,potgridl,color='slateblue',lw=2)
	ax[i,0].plot(qgridr,potgridr,color='slateblue',lw=2)

	for j in range(15):
		if(j==narr[i]):
			ax[i,0].axhline(y=vals[j],color='crimson',lw=2.,zorder=10)
		else:
			ax[i,0].axhline(y=vals[j],color='khaki',lw=1.5)
	ax[i,0].axhline(y=Einf, ls='--',color='g')
	ax[i,0].set_ylabel(r'$V(x)$',fontsize=yl_fs)

	ext = 'Quantum_mc_OTOC_{}_n_{}_basis_50'.format(potkey,narr[i])
	ext =qext+ext
	plot_1D(ax[i,1],ext,label=r'$Quantum$',color=qc, log=True,linewidth=1)
	
	ext = 'Classical_mc_OTOC_{}_T_1.0Tc_dt_0.002_E_{}'.format(potkey,Earr[i])
	ext = Cext+ext
	plot_1D(ax[i,1],ext,label=r'$Classical$',color=Cc, log=True,linewidth=1)

	for times,nbeads,sty in zip([3.0,0.95,0.6],[16,16,32],['-','--','-.']):
		ext = 'RPMD_mc_OTOC_inv_harmonic_lambda_2.0_g_{}_T_{}Tc_nbeads_{}_dt_0.005_E_{}'.format(g,times,nbeads,Earr[i])
		ext = rpext+ext
		plot_1D(ax[i,1],ext,label='$T={}T_c$'.format(times),color=rpc,style=sty, log=True,linewidth=1)

	ax[i,1].set_ylabel(r'$ln \: C_{mc}(t)$',fontsize=xl_fs)
	if(i<1): #
		ax[i,0].set_xticks([])#labels([])
		ax[i,1].set_xticks([])#labels([])

	ax[i,0].tick_params(axis='both', which='major', labelsize=tp_fs)
	ax[i,1].tick_params(axis='both', which='major', labelsize=tp_fs)




ax[1,0].set_xlabel(r'$x$',fontsize=xl_fs) #
ax[1,1].set_xlabel(r'$t$',fontsize=xl_fs) #

ax[0,1].set_ylim([-2.0,5.0])
ax[0,1].set_yticks(np.arange(-1.0,4.01))
ax[1,1].set_yticks(np.arange(0.0,12.01,2))
ax[1,1].set_xticks(np.arange(0.0,5.01))
ax[0,1].legend(fontsize=le_fs,ncol=2,loc='upper left')
ax[1,1].legend(fontsize=le_fs,ncol=1)

#ax[0,0].set_title('Energy levels',fontsize=ti_fs)
#ax[0,1].set_title('Quantum vs Classical OTOC',fontsize=ti_fs)

#fig.suptitle('Microcanonical OTOCs for the 1D double well, $\lambda = {}, g={}$'.format(lamda,g),fontsize=20) 	
plt.subplots_adjust(hspace=0.1,wspace=0.3)
fig.set_size_inches(8, 8)
fig.savefig('/home/vgs23/Images/S8_D2.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
plt.show()

