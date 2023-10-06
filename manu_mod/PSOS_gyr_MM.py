import numpy as np
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
import matplotlib.gridspec as gridspec
import matplotlib

rpext = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles/'
#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman'})
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 14 
yl_fs = 14
tp_fs = 12

le_fs = 9
ti_fs = 12
 
ms=0.15

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

fig = plt.figure()# ,ax = plt.subplots()
plt.subplots_adjust(hspace=0.4,wspace=0.05)

Tkey = 'T_{}Tc'.format(0.95)
max_runtime=1000.0
nbeads=8
fgyr_name = 'RP_max_rg_{}_{}_long_{}_nbeads_{}'.format(potkey,Tkey,max_runtime,nbeads)
gyr_arr = read_arr(fgyr_name,rpext)

gs = gridspec.GridSpec(2,2)#,figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])


ax1.axvline(x=0.55,ymin=0.0,ymax = 1.0,linestyle='--',color='gray')
N,bins,patches = ax1.hist(gyr_arr,weights=100*np.ones(len(gyr_arr)) / len(gyr_arr),bins=60)#,density=True,stacked=True,alpha=0.65,color='b')

percent1 = 0
percent2 = 0
for i in range(0,24):
	patches[i].set_facecolor('darkcyan')
	#patches[i].set_alpha(0.65)
	percent1 += N[i]
	
for i in range(24,60):
	patches[i].set_facecolor('goldenrod')
	#patches[i].set_alpha(0.65)
	percent2 += N[i]

print('percent1+percent2',percent1,percent2)
	
ax1.set_xlabel(r'max $r_g$',fontsize=xl_fs-1)
ax1.set_ylabel(r'$N_{{traj}}$',fontsize=yl_fs)
ax1.set_xticks(np.arange(0.0,1.21,0.2))
ax1.set_yticks([0,2,4,6,8])
ax1.yaxis.set_label_coords(-0.07,0.44)#(-0.13,0.44)
ax1.annotate('83%',xy=(0.63,0.22), xytext=(0.45,0.5), xycoords='axes fraction',fontsize=tp_fs+2,arrowprops=dict(facecolor='black', width=0.03,headlength=5.0,headwidth=5.0))
ax1.annotate('17%',xy=(0.18,0.17), xytext=(0.22,0.5), xycoords='axes fraction',fontsize=tp_fs+2,arrowprops=dict(facecolor='black', width=0.03,headlength=5.0,headwidth=5.0))
ax1.tick_params(axis='both', which='major', labelsize=tp_fs)
ax2.tick_params(axis='both', which='major', labelsize=tp_fs)
ax3.tick_params(axis='both', which='major', labelsize=tp_fs)

xlim = [-4,4]
ylim = [-2,2]
fname =  'RP_PSOS_x_px_{}_{}_N_50_filtered_gr_nbeads_{}_seed_{}.txt'.format(potkey,Tkey,8,0)
ext = rpext+fname
data = read_1D_plotdata(ext)
X = data[:,0]
PX = data[:,1]
ax2.scatter(X,PX,s=ms/3,color='darkcyan',rasterized=True)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_xlabel(r'$x$',fontsize=xl_fs)
ax2.set_ylabel(r'$p_x$',fontsize=yl_fs)

fname =  'RP_PSOS_x_px_{}_{}_N_50_filtered_ls_nbeads_{}_seed_{}.txt'.format(potkey,Tkey,8,0)
ext = rpext+fname
data = read_1D_plotdata(ext)
X = data[:,0]
PX = data[:,1]
ax3.scatter(X,PX,s=ms,color='goldenrod',rasterized=True)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_xlabel(r'$x$',fontsize=xl_fs)
#ax3.set_ylabel(r'$p_x$',fontsize=15)
ax3.set_yticks([])

#fig.set_size_inches(3, 4)
#fig.savefig('/home/vgs23/Images/PSOS_gyr_D3.pdf'.format(g), dpi=400,bbox_inches='tight',pad_inches=0.0)

fig.set_size_inches(5, 5)
fig.savefig('/home/vgs23/Images/PSOS_gyr_thesis.pdf'.format(g), dpi=400,bbox_inches='tight',pad_inches=0.0)


#fig.savefig('/home/vgs23/Images/FIG_4_D1.png'.format(g), dpi=400,bbox_inches='tight',pad_inches=0.0)


plt.show()
