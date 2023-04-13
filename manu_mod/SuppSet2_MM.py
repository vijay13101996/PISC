import numpy as np
import os
import ast
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Quartic_bistable import quartic_bistable
#from PISC.utils.colour_tools import lighten_color
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import itertools
from PISC.utils.plottools import plot_1D

path = '/home/vgs23/PISC/examples/2D'

qext = '{}/quantum'.format(path)
Cext = '{}/classical'.format(path)

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

tp_fs = 13
xl_fs = 17
yl_fs = 17

le_fs = 12
ti_fs = 12

if(1): #Gather system data to plot
	hbar = 1.0
	m=0.5
	
	w = 0.1	
	D = 9.375#10.0
	alpha = 0.382#35#

	lamda = 2.0
	g = 0.08

	z = 1.0

	Tc = lamda*0.5/np.pi
	T_au = Tc#10.0 
	
	pes = quartic_bistable(alpha,D,lamda,g,z)
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
	with open('{}/Datafiles/Input_log_{}.txt'.format(qext,potkey)) as f:	
		for line in f:
			pass
		param_dict = ast.literal_eval(line)

	lbx = param_dict['lbx']
	ubx = param_dict['ubx']
	lby = param_dict['lby']
	uby = param_dict['uby']
	ngridx = param_dict['ngridx']
	ngridy = param_dict['ngridy']

	print('lb and ub', lbx,ubx,lby,uby)
	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)	
	fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)	# Change ngridx!=ngridy

	vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(qext))
	vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(qext))

fig, ax = plt.subplots(2,2)
ax00 = ax[0,0]
ax01 = ax[0,1]
ax10 = ax[1,0]
ax11 = ax[1,1]

xticks = np.arange(0.0,5.01)
times_arr = [10.0,3.0,0.95,0.6]
lwd = 1.5
for i in range(4):
	if(i==0):
		axi = ax00
		yticks = np.arange(-0.5,3.01,0.5)
	elif(i==1):
		axi = ax01
		yticks = np.arange(-1.0,3.01)
	elif(i==2):
		axi = ax10
		yticks = np.arange(-3.0,4.01)
	else:
		axi = ax11	
		yticks = np.arange(-5.0,4.01)
	T = times_arr[i]*Tc
	beta = 1/T
	Tkey = 'T_{}Tc'.format(times_arr[i])
	for reg,sty in zip(['Kubo','Standard','Symmetric'],['-',':','--']):
		if(times_arr[i]==10.0 or times_arr[i]==3.0):
			ext = 'Quantum_{}_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(reg,potkey,beta,40,70)
		else:
			ext = 'Quantum_{}_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(reg,potkey,beta,20,50)
		ext =qext+'/Datafiles/' +ext
		if(i==0):
			plot_1D(axi,ext,label=reg,color='orangered', log=True,linewidth=lwd,style=sty)
		else:
			plot_1D(axi,ext,label='',color='orangered', log=True,linewidth=lwd,style=sty)
	if(i==2 or i==3):
		axi.set_xlabel(r'$t$',fontsize=xl_fs)
	if(i==0 or i==2):
		axi.set_ylabel(r'$ln \: C_T(t)$',fontsize=yl_fs)
		axi.yaxis.set_label_coords(-0.2,0.5)
	axi.set_yticks(yticks)
	axi.set_xticks(xticks)
	axi.set_title(r'$T={}T_c$'.format(times_arr[i]),fontsize=ti_fs, x=0.22,y=0.88)
		#axi.legend()

for i in range(2):
	for j in range(2):
		ax[i,j].tick_params(axis='both', which='major', labelsize=tp_fs)


fig.subplots_adjust(hspace=0.15,wspace=0.15)
fig.legend(loc = (0.16, 0.93),ncol=3, fontsize=le_fs)
fig.set_size_inches(6,7)
fig.savefig('/home/vgs23/Images/S4_thesis.pdf', dpi=400,bbox_inches='tight',pad_inches=0.0)
plt.show()
