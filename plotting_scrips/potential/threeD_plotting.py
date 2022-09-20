import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
sys.path.insert(0,"/home/lm979/Desktop/PISC/Programs_Lars")
import numpy as np
from PISC.dvr.dvr import DVR2D, DVR1D
from Programs_Lars.mylib.twoD import DVR2D_mod
from PISC.potentials.Quartic_bistable import quartic_bistable
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_2D_omp_updated
from Programs_Lars.mylib.twoD import PY_OTOC2D
from matplotlib import pyplot as plt
import os 
import time 
from Programs_Lars.mylib.testing import Check_DVR 
from Programs_Lars import Functions_DW_Morse_coupling as DMCC

from plt_util import prepare_fig, prepare_fig_ax
##########Potential#########
#quartic bistable
lamda = 2.0    
D=9.375#3Vb
g = 0.08
#Grid
Lx=4.5
lbx = -Lx
ubx = Lx
lby = -1.9#Morse, will explode quickly #-2 not enough for alpha 0.35  
uby = 12#Morse, will explode quickly
m = 0.5
ngrid = 2000#140#convergence
ngridx = ngrid
ngridy = ngrid
xg = np.linspace(lbx,ubx,ngridx)#ngridx+1 if "prettier"
yg = np.linspace(lby,uby,ngridy)
xgr,ygr = np.meshgrid(xg,yg)
#parameters
N_trunc=100
n_eig_tot=150
T_c = 0.5*lamda/np.pi
times=1.0
T=times*T_c
alpha=0.38
alpha_range=(alpha,)
z_range=(0,0.5,1.0,1.5)
#z_range=(0.0,)
for z in z_range:
	pes = quartic_bistable(alpha,D,lamda,g,z)
	from mpl_toolkits import mplot3d
	fig = prepare_fig(tex=True)
	ax = plt.axes(projection='3d')
	V=pes.potential_xy(xgr,ygr)
	#ax.contour3D(xgr, ygr, V, 20, cmap='viridis')#very useful, 3D conture
	#ax.plot_wireframe(xgr, ygr, V, color='black')
	ax.plot_surface(xgr, ygr, V, rstride=60, cstride=60, cmap='viridis')#'''viridis','hot', edgecolor='none')
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$y$')
	ax.set_zlabel(r'$V(x,y)$')
	file_dpi=600
	fig.savefig('plots/3D_z_%.2f.pdf'%z,format='pdf',bbox_inches='tight', dpi=file_dpi)
	fig.savefig('plots/3D_z_%.2f.png'%z,format='png',bbox_inches='tight', dpi=file_dpi)
plt.show()
