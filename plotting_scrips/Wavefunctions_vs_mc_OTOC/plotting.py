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
##########Potential#########
#quartic bistable
lamda = 2.0    
D=9.375#3Vb
g = 0.08
#Grid
Lx=7.0
lbx = -Lx
ubx = Lx
lby = -4#Morse, will explode quickly #-2 not enough for alpha 0.35  
uby = 10#Morse, will explode quickly
m = 0.5
ngrid = 140#convergence
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
z_range=(0.5,)
contour_eigenstates=range(6)
for z in z_range:
	pes = quartic_bistable(alpha,D,lamda,g,z)
	DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
	vals,vecs=DMCC.get_EV_and_EVECS(alpha,D,lamda,g,z,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
	#DMCC.contour_plots_eigenstates(pes,DVR,lbx,lby,ubx,uby,ngridx,ngridy,z,vecs,alpha,l_range=contour_eigenstates,remove_upper_part=True)
#plt.show(block=True)
#sys.exit()
t_arr,C_n_loop_alpha=DMCC.compute_C_n(alpha_range,z_range,contour_eigenstates,D,lamda,g,ngridx,ngridy,ngrid,lbx,ubx,lby,uby,m,n_eig_tot,N_trunc)
DMCC.plot_C_n_for_contour_eigenstates(alpha_range, z_range,contour_eigenstates,t_arr,  C_n_loop_alpha,N_trunc,log=True,deriv=False)
plt.show(block=True)

