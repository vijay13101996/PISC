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
ngrid = 140#140#convergence
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
	#DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
	#vals,vecs=DMCC.get_EV_and_EVECS(alpha,D,lamda,g,0,ngrid,n_eig_tot,lbx,ubx,lby,uby,m,ngridx,ngridy,pes,DVR)
	#DMCC.plot_pot_E_Ecoupled_and_3D(pes,ngridx,ngridy,lbx,ubx,lby,uby,m,xg,yg,vals,z,plt_V_and_E=True,vecs=vecs)
	DMCC.Compare_DW_and_Morse_energies(lbx=-8,ubx=8,lby=-4,uby=12,m=0.5,ngridx=200,ngridy=200,lamda = 2.0,D=D, g = 0.08,plotting=True,alpha_range = (0.38,),save_fig=True)

plt.show(block=True)

