import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import Veff_classical_1D_LH, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from Distribution_plotter import plot_dist

ngrid = 801

#--------------------------------------------------
lb = -10
ub = 10.0
dx = (ub-lb)/(ngrid-1)
qgrid = np.linspace(lb,ub,ngrid)

m = 0.5

lamda = 2.0
g = 0.08
pes = double_well(lamda,g)

imfile = 'DW_anharm.pdf'
llim_list = [-5,-5,-5,-5]#[-10,-10,-10,-10]#
ulim_list = [5,5,5,5]#[10,10,10,10]#

#--------------------------------------------------
tol = 1

betalist = [0.1,1.0,10.0,100.0]

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

#--------------------------------------------------
fig, ax = plt.subplots(2,2,gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

#plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,renorm='NCF',tol=tol)

plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,tol=tol)


fig.set_size_inches(4,4)
imfile = 'morse_h_anh.pdf'
#fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)
plt.show()

