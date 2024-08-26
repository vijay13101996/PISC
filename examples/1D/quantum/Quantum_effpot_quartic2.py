import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import mildly_anharmonic, Veff_classical_1D_LH, morse, anharmonic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from Distribution_plotter2 import plot_dist

ngrid = 801

#Quartic/Harmonic/Mildly anharmonic
lb = -5.
ub = 5.0
dx = (ub-lb)/(ngrid-1)
qgrid = np.linspace(lb,ub,ngrid)

m = 1
w = 0. #quadratic term (frequency)
a = 0.#-0.75 #cubic term
b = 0.25#1 #quartic term
tol = 1e-4

pes = mildly_anharmonic(m,a,b)

renorm = 'harm'
fur_renorm = 'Liu'

imfile = 'quart_{}_{}.png'.format(renorm,fur_renorm)
llim_list = [-5,-3,-2.5,-2]
ulim_list = [5,3,2.5,2]
#--------------------------------------------
tol = 1e-4

betalist = [0.1,1.0,10.0,100.0]

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

#--------------------------------------------
fig, ax = plt.subplots(2,2,gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

#plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,renorm='NCF',tol=tol,TinKlist=Tlist)

plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,tol=tol,renorm=renorm,fur_renorm=fur_renorm)

fig.set_size_inches(4,4)
fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)
plt.show()


