import numpy as np
import matplotlib
import scipy
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import mildly_anharmonic, Veff_classical_1D_LH, morse, anharmonic, polynomial
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from Truncated_pes import truncate_pes
from Distribution_plotter import plot_dist
import os 

ngrid = 801

#-------------------------------------------
#Ceriotti's anharmonic potential
lb = -5
ub = 20
dx = (ub-lb)/(ngrid-1)
qgrid = np.linspace(lb,ub,ngrid)

m = 1836
cm1toau = 1/219474.63
K2au = 0.000003167
k= 1/1.8897 # 1 Angstrom in a.u.
K2au = 0.000003167

wcm = 1000
omega = wcm*cm1toau # cm^-1 to a.u.

pes = anharmonic(m,omega,k)
trunc = False#True
if trunc:
    pes = truncate_pes(pes,ngrid,-5,5)
    llim_list = [-2,-2.5,-3,-5]
    ulim_list = [5,5,5,8]
else:
    llim_list = [-2,-2.5,-3,-5]
    ulim_list = [5,5,5,8]

#-------------------------------------------
tol = 1e-7

TinKlist = np.array([10,100,1000,10000])
Tlist = TinKlist*K2au
betalist = 1/Tlist

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

#-------------------------------------------
fig, ax = plt.subplots(2,2,gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,renorm='NCF',tol=tol,TinKlist=TinKlist)

fig.set_size_inches(4,4)
imfile = 'ceriotti.pdf'
fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)

plt.show()
#-------------------------------------------
