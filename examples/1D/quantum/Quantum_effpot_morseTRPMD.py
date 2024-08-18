import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import mildly_anharmonic, Veff_classical_1D_LH, morse, anharmonic, polynomial
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import matplotlib
import os 
from PISC.utils.plottools import plot_1D
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import scipy
from scipy.optimize import curve_fit
from Truncated_pes import truncate_pes
from Distribution_plotter2 import plot_dist

ngrid = 801

#--------------------------------------------
#Morse
lb = -0
ub = 10.0
dx = (ub-lb)/(ngrid-1)
qgrid = np.linspace(lb,ub,ngrid)

m = 1741.1
cm1toau = 219474.63
we = 3737.76/cm1toau
xe = 84.881/(cm1toau*we)
req = 1.832

alpha = np.sqrt(2*m*we*xe)
D = we/(4*xe)
w = np.sqrt(2*D*alpha**2/m)

#print('D',D,'alpha',alpha,'w',w)
pes = morse(D,alpha,req) 
trunc = False#True
if trunc:
    pes = truncate_pes(pes,ngrid,req-0.8,req+0.8,q0=req)
    imfile = 'trunc_morse_trpmd.pdf'
    llim_list =  np.array([1.4,1.4,1.4,1.4]) 
    ulim_list = np.array([2.6,2.6,2.6,2.6]) 
else:
    imfile = 'morse_trpmd.pdf'
    llim_list = np.flip([1.4,1,0.4,0.4])
    ulim_list = np.flip([2.6,3,3.6,3.6])

#--------------------------------------------
tol = 1e-4

Tlist = np.flip([1, 100, 300, 5000])
K2au = 315775.13
betalist = [1/T*K2au for T in Tlist]

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

#--------------------------------------------
fig, ax = plt.subplots(2,2,gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

#plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,renorm='NCF',tol=tol,TinKlist=Tlist)

plot_dist(fig,ax,llim_list,ulim_list,betalist,qgrid,vals,vecs,pes,m,exponentiate=True,tol=tol,TinKlist=Tlist)

fig.set_size_inches(4,4)
#fig.savefig(imfile,dpi=400,bbox_inches='tight')#,pad_inches=0.0)
plt.show()

exit()
