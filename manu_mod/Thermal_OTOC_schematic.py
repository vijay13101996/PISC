import numpy as np
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os
import matplotlib 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope

#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman' })
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})

matplotlib.rcParams['axes.unicode_minus'] = False

path = os.path.dirname(os.path.abspath(__file__))
Cext = '/home/vgs23/PISC/examples/1D/classical/Datafiles/'
qext = '/home/vgs23/PISC/examples/1D/quantum/Datafiles/'

m=0.5

lamda = 2.0
g = 0.02

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

Vb =lamda**4/(64*g)
Tc = lamda*0.5/np.pi

times = 20.0
T_au = times*Tc 
beta = 1.0/T_au 
Tkey = 'T_{}Tc'.format(times)

fig,ax = plt.subplots()

qc = 'orangered'
Cc = 'slateblue'
lwd = 2.0

xl_fs = 12
yl_fs = 12
tp_fs = 9.5
le_fs = 10
ti_fs = 8

ext = 'Classical_thermal_OTOC_{}_{}_dt_0.005'.format(potkey,Tkey)
ext = Cext+ext
data = read_1D_plotdata('{}.txt'.format(ext))
tarr = data[:,0]
Carr = data[:,1]
print('Classical')
plot_1D(ax,ext,label='$Classical$',color=Cc, log=True,linewidth=lwd)

ext = 'Quantum_Kubo_OTOC_{}_{}_basis_{}_n_eigen_{}'.format(potkey,Tkey,140,90)	
ext =qext+ext
data = read_1D_plotdata('{}.txt'.format(ext))
tarr = data[:,0]
Carr = data[:,1]
plot_1D(ax,ext,label='$Quantum$',color=qc, log=True,linewidth=lwd)

ax.set_xlim([0.0,4.2])
ax.set_ylim([-1.0,7.])

ax.set_xticks([])
ax.set_yticks([])

ax.set_xlabel('$t$',fontsize=xl_fs)
ax.set_ylabel('$ln \; O(t)$', fontsize=yl_fs)

ax.annotate(r'$Transient$',xy=(0.1,0.1), xytext=(0.001,0.185), xycoords='axes fraction',fontsize=ti_fs)
ax.annotate(r'$Exponential$',xy=(0.15,0.15), xytext=(0.325,0.425), xycoords='axes fraction',rotation=45,fontsize=ti_fs)
ax.annotate(r'$Saturation$',xy=(0.1,0.1), xytext=(0.7,0.61), xycoords='axes fraction',fontsize=ti_fs)

fig.set_size_inches(3.0, 3.0)
ax.legend(loc = 'upper left', fontsize=le_fs)

fig.savefig('/home/vgs23/Images/Thermal_OTOCs_schematic_Leverhulme.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
plt.show()


plt.show()
