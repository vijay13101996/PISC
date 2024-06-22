import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr

dim = 1

#Double Well potential
lamda = 2.0
g = 0.08#8

Tc = lamda*(0.5/np.pi)
times = 20.0
T = times*Tc
Tkey = 'T_{}Tc'.format(times)

m = 0.5

potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)

N = 1000
dt = 0.002

tarr = np.arange(0,5,dt)

#Path extensions
path = '/home/vgs23/PISC/examples/1D/'#
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

fig,ax = plt.subplots()

methodkey = 'Classical'
enskeys = ['stable_manifold','const_qp','const_q']
corrkey = 'OTOC_qq'#'qq_TCF'#'singcomm'#'OTOC'

td = 2.
tu = 3.

Am_ApAp = '$\int dq\:dp\: e^{- \\beta H} \delta(A_-) |\\frac{\partial A_+(t)}{\partial A_+(0)}|^2$'
Am_qq = '$\int dq\:dp\: e^{- \\beta H} \delta(A_-) |\\frac{\partial q_t}{\partial q_0}|^2$'

qp_ApAp = '$\int dq\:dp\: e^{- \\beta H} \delta(q) \delta(p) |\\frac{\partial A_+(t)}{\partial A_+(0)}|^2$'
qp_qq = '$\int dq\:dp\: e^{- \\beta H} \delta(q) \delta(p) |\\frac{\partial q_t}{\partial q_0}|^2$'

q_ApAp = '$\int dq\:dp\: e^{- \\beta H} \delta(q) |\\frac{\partial A_+(t)}{\partial A_+(0)}|^2$'
q_qq = '$\int dq\:dp\: e^{- \\beta H} \delta(q) |\\frac{\partial q_t}{\partial q_0}|^2$'

leg_ApAp = [Am_ApAp,qp_ApAp,q_ApAp]
leg_qq = [Am_qq,qp_qq,q_qq]

if corrkey == 'OTOC_qq':
    leg = leg_qq
elif corrkey == 'OTOC_ApAp':
    leg = leg_ApAp

for enskey,c,leg in zip(enskeys,['r','b','g'],leg):
    ext = Cext + 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
    ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color=c)
    plot_1D(ax,ext, label=leg,color=c, log=True,linewidth=1)

Tkey = 'T_{}Tc'.format(0.95)
ext = rpext + 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,'OTOC',potkey,Tkey,32,dt)
plot_1D(ax,ext, label='RPMD',color='k', log=True,linewidth=1)
slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.5,3.5)
ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

ext = cext + 'CMD_{}_{}_{}_{}_nbeads_{}_dt_{}_gamma_{}'.format(enskey,'OTOC',potkey,Tkey,8,0.01,8)
#plot_1D(ax,ext, label='CMD',color='m', log=True,linewidth=1)
#slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,td,tu)
#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='m')


#plt.plot(tarr,np.log(1 + 2*lamda*tarr + 2*lamda**2*tarr**2),'k--',label='$\sim t^2$')
#plt.plot(tarr, np.log(np.cosh(lamda*tarr)),'k--',label='$\sim \log(\cosh(\lambda t))$')
plt.legend()
plt.show()


