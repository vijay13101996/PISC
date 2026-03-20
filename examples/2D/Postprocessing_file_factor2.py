import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope,estimate_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import read_1D_plotdata
from PISC.potentials import quartic_bistable
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False
dim=2

### Double well
lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

z = 0.0
 
Tc = 0.5*lamda/np.pi
times = 3.0
T = times*Tc
beta=1/T

Tkey = 'T_{}Tc'.format(times)

m = 0.5
N = 1000
dt_therm = 0.05
dt = 0.002
time_therm = 50.0
time_total = 5.0#5.0

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
pes = quartic_bistable(alpha,D,lamda,g,z)

tarr = np.arange(0.0,time_total,dt) + 0j
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))   
#path = '/scratch/vgs23/PISC/examples/2D/'
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'fd_OTOC'
syskey = 'Papageno'

nbeads=1
beadkey = 'nbeads_{}_'.format(nbeads)
potkey_ = potkey+'_'

methodkey = 'RPMD'
enskey= 'thermal'#'const_q'#'thermal'

fig,ax =plt.subplots()

def plot(z,enskey,ax,c,ls,sl=False,ti=2.4,te=3.4,label=None):
    potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z) 
    ext = rpext + 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,'fd_OTOC',potkey,Tkey,1,dt)
    data = np.loadtxt(ext+'.txt',dtype=complex)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,ti,te)
    if(label is not None):
        ax.plot(data[:,0],np.log(data[:,1]),label=label,lw=1,color=c,ls=ls)
    else:
        ax.plot(data[:,0],np.log(data[:,1]),lw=1,color=c,ls=ls)
    if(sl):
        ax.plot(t_trunc, slope*t_trunc+ic,color='k',lw=1.5)

def quant_plot(z,ax,c,ls,n_eigen=40,basis_N=70):
    potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
    ext = qext + 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,n_eigen,basis_N)
    data = np.loadtxt(ext+'.txt',dtype=complex)
    slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1,1.5)
    ax.plot(data[:,0],np.log(data[:,1]),label='z={}, $\lambda$={} '.format(z,np.around(slope,2)),lw=1,color=c,ls=ls)
    ax.plot(t_trunc, slope*t_trunc+ic,color='k',lw=1.5)

#plot(0.0,'thermal',ax,'r','--')#[1])
plot(0.0,'const_q',ax,'g','--',sl=True,ti=2,te=3)
#plot(0.0,'const_qp',ax,'b','--')#[1])
#quant_plot(2.0,ax,'k','--')#[1])

plot(2.0,'thermal',ax,'r','-',sl=True,ti=2,te=3,label=r'$C_T(t)$')
plot(2.0,'const_q',ax,'g','-',sl=True,ti=2,te=3,label=r'$C_0(t;T)$')
plot(2.0,'const_qp',ax,'b','-',sl=True,ti=2,te=3,label=r'$C_{mc}(t)$')#[0])
#quant_plot(0.0,ax,'k','-')#[0])

ax.set_xlim([0,4])
ax.set_ylim([-2,18])
ax.tick_params(axis='both', which='major', labelsize=10)

ax.legend(ncol=3,fontsize=9,frameon=False,loc=(-0.07,1.02))
fig.set_size_inches(3,3)
plt.show()
    
 


