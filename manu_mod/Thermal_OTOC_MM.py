import numpy as np
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os
import matplotlib 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
import matplotlib.patches as mpatches
#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman' })
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

path = os.path.dirname(os.path.abspath(__file__))
Cext = '/home/vgs23/PISC/examples/2D/classical/Datafiles/'
qext = '/home/vgs23/PISC/examples/2D/quantum/Datafiles/'
rpext = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles/'

m=0.5

lamda = 2.0
g = 0.08

Vb =lamda**4/(64*g)
Tc = lamda*0.5/np.pi

D = 3*Vb
alpha = 0.382

print('Vb', Vb)
z = 1.0	

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

fig,ax = plt.subplots(1,3)

rpc = 'limegreen'#'olivedrab'
qc = 'orange'#'orangered'
Cc = 'darkslateblue'

xl_fs = 12.0
yl_fs = 12.0
tp_fs = 12
le_fs = 12
ti_fs = 11

xticks = np.arange(0.0,5.01)

### Temperature: 10 Tc -----------------------------------------------------------
	
times = 3.0#1.0
T_au = times*Tc 
beta = 1.0/T_au 
Tkey = 'T_{}Tc'.format(times)

lwd=1.6
#Quantum
ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,40,70)
ext =qext+ext
data = read_1D_plotdata('{}.txt'.format(ext))
tarr = data[:,0]
Carr = data[:,1]
plot_1D(ax[0],ext,label='Quantum',color=qc, log=True,linewidth=lwd)
print('Quantum')
slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,0.85,1.6)
qslope = np.around(abs(slope),2)
ax[0].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

#RPMD
ext = 'RPMD_thermal_OTOC_{}_{}_nbeads_{}_dt_{}'.format(potkey,Tkey,16,0.002)
ext =rpext+ext
data = read_1D_plotdata('{}.txt'.format(ext))
tarr = data[:,0]
Carr = data[:,1]
plot_1D(ax[0],ext,label='RPMD',color=rpc, log=True,linewidth=lwd,plot_error=False)
print('RPMD')
slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.0,3.0,witherror=True)
rpslope = np.around(abs(slope),2)
#ax[0].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

#Classical
ext = 'Classical_thermal_OTOC_{}_{}_dt_{}'.format(potkey,Tkey,0.002)
ext =Cext+ext
data = read_1D_plotdata('{}.txt'.format(ext))
tarr = data[:,0]
Carr = data[:,1]
print('Classical')
plot_1D(ax[0],ext,label='Classical',color=Cc, log=True,linewidth=lwd)
slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.0,3.0,witherror=True)
Cslope = np.around(abs(slope),2)
ax[0].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

ax[0].set_xlabel(r'$t$',fontsize=xl_fs)
ax[0].set_ylabel(r'$ln \; C_T(t)$',fontsize=yl_fs)
ax[0].set_xticks(xticks)
ax[0].set_title(r'$T=3T_c$', fontsize=ti_fs, x=0.18,y=0.85)
ax[0].annotate(r'$\lambda_q={}$'.format(qslope),xy=(0.1,0.85), xytext=(0.025,0.75), xycoords='axes fraction',fontsize=ti_fs-1)
ax[0].annotate(r'$\lambda_{{RPMD}} ={}$'.format(rpslope),xy=(0.1,0.85), xytext=(0.025,0.65), xycoords='axes fraction',fontsize=ti_fs-1)
ax[0].annotate(r'$\lambda_{{class}} ={}$'.format(Cslope),xy=(0.1,0.85), xytext=(0.025,0.55), xycoords='axes fraction',fontsize=ti_fs-1)



### Temperature: 0.95 Tc ----------------------------------------------------------
	
times = 0.95
T_au = times*Tc 
beta = 1.0/T_au 
Tkey = 'T_{}Tc'.format(times)

yticks = np.arange(-2,3.01)
yticksf = [f'{x}'.replace('-', '\N{MINUS SIGN}') for x in yticks]

#Quantum	
print('Quantum')
ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,20,50)
ext =qext+ext
data = read_1D_plotdata('{}.txt'.format(ext))
tarr = data[:,0]
Carr = data[:,1]
plot_1D(ax[1],ext,label='Quantum',color=qc, log=True,linewidth=lwd)
slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.25,2.1)
qslope = np.around(abs(slope),2)
ax[1].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

#RPMD
print('RPMD')
ext = 'RPMD_thermal_OTOC_{}_{}_nbeads_{}_dt_{}'.format(potkey,Tkey,16,0.002)
ext =rpext+ext
data = read_1D_plotdata('{}.txt'.format(ext))
tarr = data[:,0]
Carr = data[:,1]
plot_1D(ax[1],ext,label='RPMD',color=rpc, log=True,linewidth=lwd)
slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.1,4.1,witherror=True)
rpslope = np.around(abs(slope),2)
ax[1].plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')

#Classical
ext = 'Classical_thermal_OTOC_{}_{}_dt_{}'.format(potkey,Tkey,0.002)
ext =Cext+ext
plot_1D(ax[1],ext,label='Classical',color=Cc, log=True,linewidth=lwd)

ax[1].set_xlabel(r'$t$',fontsize=xl_fs)
ax[1].set_yticks(yticks)
ax[1].set_xticks(xticks)
ax[1].set_title(r'$T=0.95T_c$',fontsize=ti_fs, x=0.24,y=0.85)

ax[1].annotate(r'$\lambda_q={}$'.format(qslope),xy=(0.1,0.85), xytext=(0.025,0.75), xycoords='axes fraction',fontsize=ti_fs-1)
ax[1].annotate(r'$\lambda_{{RPMD}} ={}$'.format(rpslope),xy=(0.1,0.85), xytext=(0.025,0.65), xycoords='axes fraction',fontsize=ti_fs-1)
### Temperature: 0.7 Tc -----------------------------------------------------------
	
times = 0.6
T_au = times*Tc 
beta = 1.0/T_au 
Tkey = 'T_{}Tc'.format(times)

yticks = np.arange(-2,2.01)
yticksf = [f'{x}'.replace('-', '\N{MINUS SIGN}') for x in yticks]

#Quantum	
ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,20,50)
extq =qext+ext
plot_1D(ax[2],extq,label='Quantum',color=qc, log=True,linewidth=lwd)

#RPMD
ext = 'RPMD_thermal_OTOC_{}_{}_nbeads_{}_dt_{}'.format(potkey,Tkey,32,0.002)
extrp =rpext+ext
plot_1D(ax[2],extrp,label='RPMD',color=rpc, log=True,linewidth=lwd)

#Classical
ext = 'Classical_thermal_OTOC_{}_{}_dt_{}'.format(potkey,Tkey,0.002)
extcl =Cext+ext
plot_1D(ax[2],extcl,label='Classical',color=Cc, log=True,linewidth=lwd)

ax[2].set_xlabel(r'$t$',fontsize=xl_fs)
ax[2].set_xticks(xticks)
ax[2].set_ylim([-3.4,1.5])
#ax[2].set_yticks(yticks,yticksf)
ax[2].set_title(r'$T=0.6T_c$', fontsize=ti_fs,x=0.20,y=0.85)

###----------------------------------------------------------------------------------




###-----------------------------------------------------------------------------------


plt.subplots_adjust(wspace=0.195)
#ax[1].legend()#loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=3)
#ax[0].legend()
#ax[2].legend()
for i in range(3):
	ax[i].tick_params(axis='both', which='major', labelsize=tp_fs)

#fig.set_size_inches(5.3, 2)
#handles, labels = ax.get_legend_handles_labels()
#fig.legend(loc = (0.35, 0.9),ncol=3, fontsize=le_fs)
lines = ax[2].get_lines() #+ ax[2].right_ax.get_lines()


#ax[2].legend(lines, [l.get_label() for l in lines], loc='upper center')

#fig.savefig('/home/vgs23/Images/Thermal_OTOCs_D3.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
#plt.gray()
#plt.show()
fig.set_size_inches(8, 3)
#handles, labels = ax[2].get_legend_handles_labels()
#by_label = dict(zip(labels, handles))
#handles, labels = ax[2].get_legend_handles_labels()
fig.legend(lines,[l.get_label() for l in lines], loc = (0.22, 0.9),ncol=3, fontsize=le_fs)

fig.savefig('/home/vgs23/Images/Thermal_OTOCs_thesis.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
plt.show()
