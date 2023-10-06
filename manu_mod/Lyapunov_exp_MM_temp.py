import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import rc
import matplotlib.ticker as mticker
import matplotlib
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
import os
from matplotlib import container
from matplotlib.ticker import FormatStrFormatter
#plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.serif':'Times New Roman'})
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})

path = os.path.dirname(os.path.abspath(__file__))
rpext = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles/'
qext = '/home/vgs23/PISC/examples/2D/quantum/Datafiles/'
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

nbeads=32


ms = 19 #14
xl_fs = 10#14
yl_fs = 10
tp_fs = 9 #12

rpc = 'olivedrab'
qc = 'darkorange'#'orangered'
Cc = 'slateblue'

fig,ax = plt.subplots()

	
if(1):
	slope_arr  = []
	ebar_arr = []
	T_arr = []
	
	times_arr = [0.8,0.95,1.2,1.4,1.8,2.2,2.6,3.0]
	color_arr =['r','g','b','k','c','orange','m','y'] 
	ext_arr=[[3.5,5.0],[3.1,4.1],[3.0,4.0],[3.0,4.0],[3.0,4.0],[3.0,4.0],[3.0,4.0],[3.0,4.0]]

	for times,c,rang in zip(times_arr,color_arr,ext_arr):
		Tkey = 'T_{}Tc'.format(times)
		ext = 'RPMD_thermal_OTOC_{}_{}_nbeads_{}_dt_{}'.format(potkey,Tkey,16,0.002)
		ext =rpext+ext
		data = read_1D_plotdata('{}.txt'.format(ext))
		#plot_1D(ax,ext,label=Tkey,color=c, log=True,linewidth=1.5)
		slope, ic, t_trunc, OTOC_trunc,V = find_OTOC_slope(ext,rang[0],rang[1],witherror=True,return_cov=True)
		#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')
		rpslope = np.around(abs(slope),2)
		slope_arr.append(rpslope)
		T_arr.append(times)
		error = np.around(V[0,0]**0.5,2)
		ebar_arr.append(error)
		ax.errorbar(times,rpslope,yerr=error,fmt='o',ms=4.5,color=rpc,capsize=2.0,label='$RPMD$',zorder=4)
		#ms 4 for thesis
		print(Tkey,rpslope,error )	
	#plt.show()	
	
	ax.plot(T_arr,slope_arr,linestyle='-',color=rpc)	
	#plt.show()	
	
if(1):
	ext = 'RPMD_Lyapunov_exponent_{}_nbeads_{}_ext_2'.format(potkey,nbeads) #
	ext = rpext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	T_arr = data[:,0]
	lamda_arr = data[:,1]

	ext = 'Quantum_Lyapunov_exponent_{}_ext_2'.format(potkey)
	ext = qext+ext
	data = read_1D_plotdata('{}.txt'.format(ext))
	Tq_arr = data[:,0]
	lamdaq_arr = data[:,1]

	#ax.scatter(T_arr,lamda_arr,marker='x',s=ms,color='sienna',zorder=4)
	#ax.plot(T_arr,lamda_arr,color='sienna',zorder=4)

	#ax.plot([0.95,1.0],[lamda_arr[-1],2.0],color='sienna', ls='--')
	#ax.plot([1.0,1.05],[2.0,2.0],color='sienna',ls='--')
		
	ax.scatter(Tq_arr, lamdaq_arr,marker='o',s=ms,color=qc,label='$Quantum$',zorder=4)
	ax.plot(Tq_arr,lamdaq_arr,linestyle='-',color=qc)	
	ax.axhline(y=2.0,xmin=0.0, xmax = 1.0,linestyle='--',color=Cc)
	ax.axvline(x=1.0,ymin=0.0, ymax = 1.0,linestyle='-',color='gray')

	#ax.annotate('Classical', xy=(0.5,0.5), xycoords='axes fraction', xytext=(0.5,0.67) , textcoords ='axes fraction',fontsize=10)#15)
	ax.annotate('$\lambda(T)=2\pi \: k_B T/\hbar$',xy=(0.27,0.75), xycoords='axes fraction', xytext=(0.31,0.82), color='r', textcoords= 'axes fraction', fontsize=xl_fs)
		
	T_ext = np.arange(1.05,3.06,0.2)
	T_ext1 = np.array([1.8,2.2,2.6,3.0])
	#ax.scatter(T_ext, 2.0*np.ones_like(T_ext), marker='x',s=ms,color='sienna',zorder=5)
	
	ax.scatter(T_ext1, 2.0*np.ones_like(T_ext1), label='$Classical$', marker='o',s=ms,color=Cc,zorder=5)
	ax.plot(T_ext1, 2.0*np.ones_like(T_ext1),color=Cc,zorder=4)
	
	ax.scatter(1.0, 2.0, facecolors='none', edgecolors='sienna',s=ms,zorder=3)

	#ax.plot(T_ext, 2.0*np.ones_like(T_ext), color='sienna', zorder=5)

	ax.set_xlim([0.73,3.05])
	ax.set_ylim([0.48,3.1])	
	T_arr = np.linspace(0.7,1.71,1000) #
	ax.plot(T_arr, 2*np.pi*T_arr*Tc,color='r')#'k',ls=':')

	ax.set_xlabel(r'$T/T_c$',fontsize=xl_fs)
	ax.set_ylabel(r'$\lambda(T)$',fontsize=xl_fs)

	xticks = np.arange(1.0,3.01,0.5) #
	yticks = np.arange(1.0,3.01)
	ax.set_xticks(xticks)
	ax.set_yticks(yticks)
	
	#for label in ax.xaxis.get_ticklabels()[1::2]:
	#	label.set_visible(False)	
	
	ax.tick_params(axis='both', which='major', labelsize=tp_fs)
	#plt.title('Ring-polymer Lyapunov exponent as a function of temperature')

	#plt.legend()
	#plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
	#fig.set_size_inches(2.5, 2)
	#fig.savefig('/home/vgs23/Images/RP_lambda_D2.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	
	#handles, labels = ax.get_legend_handles_labels()
	#handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
	#ax.legend(handles, labels,loc='lower right',fontsize=tp_fs-1.5)
	
	fig.set_size_inches(2.2,2)#4, 2.5)
	#fig.savefig('/home/vgs23/Images/RP_lambda_Leverhulme.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)	
	fig.savefig('/home/vgs23/Images/RP_lambda_D3.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)	
	

	#plt.show()

