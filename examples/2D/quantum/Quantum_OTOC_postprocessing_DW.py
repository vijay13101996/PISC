import numpy as np
from PISC.potentials import quartic_bistable
from Quantum_OTOC_plotfunctions import plot_wf, plot_bnm, plot_Cn, plot_thermal_OTOC, plot_thermal_TCF, cplot_kubo, cplot_stan
from matplotlib import pyplot as plt
import os 
import time 
import ast
from PISC.utils.misc import find_OTOC_slope
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.plottools import plot_1D

start_time = time.time()

m=0.5

lamda = 2.0
g = 0.08

Vb =lamda**4/(64*g)

D = 3*Vb
alpha = 0.382

z = 1.0#2.3	

Tc = lamda*0.5/np.pi
times = 0.85#1.0
T_au = times*Tc 
beta = 1.0/T_au 
	
path = os.path.dirname(os.path.abspath(__file__))	
	
fig,ax = plt.subplots()

print('ehere')
alpha_list1 = [0.382]#[0.24,0.39,0.55,1.147]#,0.382,0.52,1.147]#,0.55]
alpha_list2 = [0.382]#,0.45,0.52]#,0.53,0.54,0.55]#,0.6]
alpha_list3 = [0.52]
alpha_list4 = [1.147]

if(1): # b_nm,C_n for different alpha
	print('heree')
	#for z in [1.0]:#0.0,0.5,1.0]:
		#print('z',z) 
		#plot_thermal_OTOC(path,alpha,D,lamda,g,z,beta,ax,lbltxt=r'$\z={}$'.format(z),basis_N=50,n_eigen=20,reg='Kubo')

	T_arr = np.arange(0.2,3.0,0.2)
	T_arr = [0.1]#np.concatenate([[0.05],T_arr])
	for times in T_arr:#[1.6,1.8,2.0,2.2,2.4,2.6,2.8]:#[3.0,0.95,0.6]:#np.arange(0.7,1.5,0.1):#[3.0]:#0.6,0.95,10.0]:#[1.0,1.1,1.2,1.3,1.4,1.5]:
		T = times*Tc
		beta = 1/T
		print('T', times,beta)	
		t_arr = np.linspace(0.0,5.0,1000)
		plot_thermal_OTOC(path,alpha,D,lamda,g,z,beta,ax,lbltxt=r'z={}'.format(z),basis_N=70,n_eigen=40,reg='Kubo')
		#plot_thermal_TCF(path,alpha,D,lamda,g,z,beta,ax,lbltxt=r'$T={}Tc$'.format(times),basis_N=100,n_eigen=70,reg='Kubo', t_arr=t_arr,tcftype='qp1')

	n=8#8
	M=2

	#for alph in alpha_list1:
		#print('alpha',alph)	
		#plot_wf(path,alph,D,lamda,g,z,n)
		#plot_bnm(path,alph,D,lamda,g,z,n,M,ax,lbltxt=r'$\alpha={}$'.format(alph) )
		#plot_Cn(path,alph,D,lamda,g,z,n,ax,lbltxt=r'$\alpha={}$'.format(alph))
		#plot_kubo(path,alph,D,lamda,g,z,beta,ax,lbltxt=r'$\alpha={}$'.format(alph),basis_N=100,n_eigen=50)
		#print('time', time.time()-start_time)	
	n=7
	M=3
	#for alph in alpha_list2:
		#plot_wf(path,alph,D,lamda,g,z,n)
		#plot_bnm(path,alph,D,lamda,g,z,n,M,ax,lbltxt=r'$\alpha={}$'.format(alph) )
		#plot_Cn(path,alph,D,lamda,g,z,n,ax,lbltxt=r'$\alpha={}$'.format(alph))
		#plot_kubo(path,alph,D,lamda,g,z,beta,ax,lbltxt=r'$\alpha={}$'.format(alph))
	
	n=7
	M=4
	#for alph in alpha_list3:
		#plot_wf(path,alph,D,lamda,g,z,n)
		#plot_bnm(path,alph,D,lamda,g,z,n,M,ax,lbltxt=r'$\alpha={}$'.format(alph))
		#plot_Cn(path,alph,D,lamda,g,z,n,ax,lbltxt=r'$\alpha={}$'.format(alph))

	n=8
	M=5
	#for alph in alpha_list4:
		#plot_wf(path,alph,D,lamda,g,z,n)
		#plot_bnm(path,alph,D,lamda,g,z,n,M,ax,lbltxt=r'$\alpha={}$'.format(alph))
		#plot_Cn(path,alph,D,lamda,g,z,n,ax,lbltxt=r'$\alpha={}$'.format(alph))

	alpha=0.2
	z=0.0
	n=8
	M=2
	#plot_wf(path,alpha,D,lamda,g,z,n)
	#plot_bnm(path,alpha,D,lamda,g,z,n,M,ax,lbltxt=r'Uncoupled, z=0.0'.format(alpha))
	#plot_Cn(path,0.2,D,lamda,g,z,n,ax,lbltxt=r'$\alpha={}$'.format(alpha))
	#plot_kubo(path,alpha,D,lamda,g,z,beta,ax,lbltxt=r'Uncoupled, z=0.0'.format(alpha),basis_N=50,n_eigen=20)
	#cplot_stan(path,alpha,D,lamda,g,z,beta,ax,lbltxt=r'$\alpha={}$'.format(alpha),basis_N=50,n_eigen=20)
	#plt.title(r'Kubo OTOCs for different $\alpha$ at $T=T_c$')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$C_T(t)$')


if(0): # b_nm for different n,m's
	for M in [3,8,9]:#range(5):
		for n in [3,8,9]:#range(20):	
			#plot_wf(path,alpha,D,lamda,g,z,n)	
			plot_bnm(path,alpha,D,lamda,g,z,n,M,ax,lbltxt=r'$n,m={},{}$'.format(n,M))

if(0):
	def get_cmap(n, name='hsv'):
		'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
		RGB color; the keyword argument name must be a standard mpl colormap name.'''
		return plt.cm.get_cmap(name, n)

	qext = '/home/vgs23/PISC/examples/2D/quantum/Datafiles/'
	potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

	T_arr = np.arange(0.7,1.5,0.1)#[0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.1,1.2,1.3,1.4,1.5]#np.around(np.arange(0.7,0.951,0.05),2)
	T_arr = [T_arr[1],T_arr[2],0.95,T_arr[3],T_arr[5],T_arr[7],1.8,2.2,2.6,3.0]
	cmap = get_cmap(len(T_arr))
	print('cmap')
	lamda_arr=[]
	t_arr = np.linspace(0.0,5.0,1000)
		
	for times,c in zip(T_arr,['k','r','g','b','m','y','c','tomato','olivedrab','slateblue','orangered','khaki','indianred','crimson','r']):
		T = times*Tc
		print('times', times)
		beta = 1/T
		ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,40,70)
		print('ext',ext)
		data = read_1D_plotdata('{}/{}.txt'.format(qext,ext))
		ext =qext+ext
		plot_1D(ax,ext,label='T = {}Tc'.format(np.around(times,2)),color=c, log=True,linewidth=2)
		index = np.argmin(np.abs(np.log(data[10:,1])))
		tst = data[index+10,0]
		print('tst',tst) 
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,tst,tst+0.55)
		lamda_arr.append(slope)
		ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	if(0):
		times = 0.95
		T_arr.append(times)
		T=times*Tc
		beta=1/T
		ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,40,70)
		print('ext',ext)
		ext =qext+ext
		plot_1D(ax,ext,label='T = {}Tc'.format(times),color=c, log=True,linewidth=2)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.2,2.1)
		lamda_arr.append(slope)
		ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
	plt.legend(ncol=2)
	plt.show()

	plt.scatter(T_arr,lamda_arr)
	plt.plot(T_arr,lamda_arr)
	plt.xlabel(r'$T$ (in units of $T_c$)')
	plt.ylabel(r'$\lambda_q$')
	plt.show()
	
	print(T_arr,lamda_arr)	
	store_1D_plotdata(T_arr,lamda_arr,'Quantum_Lyapunov_exponent_{}_ext_2'.format(potkey),'{}/Datafiles'.format(path))

plt.show()
