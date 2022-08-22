import numpy as np
from PISC.potentials.Quartic_bistable import quartic_bistable
from Quantum_OTOC_plotfunctions import plot_wf, plot_bnm, plot_Cn, plot_kubo, cplot_kubo, cplot_stan
from matplotlib import pyplot as plt
import os 
import time 
import ast

start_time = time.time()

m=0.5

lamda = 2.0
g = 0.08

Vb =lamda**4/(64*g)

D = 3*Vb
alpha = 0.382

print('Vb', Vb)
z = 1.0#2.3	

Tc = lamda*0.5/np.pi
times = 10.0#1.0
T_au = times*Tc 
beta = 1.0/T_au 
	
path = os.path.dirname(os.path.abspath(__file__))	
	
fig,ax = plt.subplots()

alpha_list1 = [0.382]#[0.24,0.39,0.55,1.147]#,0.382,0.52,1.147]#,0.55]
alpha_list2 = [0.382]#,0.45,0.52]#,0.53,0.54,0.55]#,0.6]
alpha_list3 = [0.52]
alpha_list4 = [1.147]
	
if(1): # b_nm,C_n for different alpha
	n=8#8
	M=2
	for alph in alpha_list1:
		print('alpha',alph)	
		#plot_wf(path,alph,D,lamda,g,z,n)
		#plot_bnm(path,alph,D,lamda,g,z,n,M,ax,lbltxt=r'$\alpha={}$'.format(alph) )
		#plot_Cn(path,alph,D,lamda,g,z,n,ax,lbltxt=r'$\alpha={}$'.format(alph))
		plot_kubo(path,alph,D,lamda,g,z,beta,ax,lbltxt=r'$\alpha={}$'.format(alph),basis_N=100,n_eigen=50)
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
	plt.title(r'Kubo OTOCs for different $\alpha$ at $T=T_c$')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$C_T(t)$')

if(0): # b_nm for different n,m's
	for M in [2]:#range(5):
		for n in range(20):	
			#plot_wf(path,0.2,D,lamda,g,0.0,n)	
			plot_bnm(path,alpha,D,lamda,g,z,n,M,ax,lbltxt=r'$n,m={},{}$'.format(n,M))

plt.legend()
plt.show()
