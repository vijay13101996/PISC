import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from matplotlib import pyplot as plt
from PISC.engine import OTOC_f_BH2
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.utils.misc import find_OTOC_slope
import os
import time

path = os.path.dirname(os.path.abspath(__file__))	
start_time = time.time()

"""
This code constructs the Hamiltonian matrix for the 2-site Bose-Hubbard 
model in the number-state basis, diagonalises the same and computes the
eigenstate energies and wavefunctions in that basis. The OTOC is also 
computed from these by augmenting the existing quantum OTOC code.
"""

#Model parameters
U = 1.0#1e-2#-1.0/20
K = 0.04/U#3.02e-4#-0.4
N = 100#100#20
eps1 = 0.0
eps2 = 0.0#e-3

print('lamda', np.sqrt(N*U*K))

E0 = U*N**2/4 - U*N/2

potkey = 'BH2_U_{}_K_{}_N_{}'.format(U,K,N)

lamda = np.sqrt(N*U*K)
Vb = N*K

print('lambda, Eb', lamda, Vb)

beta_c = 2*np.pi/lamda
beta = beta_c#362.0#362.0

print('kbT/Vb', 1/(beta*Vb))
#print('E_0', E0)

def prefactor(l):
	return np.sqrt((N-l)*(l+1))

H = np.zeros((N+1,N+1))
for l in range(N+1):
	H[l,l] = 0.5*U*(2*l**2 - 2*l*N + N**2 - N) #+ eps1*l + eps2*(N-l)
	if(l>0):
		H[l,l-1] = -0.5*K*prefactor(l-1)
	if(l<N):
		H[l,l+1] = -0.5*K*prefactor(l)

vals, vecs = eigh(H)#,k=N)
vals = vals-vals[0]#E0
#print('E gs, sep',-N*K/2,vals[0])#np.around(vals[:100],4))
Z = np.sum(np.exp(-beta*vals))
#print('Z', Z)
#print('H',H)
print('vals', vals[:8])

#plt.scatter(range(len(vals)),2*vals/(N*K))
#plt.show()

if(1):
	# Dummy variables (needs to be passed thanks to poor code structure)
	m = 1
	dx = 0.1
	dy = 0.1
	x_arr = np.linspace(0,1,100) 
	#-------------------------------------------------------------------

	n_eigen = 60#N+1 #- 5
	basis_N = N+1 
	k_arr = range(N+1)
	m_arr = range(N+1)
	
	t_arr = np.linspace(0.0,5.0,1000)
	OTOC_arr = np.zeros_like(t_arr)+0j 

	reg = 'Standard'
	corr='xxC'#'qq1'#'xxC'
	
	if(reg=='Kubo'):
		OTOC_arr = OTOC_f_BH2.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,corr,'kubo',OTOC_arr)
	elif(reg=='Standard'):	
		OTOC_arr = OTOC_f_BH2.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr)
	elif(reg=='Symmetric'):
		OTOC_arr = OTOC_f_BH2.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)

	fname = 'Quantum_{}_{}_{}_beta_{}_neigen_{}_basis_{}'.format(reg,corr,potkey,beta,n_eigen,basis_N)
	store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))

	print('time', time.time() - start_time)
	print('CT', OTOC_arr[0])
	plt.plot(t_arr,np.log(abs(OTOC_arr)))
	

	ext = path+'/Datafiles/'+fname
	
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,0.5,1.5)
	print('slope',slope)
	plt.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
	

	plt.show()	


#plt.scatter(eps*np.ones_like(vals),vals/N,color='k',s=2)
#plt.show()

