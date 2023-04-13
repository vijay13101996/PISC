import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, morse, quartic, mildly_anharmonic, harmonic1D
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, store_2D_imagedata_column, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
import time
start_time = time.time()

### Start by comparing the ABC elements with Yair's code

ngrid = 100

if(0):
	L =10
	lb = -L
	ub = L
	m = 1.0

	a = 0.3
	b = 0.09
	pes = mildly_anharmonic(m,a,b)

	T = 1.0#0.125
	beta = 1/T
	potkey = 'mildly_anharmonic_a_{}_b_{}'.format(a,b)
	Tkey = 'T_{}'.format(np.around(T,3))


if(1):
	lb = -5#-4
	ub = 14#12
	m = 1.0
	
	delta_anh = 0.05
	w_10 = 1.0
	wb = w_10
	wc = w_10 + delta_anh
	alpha = (m*delta_anh)**0.5
	D = m*wc**2/(2*alpha**2)


	pes = morse(D,alpha)
	T = 1.0
	beta = 1/T

	potkey = 'Morse_D_{}_alpha_{}'.format(D,alpha)
	Tkey = 'T_{}'.format(T)
	


DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 
print('vecs',vals)

xgrid = np.linspace(lb,ub,ngrid+1)
potgrid = pes.potential(xgrid)
plt.plot(xgrid,potgrid)
#plt.plot(DVR.grid[:ngrid-1],vecs[:,2])
#print('norm', np.sum(vecs[:,1]*vecs[:,3]*DVR.dx*DVR.grid[:ngrid-1]))
#print('DVR.grid,xgrid', DVR.grid,xgrid)
plt.show()

x_arr = DVR.grid[:DVR.ngrid-1]
basis_N = 50
n_eigen = 30

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t0 = 0.0
ngrid = 100
t1_arr = np.arange(-25,25,0.5)#np.linspace(-20,20.0,ngrid+1)
t2_arr = np.arange(-25,25,0.5)#np.linspace(-20,20.0,ngrid+1)
R2_corr_arr = np.zeros((len(t1_arr),len(t2_arr))) + 0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'all'
	
if(1):
	R2_corr_arr = OTOC_f_1D_omp_updated.otoc_tools.r2_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,\
						vals,m_arr,t2_arr,t1_arr,t0,beta,n_eigen,corrkey,R2_corr_arr.T)
		
	print('time', time.time() - start_time)


	plt.imshow(np.real(R2_corr_arr).T)
	plt.show()	

	#print('fRE', np.real(R2_corr_arr))

	t1,t2 = np.meshgrid(t1_arr, t2_arr)
	fname = 'Quantum_R2_{}_{}_{}_basis_{}_n_eigen_{}_ngrid_{}'.format(corrkey,potkey,Tkey,basis_N,n_eigen,ngrid)	
	store_2D_imagedata_column(np.real(t1),np.real(t2),np.real(R2_corr_arr),fname,'{}/Datafiles'.format(path),extcol = np.imag(R2_corr_arr))

if(0):
	fname = '{}/Datafiles/Quantum_R2_{}_{}_{}_basis_{}_n_eigen_{}_ngrid_{}.txt'.format(path,corrkey,potkey,Tkey,basis_N,n_eigen,ngrid)	

	abcF = '{}/Datafiles/Quantum_R2_ABC_{}_{}_basis_{}_n_eigen_{}_ngrid_{}.txt'.format(path,potkey,Tkey,basis_N,n_eigen,ngrid)	
	acbF = '{}/Datafiles/Quantum_R2_ACB_{}_{}_basis_{}_n_eigen_{}_ngrid_{}.txt'.format(path,potkey,Tkey,basis_N,n_eigen,ngrid)	
	bcaF = '{}/Datafiles/Quantum_R2_BCA_{}_{}_basis_{}_n_eigen_{}_ngrid_{}.txt'.format(path,potkey,Tkey,basis_N,n_eigen,ngrid)	
	cbaF = '{}/Datafiles/Quantum_R2_CBA_{}_{}_basis_{}_n_eigen_{}_ngrid_{}.txt'.format(path,potkey,Tkey,basis_N,n_eigen,ngrid)	
	
	abcf = np.loadtxt(abcF)
	acbf = np.loadtxt(acbF)
	bcaf = np.loadtxt(bcaF)
	cbaf = np.loadtxt(cbaF)

	totf = (-cbaf + bcaf + acbf - abcf)	
	R2x = abcf[:,0]
	R2y = abcf[:,1]
	R2re = totf[:,2]
	R2im = totf[:,3]
	

	ABC = '{}/Datafiles/C_ABC.dat'.format(path)
	ACB = '{}/Datafiles/C_ACB.dat'.format(path)
	BCA = '{}/Datafiles/C_BCA.dat'.format(path)
	CBA = '{}/Datafiles/C_CBA.dat'.format(path)
	
	abc = np.loadtxt(ABC)
	acb = np.loadtxt(ACB)
	bca = np.loadtxt(BCA)
	cba = np.loadtxt(CBA)

	tot = (-cba + bca + acb - abc)	
	r2x = abc[:,0]
	r2y = abc[:,1]
	r2re = tot[:,2]
	r2im = tot[:,3]
	
	data = np.loadtxt(fname)
	X = data[:,0]
	Y = data[:,1]
	Fre = data[:,2]
	Fim = data[:,3]

	#print('?', r2re-Fre)
	plt.scatter(r2x,r2y,c=(r2re-Fre).flatten())
	#plt.scatter(X,Y,c=(Fim).flatten())
	plt.colorbar()	

	plt.xlim([-20,20])
	plt.ylim([-20,20])
	plt.show()	
