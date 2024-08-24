import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well, morse, quartic, harmonic1D, mildly_anharmonic
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope

ngrid = 600

L = 12
lb = -L
ub = L

m=1.0
a=0.0
b=1.0
omega=0.0
n_anharm=4

pes = mildly_anharmonic(m,a,b,omega,n=n_anharm)

T_au = 20.0	
potkey = 'quartic_a_{}'.format(a)
Tkey = 'T_{}'.format(T_au)	

beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

print('vals',vals[:5],vecs.shape)
print('delta omega', vals[1]-vals[0])
if(0): # Plots of PES and WF
	qgrid = np.linspace(lb,ub,ngrid-1)
	potgrid = pes.potential(qgrid)
	hessgrid = pes.ddpotential(qgrid)
	idx = np.where(hessgrid[:-1] * hessgrid[1:] < 0 )[0] +1
	idx=idx[0]
	print('idx', idx, qgrid[idx], hessgrid[idx], hessgrid[idx-1])
	print('E inflection', potgrid[idx])	

	#path = os.path.dirname(os.path.abspath(__file__))
	#fname = 'Energy levels for Morse oscillator, D = {}, alpha={}'.format(D,alpha)
	#store_1D_plotdata(qgrid,potgrid,fname,'{}/Datafiles'.format(path))

	#potgrid =np.vectorize(pes.potential)(qgrid)
	#potgrid1 = pes1.potential(qgrid)
	fig = plt.figure()
	ax = plt.gca()
	ax.set_ylim([0,20])
	plt.plot(qgrid,potgrid)
	plt.plot(qgrid,abs(vecs[:,10])**2)	
	#plt.plot(qgrid,potgrid1,color='k')
	#plt.plot(qgrid,-0.16*qgrid**2 + 0.32)
	for i in range(20):
			plt.axhline(y=vals[i])
	#plt.suptitle(r'Energy levels for Double well, $\lambda = {}, g={}$'.format(lamda,g)) 
	fig.savefig('/home/vgs23/Images/PES_temp.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	#plt.show()

x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 300
n_eigen = 250

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,30.0,2000)
C_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'qq_TCF'#'OTOC'#'qp_TCF'
enskey = 'Standard'#'mc'#'Kubo'

corrcode = {'OTOC':'xxC','qq_TCF':'qq1','qp_TCF':'qp1'}
enscode = {'Kubo':'kubo','Standard':'stan'}	

if(1): #Thermal correlators
	if(enskey == 'Symmetrized'):
		C_arr = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,corrcode[corrkey],0.5,C_arr)
	else:
		C_arr = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,corrcode[corrkey],enscode[enskey],C_arr) 
	fname = 'Quantum_{}_{}_{}_{}_basis_{}_n_eigen_{}'.format(enskey,corrkey,potkey,Tkey,basis_N,n_eigen)	
	print('fname',fname)	

if(0): #Microcanonical correlators
	n = 2
	C_arr = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,t_arr,corrcode[corrkey],C_arr)	
	fname = 'Quantum_mc_{}_{}_n_{}_basis_{}'.format(corrkey,potkey,n,basis_N)
	print('fname', fname)	

path = os.path.dirname(os.path.abspath(__file__))	
store_1D_plotdata(t_arr,C_arr,fname,'{}/Datafiles'.format(path))

fig,ax = plt.subplots()
plt.plot(t_arr,((C_arr)))
#fig.savefig('/home/vgs23/Images/TCF_quartic.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
plt.show()

#Compute rfourier transform of the TCF
Phi = np.fft.fft(C_arr) 
freq = np.fft.fftfreq(t_arr.shape[-1],t_arr[1]-t_arr[0])
fig,ax = plt.subplots()
plt.plot(2*np.pi*freq,(abs(Phi)))
#plt.scatter(2*np.pi*freq,abs(Phi))
plt.plot(2*np.pi*freq,(np.real(Phi)))
plt.show()


