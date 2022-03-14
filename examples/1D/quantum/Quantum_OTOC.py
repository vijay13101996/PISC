import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.eckart import eckart
from PISC.potentials.razavy import razavy
from PISC.potentials.trunc_harmonic import trunc_harmonic
from PISC.potentials.Morse_1D import morse
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
#import OTOC_f_1D
#import OTOC_f_2D
from matplotlib import pyplot as plt
import os 

L = 60.0
lb = -L//2
ub = L
m = 0.5
ngrid = 400

if(1):
	w = 0.1
	D = 5.0
	alpha = (0.5*m*w**2/D)**0.5
	pes = morse(D,alpha)
	Tc = 1.0

if(0):
	lamda = 0.8
	g = 1/50.0
	Tc = lamda*(0.5/np.pi)    
	pes = double_well(lamda,g)
	#potkey = 'inv_harmonic'

if(0):
	D = 0.32
	k = 0.4
	w = 5.0#0.005
	Tc = 0.8*(0.5/np.pi)#2*D**0.5*k*(0.5/np.pi) 
	pes1 = eckart(D,k,w)
	potkey = 'eckart'

if(0):
	D = 100.0
	k = 0.4
	xi = 0.32**0.5
	Tc = 0.8*(0.5/np.pi)
	pes = razavy(D,k,xi)
	potkey = 'razavy'

if(0):
	lamda = 0.8
	g = 1/50.0
	wall = 8.0
	pes = trunc_harmonic(lamda,g,wall)
	potkey = 'trunc_harmonic'

times = 1
T_au = times*Tc
print('T in au',T_au) 
beta = 1.0/T_au 

#DVR = DVR1D(ngrid,lb,ub,m,np.vectorize(pes.potential))
DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

print('vals',vals[:20])

if(1):
	qgrid = np.linspace(lb,ub,2000)
	potgrid = pes.potential(qgrid)
	#potgrid =np.vectorize(pes.potential)(qgrid)
	#potgrid1 = pes1.potential(qgrid)
	fig = plt.figure()
	ax = plt.gca()
	ax.set_ylim([0,10])
	plt.plot(qgrid,potgrid)	
	#plt.plot(qgrid,potgrid1,color='k')
	#plt.plot(qgrid,-0.16*qgrid**2 + 0.32)
	for i in range(10):
			plt.axhline(y=vals[i])
	plt.show()

x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 30
n_eigen = 30

k_arr = np.arange(basis_N)
m_arr = np.arange(basis_N)

t_arr = np.linspace(0,10.0,1000)
OTOC_arr = np.zeros_like(t_arr)

OTOC_arr = OTOC_f_1D.position_matrix.compute_otoc_arr_t(vecs,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 

ind_zero = 124
ind_max = 281
ist =  25 #180#360
iend = 35 #220#380

t_trunc = t_arr[ist:iend]
OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
print('slope',slope)

a = -OTOC_arr
x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
#print('min max',t_arr[124],t_arr[281])

plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
plt.show()

path = os.path.dirname(os.path.abspath(__file__))
fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))
	
 
