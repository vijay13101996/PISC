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

ngrid = 1000

L=10#4*np.sqrt(1/(4+np.pi))#10
lb=0
ub=L

lbc = 0
ubc = L

m=0.5

print('L',L)

potkey = '1D_Box_m_{}_L_{}'.format(m,np.around(L,2))


anal = True
#anal = False

def potential(x):
    if(x<lbc or x>ubc):
        return 1e12
    else:
        #print('x',x)
        return 0.0#x**4

neigs = 200
potential = np.vectorize(potential)

#xgrid = np.linspace(lb,ub,ngrid)
#plt.plot(xgrid,potential(xgrid))
#plt.ylim([0,1000])
#plt.show()

#----------------------------------------------------------------------

DVR = DVR1D(ngrid, lb, ub,m, potential)
if(not anal):
    vals,vecs = DVR.Diagonalize(neig_total=neigs)


x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

#----------------------------------------------------------------------

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:]/L)

vals = vals_anal
vecs = vecs_anal

#---------------------------------------------------------------------------------------------------
T_au = 0.05
Tkey = 'T_{}'.format(T_au)	

beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

print('vals',vals[:5],vecs.shape)
print('delta omega', vals[1]-vals[0])


x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 40
n_eigen = 20

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,100.0,4000)
C_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'qq_TCF'#'OTOC'#'qp_TCF'
enskey = 'Symmetrized'#'mc'#'Kubo'

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

#Find points of local maxima of Carr
a = C_arr
#ind = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]

#maxi = np.max(C_arr[ind])
#mini = np.min(C_arr[ind])

#diff = maxi - mini
#print('diff',diff)
#exit()

print('C_arr',C_arr)

fig,ax = plt.subplots()
plt.plot(t_arr,((C_arr)))
#plt.scatter(t_arr[ind],C_arr[ind])
#fig.savefig('/home/vgs23/Images/TCF_quartic.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
plt.show()

exit()
#Compute rfourier transform of the TCF
Phi = np.fft.fft(C_arr) 
freq = np.fft.fftfreq(t_arr.shape[-1],t_arr[1]-t_arr[0])
fig,ax = plt.subplots()
plt.plot(2*np.pi*freq,(abs(Phi)))
#plt.scatter(2*np.pi*freq,abs(Phi))
plt.plot(2*np.pi*freq,(np.real(Phi)))
plt.show()


