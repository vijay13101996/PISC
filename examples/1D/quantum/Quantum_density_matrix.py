import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.Quartic import quartic
from PISC.potentials.Morse_1D import morse
from PISC.potentials import harmonic1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
#import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope

ngrid = 200

if(0):
	lb = -10.0
	ub = 10.0
	m = 0.5

	g = 0.06
	lamda = 1.5

	Tc = lamda*(0.5/np.pi)

	pes = triple_well(lamda,g)
	potkey = 'triple_well_lambda_{}_g_{}'.format(lamda,g)

if(0):
	lb = -10.0
	ub = 30.0
	m = 0.5

	w = 17.5/3#2.2#5.7,9.5,29
	D = 9.375#9.375#4.36
	alpha = 0.44#(0.5*m*w**2/D)**0.5#0.363
	print('alpha', alpha, (2*D*alpha**2/m)**0.5)
	#0.81#0.41#0.175#0.255#1.165
	pes = morse(D,alpha)
	potkey = 'morse'#_lambda_{}_g_{}'
	
if(1):
	L = 10.0#10.0
	lb = -L
	ub = L
	m = 0.5

	#1.5: 0.035,0.075,0.13
	#2.0: 0.08,0.172,0.31

	lamda = 2.0
	g = 0.1#0.085
	Tc = lamda*(0.5/np.pi)    
	pes = double_well(lamda,g)
	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))

if(0):
	L = 5
	lb = -L
	ub = L
	m = 1.0

	a = 1.0
	pes = quartic(a)
	potkey = 'quartic'

if(0):
    L = 5
    lb = -L
    ub = L

    m = 1.0
    omega = 1.0

    pes = harmonic1D(m,omega)
    potkey = 'harmonic'

times = 0.95#0.8
T_au = 5#times*Tc
beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

x_arr = DVR.grid

rho_beta = np.zeros((DVR.ngrid,DVR.ngrid))


print('vecs',vecs.shape)

for n in range(len(vals)):
    for i in range(DVR.ngrid):
        for j in range(DVR.ngrid):
            rho_beta[i,j] += (vecs[i,n]*vecs[j,n]*np.exp(-beta*vals[n]))

i=40
j=48
x = x_arr[i]
xp = x_arr[j]

#rho_ij = ((m*omega)/(2*np.pi*np.sinh(beta*omega)))**0.5*np.exp(-m*omega*((x**2+xp**2)*np.cosh(beta*omega)-2*x*xp)/(2*np.sinh(beta*omega)))

#print('rho_ij',rho_ij,rho_beta[i,j])

Z = np.trace(rho_beta)*DVR.dx
#Zanal = 1/(2*np.sinh(beta*omega/2))
#print('Z',Z,Zanal)

rho_beta = rho_beta/Z

plt.imshow(rho_beta,extent=[lb,ub,lb,ub])
plt.colorbar()
plt.show()

#path = os.path.dirname(os.path.abspath(__file__))



