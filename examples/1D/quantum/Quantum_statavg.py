import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials.double_well_potential import double_well

def gauss_op(q,sigma=0.21,mu=0.0):
	return np.exp(-(q-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def wf_expt(xarr,wf,op,dx):
	return np.sum(abs(wf)**2*gauss_op(xarr)*dx)

ngrid = 400
L = 10.0
lb = -L
ub = L
m = 0.5

lamda = 2.0
g = 0.08
Tc = lamda*(0.5/np.pi)    
pes = double_well(lamda,g)
potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))
print('minima',lamda/np.sqrt(8*g))

times = 1.0
T_au = times*Tc
beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 
neigen = 200
print('vals',vals[:20],vecs.shape)

x_arr = DVR.grid[1:DVR.ngrid]

Z = 0.0
for n in range(neigen):
	Z+=np.exp(-beta*vals[n])

exp = 0.0
for n in range(neigen):
	bolt_wgt = np.exp(-beta*vals[n])/Z
	exp += bolt_wgt*wf_expt(x_arr,vecs[:,n],gauss_op,DVR.dx)	

print('exp', exp)
