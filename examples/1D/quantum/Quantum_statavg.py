import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D
from matplotlib import pyplot as plt

def gauss_op(q,sigma=0.21,mu=0.0):
	return np.exp(-(q-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def wf_expt(xarr,wf,op,dx):
	return np.sum(abs(wf)**2*gauss_op(xarr)*dx)

ngrid = 1000
L = 20.0
lb = -L
ub = L
m = 0.5

if(0):
    pes = quartic(0.02/0.25)
    #pes = harmonic1D(1.0,1.0)

if(1):
    lamda = 2.5
    g = 0.02
    Tc = lamda*(0.5/np.pi)    
    pes = double_well(lamda,g)
    Vb = lamda**4/(64*g)
    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
    print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))
    print('minima',lamda/np.sqrt(8*g))

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

minVb = np.argmin(abs(vals-Vb))
print('minVb',minVb,vals[minVb+1]-vals[minVb])

vals = vals[:50]
plt.plot(np.arange(len(vals)),vals)
plt.scatter(np.arange(len(vals)),vals,s=2,color='r')
plt.show()

exit()

levels = np.diff(vals)[:250]
#print('levels',levels)

levelratios = np.zeros(len(levels)-1)
for n in range(len(levels)-1):
    levelratios[n] = levels[n+1]/levels[n]

plt.plot(np.arange(len(levelratios)),levelratios)
plt.scatter(np.arange(len(levelratios)),levelratios,s=2,color='r')
plt.ylim(0,20)
plt.show()

exit()


#plt.hist(levels,bins=40)
#plt.show()

plt.plot(np.arange(len(levels)),levels)
plt.plot(np.arange(len(vals[:250])),vals[:250])
plt.scatter(np.arange(len(levels)),levels)
plt.scatter(np.arange(len(vals[:250])),vals[:250])
plt.scatter(minVb,vals[minVb],color='k')


plt.axhline(y=Vb, color='r', linestyle='-')
plt.ylim(0,2*Vb)
plt.xlim(0,50)
plt.show()

print('vals',vals[:20],vecs.shape)

exit()

times = 1.0
T_au = times*Tc
beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

x_arr = DVR.grid[1:DVR.ngrid]

Z = 0.0
for n in range(neigen):
	Z+=np.exp(-beta*vals[n])

exp = 0.0
for n in range(neigen):
	bolt_wgt = np.exp(-beta*vals[n])/Z
	exp += bolt_wgt*wf_expt(x_arr,vecs[:,n],gauss_op,DVR.dx)	

print('exp', exp)
