import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.potentials import triple_well
from matplotlib import pyplot as plt

def gauss_op(q,sigma=0.21,mu=0.0):
	return np.exp(-(q-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def wf_expt(xarr,wf,op,dx):
	return np.sum(abs(wf)**2*gauss_op(xarr)*dx)

ngrid = 1000
L = 10.0
lb = -L
ub = L
m = 0.5

lamda = 2.0
g = 0.02

pes = triple_well(lamda,g,s=0.5)


DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

if(0):
    qgrid = np.linspace(lb,ub,ngrid)
    V = pes.potential(qgrid)
    #plt.ylim(0,200)
    for i in range(200):
        plt.axhline(y=vals[i],color='r',ls=':',alpha=0.5)

    plt.plot(qgrid,V)
    plt.show()

    exit()


levels = np.diff(vals)[:400]
print('levels',levels)

plt.hist(levels,bins=60)
#plt.plot(np.arange(len(levels)),levels)
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
