import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import mildly_anharmonic, Veff_classical_1D_LH, Veff_classical_1D_GH, Veff_classical_1D_FH, Veff_classical_1D_FK,double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D

ngrid = 501
m = 1.0
lamda = 2.0

if(0): #Highly anharmonic
    L = 10
    lb = -L
    ub = L
    dx = (ub-lb)/(ngrid-1)
    
    m = 1.0
    lamda = 2.0
    g = 0.11
    pes = double_well(lamda,g)
      
if(1): #Low anharmonicity 
    L = 10
    lb = -L
    ub = L
    dx = (ub-lb)/(ngrid-1)
    
    g = 0.02
    pes = double_well(lamda,g)

potkey = 'Double Well'

Vb = lamda**4/(64*g)
Tc = 0.5*lamda/np.pi
lamdas = np.sqrt(2)*lamda
Xie = np.around(Vb/lamdas,2)

print('Tc, Vb', 0.5*lamda/np.pi, Vb)

times = 1.#415#1.5
T_au = times*Tc#0.75
beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

print('vals',vals[:20],vecs.shape,vals.shape)

qgrid = np.linspace(lb,ub,ngrid)

Pq = np.zeros_like(qgrid)

#Quantum
Zq = np.sum(np.exp(-beta*vals))
for n in range(len(vals)):
        Pq += np.exp(-beta*vals[n])*vecs[:,n]**2/Zq

Qind = find_maxima(Pq)


#Analytic form of probability density for the harmonic potential
Pq_analytic = np.exp(-np.tanh(0.5*beta)*qgrid**2)/(np.pi/np.tanh(0.5*beta))**0.5        

#Classical
Zcl = np.sum(dx*np.exp(-beta*pes.potential(qgrid)))
Pcl = np.exp(-beta*pes.potential(qgrid))/Zcl

Cind = find_maxima(Pcl)

#Local harmonic approximation
pes_eff_LH = Veff_classical_1D_LH(pes,beta,m) 
Veff_LH = np.zeros_like(qgrid) 
for i in range(len(qgrid)): 
        Veff_LH[i] = pes_eff_LH.potential(qgrid[i])
Zeff_LH = np.sum(dx*np.exp(-beta*Veff_LH))
Peff_LH = np.exp(-beta*Veff_LH)/Zeff_LH

LHind = find_maxima(Peff_LH)


#Feynman Hibbs approximation
pes_eff_FH = Veff_classical_1D_FH(pes,beta,m,qgrid) 
Veff_FH = np.zeros_like(qgrid) 
for i in range(len(qgrid)): 
        Veff_FH[i] = pes_eff_FH.potential(qgrid[i])
Zeff_FH = np.sum(dx*np.exp(-beta*Veff_FH))
Peff_FH = np.exp(-beta*Veff_FH)/Zeff_FH

#Feynman Kleinert approximation (upto quadratic)
pes_eff_FK = Veff_classical_1D_FK(pes,beta,m) 
Veff_FK = np.zeros_like(qgrid) 
for i in range(len(qgrid)): 
        Veff_FK[i] = pes_eff_FK.potential(qgrid[i])
Zeff_FK = np.sum(dx*np.exp(-beta*Veff_FK))
Peff_FK = np.exp(-beta*Veff_FK)/Zeff_FK

FKind = find_maxima(Peff_FK)


#Check normalisation
print('norm', dx*Pq.sum(), dx*Pq_analytic.sum(),dx*Pcl.sum(), dx*Peff_LH.sum())# dx*Peff_GH.sum())#,np.sum(vecs[:,10]**2*dx))

potgrid = pes.potential(qgrid)
hessgrid = pes.ddpotential(qgrid)
potgridn = potgrid[hessgrid<0]
qgridn = qgrid[hessgrid<0]

lamdac = np.sqrt(2*abs(min(hessgrid))/m)
maxhessn = np.argmin(hessgrid)
Tca = lamdac/(2*np.pi)
betac = 1/Tca
#print('lamdac',lamdac,Tca,betac)


fig = plt.figure()
ax = plt.gca()

ax.set_ylim([0,20])#([0,5])
ax.set_xlim([-10,10])#[-5,5.])

for i in range(28): # 4,28 for high,low anharmonicity
    ax.axhline(y=vals[i],color="0.7",ls='--')

s = 20
ax.plot(qgrid,potgrid)
#ax.plot(qgrid, pes.ddpotential(qgrid))
ax.plot(qgrid,s*Pq,color='r',label=r'$Quantum$')
#ax.plot(qgrid,Pq_analytic)
ax.plot(qgrid,s*Pcl,color='g',linestyle=':',label=r'$Classical$')
ax.plot(qgrid,s*Peff_LH,color='k',label=r'$V_{eff} \; LH$')
#ax.plot(qgrid,Peff_GH,color='m',label='Veff GH')
#ax.plot(qgrid,Peff_FH,color='c',linestyle = '--',label='Veff FH')
ax.plot(qgrid,s*Peff_FK,color='b',linestyle='-.',label=r'$V_{eff} \; FK$')

ax.scatter(qgrid[LHind],s*Peff_LH[LHind],color='k',s=20)
ax.scatter(qgrid[Qind], s*Pq[Qind],color='r',s=20)  
#ax.scatter(qgrid[FKind], s*Peff_FK[FKind],color='b')
#ax.scatter(qgrid[Cind], s*Pcl[Cind],color='g')

plt.title(r'$\Xi_e = {}, T={}Tc, \beta V_b = {}$'.format(Xie,times, np.around(Vb/T_au,2)),fontsize=10)
ax.legend(loc='upper center')


fig.set_size_inches(3, 3)
fig.savefig('DW_harm_Tvvl.png',dpi=400,bbox_inches='tight',pad_inches=0.0)
plt.show()


