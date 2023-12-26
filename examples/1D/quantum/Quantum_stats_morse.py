import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import morse, Veff_classical_1D_LH, Veff_classical_1D_GH, Veff_classical_1D_FH, Veff_classical_1D_FK,double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D

ngrid = 401

lb = -5
ub = 20.0
dx = (ub-lb)/(ngrid-1)

if(1): # Anharmonicity parametrisation
    m = 1
    we = 1
    xe = 0.25

    alpha = np.sqrt(2*m*we*xe)
    D = we/(4*xe)

    pes = morse(D,alpha)#,0.0)

    print('w', 2*D*alpha**2/(m))

if(0): # Champagne-bottle PES
    #Morse potential
    
    m = 1741.1 #reduced mass of H in OH bond, in atomic units
    D = 0.18748
    alpha = 1.1605
    req = 0.0#1.8324
    pes = morse(D,alpha,req)

    we = alpha*np.sqrt(2*D/m)
    xe = we/(4*D)

    print('we, xe', we, xe)

    #Morse_fit potential
    coeff = np.flip([0,0, 0.24769049, -0.2977598, 0.21371043, -0.10186151, 
                    0.02996532, -0.00479863, 0.00031677]) #8th order fit
    #coeff = np.flip([0,0, 0.30329442, -0.34246836, 0.16801898, -0.03832334, 0.0032954 ]) #6th order fit
    #coeff = np.flip([0,0, 0.30942813, -0.18122256, 0.02734953]) #4th order fit
    pes_fit = morse_fit(coeff,req) 

qgrid = np.linspace(lb,ub,ngrid)

#K2au = 0.000003166808534191
#T_K = 300
#T_au = T_K*K2au
times = 8
beta = times/we#1.0/T_au 
print('beta',beta, beta*we)
#print('T in au, beta',T_au, beta) 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 

print('vals num',vals[:20])
print('vals anh', we*(np.arange(20)+0.5)-we*xe*(np.arange(20)+0.5)**2)

#DVRF = DVR1D(ngrid,lb,ub,m,pes_fit.potential)
#valsF,vecsF = DVRF.Diagonalize()

Pq = np.zeros_like(qgrid)

#Quantum
Zq = np.sum(np.exp(-beta*vals))
for n in range(len(vals)):
	Pq += np.exp(-beta*vals[n])*vecs[:,n]**2/Zq

Qind = find_maxima(Pq)

if(0):#Quantum from fit potential
    PqF = np.zeros_like(qgrid)
    ZqF = np.sum(np.exp(-beta*valsF))
    for n in range(len(valsF)):
        PqF += np.exp(-beta*valsF[n])*vecsF[:,n]**2/ZqF

    QindF = find_maxima(PqF)


#Classical
Zcl = np.sum(dx*np.exp(-beta*pes.potential(qgrid)))
Pcl = np.exp(-beta*pes.potential(qgrid))/Zcl

Cind = find_maxima(Pcl)

#Local harmonic approximation
pes_eff_LH = Veff_classical_1D_LH(pes,beta,m) 
Veff_LH = np.zeros_like(qgrid) 
for i in range(len(qgrid)): 
	Veff_LH[i] = pes_eff_LH.potential(qgrid[i])

qgrid_LH = qgrid[~np.isnan(Veff_LH)]
Veff_LH = Veff_LH[~np.isnan(Veff_LH)]
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
print('norm', dx*Pq.sum(), dx*Pcl.sum(), dx*Peff_LH.sum())

potgrid = pes.potential(qgrid)
hessgrid = pes.ddpotential(qgrid)
potgridn = potgrid[hessgrid<0]
qgridn = qgrid[hessgrid<0]

lamdac = np.sqrt(2*abs(min(hessgrid))/m)
maxhessn = np.argmin(hessgrid)
Tc = 1.415*lamdac/(2*np.pi)
betac = 1/Tc
print('lamdac',lamdac,Tc,betac)


fig,ax = plt.subplots()
ax = plt.gca()
ax.set_ylim([0,2*D])
ax.set_xlim([-3,15])#[lb,ub])

for i in range(10):
    ax.axhline(y=vals[i],color="0.7",ls='--')

s =1 
ax.plot(qgrid,potgrid)
#ax.plot(qgrid,hessgrid)

ax.plot(qgridn,potgridn,color='m')
ax.scatter(qgrid[maxhessn],potgrid[maxhessn],marker='s',color='m',zorder=4)

#plt.plot(qgrid,Pq_analytic)
ax.plot(qgrid_LH,s*Peff_LH,color='k',lw=1.5,label=r'$V_{eff} \; LH$')
#plt.plot(qgrid,Peff_GH,color='m',label='Veff GH')
#plt.plot(qgrid,Peff_FH,color='c',linestyle = '--',label='Veff FH')
ax.plot(qgrid,s*Peff_FK,color='b',linestyle='-',label=r'$V_{eff} \; FK$',alpha=0.9)
ax.plot(qgrid,s*Pq,color='r',label=r'$Quantum$')
ax.plot(qgrid,s*Pcl,color='g',label=r'$Classical$',alpha=0.9)

ax.scatter(qgrid[LHind],s*Peff_LH[LHind],color='k',zorder=5)
ax.scatter(qgrid[Qind], s*Pq[Qind],color='r',zorder=6)	
#plt.scatter(qgrid[FKind], s*Peff_FK[FKind],color='b')
#plt.scatter(qgrid[Cind], s*Pcl[Cind],color='g')

plt.title(r'$\chi_e={}, \; \beta \hbar \omega = {}$'.format(xe,times))
#plt.title(r'$T={}, \beta E_0={} $'.format(T_au,np.around(0.5*beta,2)))
ax.legend(loc='upper center')
fig.set_size_inches(3,3)
fig.savefig('Morse_anh_Tvl.png',dpi=400,bbox_inches='tight',pad_inches=0.0)
plt.show()



