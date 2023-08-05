import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.utils.misc import find_maxima
from PISC.potentials import mildly_anharmonic, Veff_classical_1D_LH, Veff_classical_1D_GH, Veff_classical_1D_FH, Veff_classical_1D_FK,double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D

ngrid = 501

L = 5
lb = -L
ub = L
dx = (ub-lb)/(ngrid-1)
m = 1.0

a = 0.1
b = 0.5
pes = mildly_anharmonic(m,a,b)

if(0):
	lamda = 2.0
	g = 0.11
	pes = double_well(lamda,g)
	potkey = 'Double Well'

	Vb = lamda**4/(64*g)
	print('Tc, Vb', 0.5*lamda/np.pi, Vb)

T_au = 0.25#0.75
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

#Global harmonic approximation
def hess(q):
	return 1.0 - 0.01*b*q**2
if(0):
	pes_eff_GH = Veff_classical_1D_GH(pes,beta,m) 
	Veff_GH = np.zeros_like(qgrid) 
	for i in range(len(qgrid)): 
		Veff_GH[i] = pes_eff_GH.potential(qgrid[i])

	Zeff_GH = np.sum(dx*np.exp(-beta*Veff_GH))
	Peff_GH = np.exp(-beta*Veff_GH)/Zeff_GH

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

if(0):
	potgrid = pes.potential(qgrid)
	
	fig = plt.figure()
	ax = plt.gca()
	ax.set_ylim([0,2])
	ax.set_xlim([-5,5.])
	
	for i in range(4):
			plt.axhline(y=vals[i],color='r')
	
	s = 5
	plt.plot(qgrid,potgrid)
	#plt.plot(qgrid, pes.ddpotential(qgrid))
	plt.plot(qgrid,s*Pq,color='r',label=r'$Quantum$')
	#plt.plot(qgrid,Pq_analytic)
	plt.plot(qgrid,s*Pcl,color='g',linestyle=':',label=r'$Classical$')
	plt.plot(qgrid,s*Peff_LH,color='k',label=r'$V_{eff} \; LH$')
	#plt.plot(qgrid,Peff_GH,color='m',label='Veff GH')
	#plt.plot(qgrid,Peff_FH,color='c',linestyle = '--',label='Veff FH')
	plt.plot(qgrid,s*Peff_FK,color='b',linestyle='-.',label=r'$V_{eff} \; FK$')
	
	plt.scatter(qgrid[LHind],s*Peff_LH[LHind],color='k')
	plt.scatter(qgrid[Qind], s*Pq[Qind],color='r')	
	plt.scatter(qgrid[FKind], s*Peff_FK[FKind],color='b')
	plt.scatter(qgrid[Cind], s*Pcl[Cind],color='g')
	
	plt.suptitle(r'Double Well Potential, $\lambda=2.0, g={}$'.format(g))
	plt.title(r'$T={}, \beta V_b = {}$'.format(T_au, np.around(Vb/T_au,2)))
	plt.legend()
	plt.show()

if(1):
	potgrid = pes.potential(qgrid)
	
	fig = plt.figure()
	ax = plt.gca()
	ax.set_ylim([0,5])
	ax.set_xlim([-4.5,4.5])
	
	#for i in range(4):
	#		plt.axhline(y=vals[i],color='r')
	
	s = 5
	plt.plot(qgrid,potgrid)
	#plt.plot(qgrid, pes.ddpotential(qgrid))
	plt.plot(qgrid,s*Pq,color='r',label=r'$Quantum$')
	#plt.plot(qgrid,Pq_analytic)
	plt.plot(qgrid,s*Pcl,color='g',linestyle=':',lw=3,label=r'$Classical$')
	plt.plot(qgrid,s*Peff_LH,color='k',lw=1.5,label=r'$V_{eff} \; LH$')
	#plt.plot(qgrid,Peff_GH,color='m',label='Veff GH')
	#plt.plot(qgrid,Peff_FH,color='c',linestyle = '--',label='Veff FH')
	plt.plot(qgrid,s*Peff_FK,color='b',linestyle='-.',label=r'$V_{eff} \; FK$')
	
	#plt.scatter(qgrid[LHind],s*Peff_LH[LHind],color='k')
	#plt.scatter(qgrid[Qind], s*Pq[Qind],color='r')	
	#plt.scatter(qgrid[FKind], s*Peff_FK[FKind],color='b')
	#plt.scatter(qgrid[Cind], s*Pcl[Cind],color='g')
	
	plt.suptitle(r'Harmonic Oscillator Potential, $\omega=1.0, m=1.0$')
	plt.title(r'$T={}, \beta E_0={} $'.format(T_au,np.around(0.5*beta,2)))
	plt.legend()
	plt.show()



