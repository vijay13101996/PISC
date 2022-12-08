import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well, morse, quartic, asym_double_well
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope

ngrid = 600

if(0): # Triple well potential
	lb = -10.0
	ub = 10.0
	m = 0.5

	g = 0.06
	lamda = 1.5

	Tc = lamda*(0.5/np.pi)

	pes = triple_well(lamda,g)
	potkey = 'triple_well_lambda_{}_g_{}'.format(lamda,g)

if(0): # Morse potential
	lb = -1.0
	ub = 4.0
	m = 0.5

	D = 9.375
	alpha = 1.147
	pes = morse(D,alpha)
	
	w_m = (2*D*alpha**2/m)**0.5
	Vb = D/3

	print('alpha, w_m', alpha, Vb/w_m)
	T_au = 1.0
	potkey = 'morse'
	Tkey = 'T_{}'.format(T_au)
	
if(0): #Double well potential
	L = 10.0#6.0
	lb = -L
	ub = L
	m = 0.5

	#1.5: 0.035,0.075,0.13
	#2.0: 0.08,0.172,0.31

	lamda = 2.0
	g = 0.02#8
	pes = double_well(lamda,g)
	
	Tc = lamda*(0.5/np.pi)    
	times = 20.0#0.8
	T_au = times*Tc

	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	Tkey = 'T_{}Tc'.format(times)	

	print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))

if(1): #Asymmetric double well
	L = 6.0#6.0
	lb = -L
	ub = L
	m = 0.5

	#1.5: 0.035,0.075,0.13
	#2.0: 0.08,0.172,0.31

	lamda = 2.0
	g = 0.08
	k = 0.05
	pes = asym_double_well(lamda,g,k)
	
	Tc = lamda*(0.5/np.pi)    
	times = 1.0#0.8
	T_au = times*Tc

	potkey = 'asym_double_well_lambda_{}_g_{}_k_{}'.format(lamda,g,k)
	Tkey = 'T_{}Tc'.format(times)	

	print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))
	

if(0): # Quartic potential
	L = 8
	lb = -L
	ub = L
	m = 1.0

	a = 1.0
	pes = quartic(a)

	T_au = 1.0/8	
	potkey = 'TESTquart'.format(a)
	Tkey = 'T_{}'.format(T_au)	

beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

print('vals',vals[:5],vecs.shape)
print('delta omega', vals[1]-vals[0])
if(1): # Plots of PES and WF
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
basis_N = 50
n_eigen = 30

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,5.0,2000)
C_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'OTOC'#'qq_TCF'#'OTOC'#'qp_TCF'
enskey = 'Kubo'#'mc'#'Kubo'

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
plt.plot(t_arr,C_arr)
fig.savefig('/home/vgs23/Images/OTOC_temp.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
	

#plt.show()

if(0): #Stray code
	bnm_arr=np.zeros_like(OTOC_arr)
	OTOC_arr[:] = 0.0
	lda = 0.5
	Z = 0.0
	coefftot = 0.0
	for n in range(0,2):
		Z+= np.exp(-beta*vals[n])		
		for M in range(5):
			bnm =OTOC_f_1D_omp_updated.otoc_tools.quadop_matrix_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,M+1,t_arr,1,'cm',OTOC_arr)
			coeff = 0.0
			if(n!=M):
				coeff =(1/beta)*((np.exp(-beta*vals[n]) - np.exp(-beta*vals[M]))/(vals[M]-vals[n]))
			else:
				coeff = np.exp(-beta*(vals[n]))
			coefftot+=coeff
			if(coeff>=1e-4):
				coeffpercent = coeff*100/0.104
				print('coeff',n,M,coeffpercent)
				plt.plot(t_arr,abs(bnm)**2,label='n,M, % ={},{},{}'.format(n,M,np.around(coeffpercent,2)))
				#print('contribution of b_nm for lambda={},n,m={},{}'.format(lda,n,M), np.exp(-beta*vals[n])*np.exp(-lda*beta*(vals[M]-vals[n])))
			bnm_arr+=coeff*abs(bnm)**2
	bnm_arr/=Z
	print('Z',Z,coefftot)
	plt.plot(t_arr,np.log(bnm_arr),color='m',linewidth=3)
	plt.legend()
	#Clamda = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
	
	#cmc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,3,m_arr,t_arr,'xxC',OTOC_arr)  
	#for i in range(15):
	#	OTOC= (OTOC_f_1D_omp.position_matrix.compute_b_mat_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n,i,t_arr,OTOC_arr))
	#	OTOC_arr += abs(OTOC)**2

	#plt.plot(t_arr,bnm_arr,color='k')
	#plt.plot(t_arr,qqkubo)	
	#plt.plot(t_arr,np.log(abs(Clamda)))
	#plt.plot(t_arr,np.log(abs(cmc)))
	#plt.plot(t_arr,np.log(abs(Cstan)), label='Standard OTOC')	
	#plt.plot(t_arr,np.log(abs(Csym)), label='Symmetrized OTOC')
	#plt.plot(t_arr,np.log(abs(Ckubo)),color='k',label=r'Kubo thermal OTOC, $\lambda_K={:.2f}$'.format(np.real(slope1)))
	#plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
	#plt.plot(t_trunc1,slope1*t_trunc1+ic1,linewidth=4,color='k')
	#plt.plot(t_arr,np.log(abs(Cstan)),color='r',label=r'Standard thermal OTOC, $\lambda_S={:.2f} > 2\pi/\beta$'.format(np.real(slope2)))
	#plt.plot(t_trunc2,slope2*t_trunc2+ic2,linewidth=4,color='r')
	#plt.plot(t_arr,np.imag(Cstan), label='Standard xx TCF')	
	#plt.plot(t_arr,np.imag(Csym), label='Symmetrized xx TCF')
	#plt.plot(t_arr,np.real(Ckubo),color='k',label='Kubo xx TCF')

	#plt.suptitle('Double well potential')
	#plt.title(r'$Kubo \; vs \; Standard \; OTOC \; at \; T=T_c$')	
	#plt.title(r'$xp \; TCF \; behaviour \; at \; T=T_c$')
	#plt.legend()
	#plt.show()

	#F = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxF','stan',OTOC_arr) 
	#qq = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qq1','kubo',OTOC_arr) 
	#Cstan= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
	#OTOC_arr*=0.0
	#Ckubo= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qq1','kubo',OTOC_arr)
	#Csym = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
	#Cstan = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.0,OTOC_arr)
	#fname = 'Quantum_{}_{}_{}_{}_basis_{}_n_eigen_{}'.format(enskey,corrkey,potkey,Tkey,basis_N,n_eigen)
	#store_1D_plotdata(t_arr,Ckubo,fname,'{}/Datafiles'.format(path))
	
	n = 2
	print('E', np.around(vals[n],2),vals[n])
	#c_mc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)	
	#fname = 'Quantum_{}_{}_{}_basis_{}_n_eigen_{}_n_{}'.format(enskey,corrkey,potkey,basis_N,n_eigen, n )#np.around(vals[n],2))
	#store_1D_plotdata(t_arr,c_mc,fname,'{}/Datafiles'.format(path))

	#plt.plot(t_arr,qq,lw=2)
	#plt.plot(t_arr,np.log(abs(Csym)),linewidth=2)	
	#plt.show()	



	
