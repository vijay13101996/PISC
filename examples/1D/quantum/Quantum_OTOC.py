import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.triple_well_potential import triple_well
from PISC.potentials.Quartic import quartic
from PISC.potentials.eckart import eckart
from PISC.potentials.razavy import razavy
from PISC.potentials.trunc_harmonic import trunc_harmonic
from PISC.potentials.Morse_1D import morse
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp, OTOC_f_1D_omp_updated
#import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope

ngrid = 400

if(0):
	lb = -10.0
	ub = 10.0
	m = 0.5

	g = 0.06
	lamda = 1.5

	Tc = lamda*(0.5/np.pi)

	pes = triple_well(lamda,g)
	potkey = 'triple_well_lambda_{}_g_{}'.format(lamda,g)

if(1):
	lb = -3.0
	ub = 20.0
	m = 0.5

	w = 17.5/3#2.2#5.7,9.5,29
	D = 9.375#9.375#4.36
	alpha = 0.8#(0.5*m*w**2/D)**0.5#0.363
	print('alpha', alpha, 1.5*(2*D*alpha**2/m)**0.5)
	#0.81#0.41#0.175#0.255#1.165
	pes = morse(D,alpha)
	Tc = 1.0
	potkey = 'morse'#_lambda_{}_g_{}'
	
if(0):
	L = 10.0#10.0
	lb = -L
	ub = L
	m = 0.5

	#1.5: 0.035,0.075,0.13
	#2.0: 0.08,0.172,0.31

	lamda = 2.0#1.5#3.0#2.5#2.0#1.5
	g = 0.08#0.035#0.13#0.265#0.155#0.08#0.035
	Tc = lamda*(0.5/np.pi)    
	pes = double_well(lamda,g)
	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))

if(0):
	L = 10
	lb = -L
	ub = L
	m = 1.0

	a = 1.0
	pes = quartic(a)
	potkey = 'quartic'

times = 1.0#0.8
T_au = times*Tc
beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

#DVR = DVR1D(ngrid,lb,ub,m,np.vectorize(pes.potential))
DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize()#_Lanczos(200) 

print('vals',vals[:20],vecs.shape)

if(1):
	qgrid = np.linspace(lb,ub,ngrid-1)
	potgrid = pes.potential(qgrid)
	
	#path = os.path.dirname(os.path.abspath(__file__))
	#fname = 'Energy levels for Morse oscillator, D = {}, alpha={}'.format(D,alpha)
	#store_1D_plotdata(qgrid,potgrid,fname,'{}/Datafiles'.format(path))

	#potgrid =np.vectorize(pes.potential)(qgrid)
	#potgrid1 = pes1.potential(qgrid)
	fig = plt.figure()
	ax = plt.gca()
	ax.set_ylim([0,50])
	plt.plot(qgrid,potgrid)
	plt.plot(qgrid,abs(vecs[:,6])**2)	
	#plt.plot(qgrid,potgrid1,color='k')
	#plt.plot(qgrid,-0.16*qgrid**2 + 0.32)
	for i in range(20):
			plt.axhline(y=vals[i])
	#plt.suptitle(r'Energy levels for Double well, $\lambda = {}, g={}$'.format(lamda,g)) 
	plt.show()

x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 80
n_eigen = 20

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,5.0,1000)
OTOC_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))
	
if(1):
	n=2
	M=4
	#OTOC_arr = OTOC_f_1D_omp.position_matrix.compute_c_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n,m_arr,t_arr,OTOC_arr)
	#OTOC_arr = OTOC_f_1D_omp.position_matrix.compute_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
	#F = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxF','stan',OTOC_arr) 
	#qp = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qp1','kubo',OTOC_arr) 
	Cstan= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
	OTOC_arr*=0.0
	Ckubo= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr)
	#Csym = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
	#Cstan = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.0,OTOC_arr)
	#c0 = 0.0j
	#c0 = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_elts(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,0.0,'xxC',c0)
	#print('C(0)', c0)	
	fname = 'Quantum_Kubo_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	store_1D_plotdata(t_arr,Ckubo,fname,'{}/Datafiles'.format(path))

	slope1, ic1, t_trunc1, OTOC_trunc1 = find_OTOC_slope(path+'/Datafiles/'+fname,1.2,2.0)
	fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	store_1D_plotdata(t_arr,Cstan,fname,'{}/Datafiles'.format(path))

	slope2, ic2, t_trunc2, OTOC_trunc2 = find_OTOC_slope(path+'/Datafiles/'+fname,1.0,1.75)
	

	#qqkubo = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qq1','kubo',OTOC_arr)
	#print('t=0 value', qqkubo[0])
	
	#n = 2
	#c_mc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)	
	#plt.plot(t_arr,np.log(abs(c_mc)),linewidth=2)	
	if(0):
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
	plt.plot(t_arr,np.log(abs(Ckubo)),color='k',label=r'Kubo thermal OTOC, $\lambda_K={:.2f}$'.format(np.real(slope1)))
	#plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
	plt.plot(t_trunc1,slope1*t_trunc1+ic1,linewidth=4,color='k')
	plt.plot(t_arr,np.log(abs(Cstan)),color='r',label=r'Standard thermal OTOC, $\lambda_S={:.2f} > 2\pi/\beta$'.format(np.real(slope2)))
	plt.plot(t_trunc2,slope2*t_trunc2+ic2,linewidth=4,color='r')
	#plt.plot(t_arr,np.imag(Cstan), label='Standard xx TCF')	
	#plt.plot(t_arr,np.imag(Csym), label='Symmetrized xx TCF')
	#plt.plot(t_arr,np.real(Ckubo),color='k',label='Kubo xx TCF')

	plt.suptitle('Double well potential')
	plt.title(r'$Kubo \; vs \; Standard \; OTOC \; at \; T=T_c$')	
	#plt.title(r'$xp \; TCF \; behaviour \; at \; T=T_c$')
	plt.legend()
	plt.show()

	path = os.path.dirname(os.path.abspath(__file__))
	fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))

if(0):
	sigma = 1.0#0.5
	qgrid = np.linspace(lb,ub,ngrid+1)
	qgrid = qgrid[1:ngrid]
	husimi = Husimi_1D(qgrid,sigma)

	n = 2
	wf = vecs[:,n]

	print('len',len(wf),len(qgrid))
	
	qbasis = np.linspace(-10,10,200)
	pbasis = np.linspace(-4,4,200)

	dist = husimi.Husimi_distribution(qbasis,pbasis,wf)

	plt.imshow(dist,origin='lower')
	plt.show()
	
