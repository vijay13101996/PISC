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

if(0):
	lb = -6.0
	ub = 30.0
	m = 0.5

	w = 0.1
	D = 10.0
	alpha = 0.255#0.81#0.41#0.175#0.255#1.165
	pes = morse(D,alpha)
	Tc = 1.0
	potkey = 'morse'#_lambda_{}_g_{}'
	
if(1):
	L = 10
	lb = -L
	ub = L
	m = 0.5

	lamda = 1.5#3.0#2.5#2.0#1.5
	g = 0.035#0.265#0.155#0.08#0.035
	Tc = lamda*(0.5/np.pi)    
	pes = double_well(lamda,g)
	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	print('g, Vb', g, lamda**4/(64*g))

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
	ax.set_ylim([0,20])
	plt.plot(qgrid,potgrid)
	plt.plot(qgrid,abs(vecs[:,3])**2)	
	#plt.plot(qgrid,potgrid1,color='k')
	#plt.plot(qgrid,-0.16*qgrid**2 + 0.32)
	for i in range(20):
			plt.axhline(y=vals[i])
	#plt.suptitle(r'Energy levels for Double well, $\lambda = {}, g={}$'.format(lamda,g)) 
	plt.show()

x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 140
n_eigen = 50

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,200.0,1000)
OTOC_arr = np.zeros_like(t_arr) +0j

if(1):
	n=1
	M=4
	#OTOC_arr = OTOC_f_1D_omp.position_matrix.compute_c_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n,m_arr,t_arr,OTOC_arr)
	#OTOC_arr = OTOC_f_1D_omp.position_matrix.compute_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
	#F = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxF','stan',OTOC_arr) 
	#qp = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qp1','kubo',OTOC_arr) 
	#C = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
	Ckubo = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr)
	#Csym = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
	#Cstan = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.0,OTOC_arr)
	
	#qqkubo = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qq1','kubo',OTOC_arr)
	#print('t=0 value', qqkubo[0])
	n = 2
	#c_mc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)
	#plt.plot(t_arr,c_mc,linewidth=2)	
	if(0):
		bnm_arr=np.zeros_like(OTOC_arr)
		OTOC_arr[:] = 0.0
		lda = 0.5
		for n in range(2,3):
			for M in range(5):
				bnm =OTOC_f_1D_omp_updated.otoc_tools.quadop_matrix_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,M+1,t_arr,1,'cm',OTOC_arr)
				plt.plot(t_arr,abs(bnm)**2,label='n,M={},{}'.format(n,M))
				print('contribution of b_nm for lambda={},n,m={},{}'.format(lda,n,M), np.exp(-beta*vals[n])*np.exp(-lda*beta*(vals[M]-vals[n])))
				bnm_arr+=abs(bnm)**2
		plt.legend()
	#Clamda = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
	
	#cmc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,3,m_arr,t_arr,'xxC',OTOC_arr)  
	#for i in range(15):
	#	OTOC= (OTOC_f_1D_omp.position_matrix.compute_b_mat_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n,i,t_arr,OTOC_arr))
	#	OTOC_arr += abs(OTOC)**2

	if(1):	
		OTOC_arr = Ckubo.copy()
		ind_zero = 124
		ind_max = 281
		ist = 166#119#70#80#100 
		iend = 270#180#110#140#173

		t_trunc = t_arr[ist:iend]
		OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
		slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
		print('slope',slope,2*np.pi/beta)
		print('time used', t_arr[iend]-t_arr[ist])

		a = -OTOC_arr
		x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
		#print('min max',t_arr[124],t_arr[281])

		#fig,ax = plt.subplots()
		#ax.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
	
	#plt.plot(t_arr,bnm_arr,color='k')
	#plt.plot(t_arr,qqkubo)	
	#plt.plot(t_arr,np.log(abs(Clamda)))
	#plt.plot(t_arr,np.log(abs(cmc)))
	#plt.plot(t_arr,np.log(abs(Cstan)), label='Standard OTOC')	
	#plt.plot(t_arr,np.log(abs(Csym)), label='Symmetrized OTOC')
	plt.plot(t_arr,np.log(abs(Ckubo)),color='k',label='Kubo OTOC')
	#plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
	#plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
	#plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
	#plt.plot(t_arr,np.imag(Cstan), label='Standard xx TCF')	
	#plt.plot(t_arr,np.imag(Csym), label='Symmetrized xx TCF')
	#plt.plot(t_arr,np.real(Ckubo),color='k',label='Kubo xx TCF')

	plt.suptitle('Double well potential')
	plt.title(r'$OTOC \; behaviour \; at \; T=T_c$')	
	#plt.title(r'$xp \; TCF \; behaviour \; at \; T=T_c$')
	plt.legend()
	plt.show()

	path = os.path.dirname(os.path.abspath(__file__))
	fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))

if(0):
	fig,ax = plt.subplots()
		
	fname = '/home/vgs23/PISC/examples/1D/quantum/Datafiles/Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	plot_1D(ax,fname,label='Quantum',color='c',log=True)

	fname = '/home/vgs23/PISC/examples/1D/classical/Datafiles/Classical_OTOC_{}_T_1.0Tc_dt_{}'.format(potkey,0.005)
	plot_1D(ax,fname,label='Classical',color='b',log=True)

	data = read_1D_plotdata('{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = data[:,1]
		   
	ist =350#119#70#80#100 
	iend = 450#180#110#140#173

	t_trunc = t_arr[ist:iend]
	OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
	slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
	print('slope',slope,2*np.pi/beta)

	a = -OTOC_arr
	x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
	#print('min max',t_arr[124],t_arr[281])
		
	plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
	plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
	
	fname = '/home/vgs23/PISC/examples/1D/cmd/Datafiles/CMD_OTOC_{}_T_1.0Tc_nbeads_{}_gamma_{}_dt_{}'.format(potkey,4,16,0.005)	
	plot_1D(ax,fname,label='CMD,B=4',color='r',log=True)

	
	fname = '/home/vgs23/PISC/examples/1D/cmd/Datafiles/CMD_OTOC_{}_T_1.0Tc_nbeads_{}_gamma_{}_dt_{}'.format(potkey,8,16,0.005)	
	#plot_1D(ax,fname,label='CMD,B=8',color='g',log=True)

	fname = '/home/vgs23/PISC/examples/1D/cmd/Datafiles/CMD_OTOC_{}_T_1.0Tc_nbeads_{}_gamma_{}_dt_{}'.format(potkey,16,16,0.005)	
	#plot_1D(ax,fname,label='CMD,B=16',color='k',log=True)
	plt.legend()
	plt.show()	

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
	
