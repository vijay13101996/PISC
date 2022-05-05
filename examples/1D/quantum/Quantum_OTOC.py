import numpy as np
from PISC.dvr.dvr import DVR1D
from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials.double_well_potential import double_well
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
 
L = 10
lb = -L#-7.0
ub = L#40.0
m = 0.5
ngrid = 400

if(1):
	w = 0.1
	D = 5.0
	alpha = 0.175#0.41#0.175#0.255#1.165
	pes = morse(D,alpha)
	Tc = 1.0
	potkey = 'morse'#_lambda_{}_g_{}'
	
if(0):
	lamda = 2.0#1.5#1.5#0.8
	g = 1/50.0#0.035#lamda**2/32#0.035#1/8#50.0
	Tc = lamda*(0.5/np.pi)    
	pes = double_well(lamda,g)
	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	print('g, Vb', g, lamda**4/(64*g))

if(0):
	a = 1.0
	pes = quartic(a)
	potkey = 'quartic'

times = 1.0#0.8
T_au = 0.2#times*Tc
print('T in au',T_au) 
beta = 1.0/T_au 

#DVR = DVR1D(ngrid,lb,ub,m,np.vectorize(pes.potential))
DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize()#_Lanczos(200) 

print('vals',vals[:20],vecs.shape)

if(0):
	qgrid = np.linspace(lb,ub,2000)
	potgrid = pes.potential(qgrid)
	
	path = os.path.dirname(os.path.abspath(__file__))
	fname = 'Energy levels for Morse oscillator, D = {}, alpha={}'.format(D,alpha)
	store_1D_plotdata(qgrid,potgrid,fname,'{}/Datafiles'.format(path))

	#potgrid =np.vectorize(pes.potential)(qgrid)
	#potgrid1 = pes1.potential(qgrid)
	fig = plt.figure()
	ax = plt.gca()
	ax.set_ylim([0,20])
	plt.plot(qgrid,potgrid)	
	#plt.plot(qgrid,potgrid1,color='k')
	#plt.plot(qgrid,-0.16*qgrid**2 + 0.32)
	for i in range(20):
			plt.axhline(y=vals[i])
	plt.suptitle(r'Energy levels for Double well, $\lambda = 1.5, g={}$'.format(g)) 
	plt.show()

x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 100
n_eigen = 40

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,20.0,1000)
OTOC_arr = np.zeros_like(t_arr)

if(1):
	n=1
	M=4
	#OTOC_arr = OTOC_f_1D_omp.position_matrix.compute_c_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n,m_arr,t_arr,OTOC_arr)
	#OTOC_arr = OTOC_f_1D_omp.position_matrix.compute_otoc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
	qq = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxF','stan',OTOC_arr) 
	#cmc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,5,m_arr,t_arr,'xxC',OTOC_arr)  
	#for i in range(15):
	#	OTOC= (OTOC_f_1D_omp.position_matrix.compute_b_mat_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n,i,t_arr,OTOC_arr))
	#	OTOC_arr += abs(OTOC)**2
	
	ind_zero = 124
	ind_max = 281
	ist = 131#119#70#80#100 
	iend = 240#180#110#140#173

	t_trunc = t_arr[ist:iend]
	OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
	slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
	print('slope',slope,2*np.pi/beta)

	a = -OTOC_arr
	x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
	#print('min max',t_arr[124],t_arr[281])

	fig,ax = plt.subplots()
	
	plt.plot(t_arr,(qq))	
	#plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
	#plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
	#plt.plot(t_arr,slope*t_arr+ic,'--',color='k')
	plt.show()

	path = os.path.dirname(os.path.abspath(__file__))
	fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))

if(0):
	fig,ax = plt.subplots()
		
	fname = '/home/vgs23/PISC/examples/1D/quantum/Datafiles/Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	plot_1D(ax,fname,label='Quantum',color='c',log=True)

	fname = '/home/vgs23/PISC/examples/1D/classical/Datafiles/Classical_OTOC_{}_T_0.8Tc_dt_{}'.format(potkey,0.01)
	plot_1D(ax,fname,label='Classical',color='b',log=True)

	data = read_1D_plotdata('{}.txt'.format(fname))
	tarr = data[:,0]
	OTOC_arr = data[:,1]
		   
	#print('OTOC',OTOCarr) 
	ist =  400#119#70#80#100 
	iend = 550#180#110#140#173

	t_trunc = t_arr[ist:iend]
	OTOC_trunc = (np.log(OTOC_arr))[ist:iend]
	slope,ic = np.polyfit(t_trunc,OTOC_trunc,1)
	print('slope',slope,2*np.pi/beta)

	a = -OTOC_arr
	x = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True])
	#print('min max',t_arr[124],t_arr[281])
		
	plt.plot(t_trunc,slope*t_trunc+ic,linewidth=4,color='k')
	plt.plot(t_arr,slope*t_arr+ic,'--',color='k')

	fname = '/home/vgs23/PISC/examples/1D/cmd/Datafiles/CMD_OTOC_{}_T_1Tc_nbeads_{}_gamma_{}_dt_{}'.format(potkey,4,16,0.005)	
	plot_1D(ax,fname,label='CMD,B=4',color='r',log=True)

	fname = '/home/vgs23/PISC/examples/1D/cmd/Datafiles/CMD_OTOC_{}_T_1Tc_nbeads_{}_gamma_{}_dt_{}'.format(potkey,8,16,0.005)	
	#plot_1D(ax,fname,label='CMD,B=8',color='g',log=True)

	fname = '/home/vgs23/PISC/examples/1D/cmd/Datafiles/CMD_OTOC_{}_T_1Tc_nbeads_{}_gamma_{}_dt_{}'.format(potkey,16,16,0.005)	
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
	
