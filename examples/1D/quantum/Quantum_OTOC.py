import numpy as np
import sys
sys.path.insert(0,"/home/lm979/Desktop/PISC")
from PISC.dvr.dvr import DVR1D
from PISC.potentials.double_well_potential import double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
#import OTOC_f_2D
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from plt_util import prepare_fig, prepare_fig_ax

ngrid = 400
	
if(1):#define Potential 
	L = 10.0#10.0
	lb = -L
	ub = L
	m = 0.5
	lamda = 2.0
	g = 0.08
	Tc = lamda*(0.5/np.pi)    
	pes = double_well(lamda,g)
	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize()#_Lanczos(200) 

if(0):#plot energy levels of dw
	qgrid = np.linspace(lb,ub,ngrid-1)
	potgrid = pes.potential(qgrid)
	#path = os.path.dirname(os.path.abspath(__file__))
	#fname = 'Energy levels 	
	#store_1D_plotdata(qgrid,potgrid,fname,'{}/Datafiles'.format(path)
	fig = plt.figure()
	ax = plt.gca()
	ax.set_ylim([0,50])
	plt.plot(qgrid,potgrid)
	plt.plot(qgrid,abs(vecs[:,2])**2)
	for i in range(10):
			plt.axhline(y=vals[i])
	#plt.suptitle(r'Energy levels for Double well, $\lambda = {}, g={}$'.format(lamda,g)) 
	plt.show()

x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 130#200#120
n_eigen = 80#100#70

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,6.0,600)
OTOC_arr = np.ones_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

fig,ax=prepare_fig_ax(tex=True,dim=1)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\log \, C_{T}(t)$')
print('calculating KUBO')

times = 2#0.8#0.9#0.95#1,2,5,10
T_au = times*Tc
beta = 1.0/T_au 
print('T in au, beta',T_au, beta)
low_T=[0.8,0.9,0.95,1.0]
high_T=[1.0,2.0,5.0,10.0]
all_T=[0.8,0.9,0.95,1.0,2.0,5.0,10.0]
#high_T=[5.0,10.0]#testing
colors=['r','g','b','k']
#low_T=[0.8,0.95]
#colors=['r','g']
file_dpi=600
fname = 'Quantum_Kubo_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)

print('Kubo')
if(0):#calculate Kubo Quantum OTOC
	for times in all_T:	
		OTOC_arr*=0.0
		T_au = times*Tc
		beta = 1.0/T_au
		print(beta) 
		CKubo= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr) 
		print(CKubo[0])
		CKubo/=CKubo[0]
		fname = 'Quantum_Kubo_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
		store_1D_plotdata(t_arr,CKubo,fname,'{}/Datafiles'.format(path))
if(0):#read the corresponding plotdata, plot high T	
	for times,c in zip(high_T,colors):#rgbk
		T_au = times*Tc
		beta = 1.0/T_au
		fname = 'Quantum_Kubo_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
		slope1, ic1, t_trunc1, OTOC_trunc1 = find_OTOC_slope(path+'/Datafiles/'+fname,1.0,1.75)
		plot_1D(ax,'./Datafiles/'+fname, label=r'$T=%.1f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)	
		#plt.plot(t_trunc2,slope2*t_trunc2+ic2,linewidth=4,color='k')
	ax.legend(fontsize='small',fancybox=True)
	plt.show()
	fig.savefig('plots/Thermal_OTOC_Kubo_high_temps.pdf',format='pdf',bbox_inches='tight',dpi=file_dpi)
	fig.savefig('plots/Thermal_OTOC_Kubo_high_temps.png',format='png',bbox_inches='tight',dpi=file_dpi)
if(0):#read the corresponding plotdata, plot low T	
	for times,c in zip(low_T,colors):#rgbk
		T_au = times*Tc
		beta = 1.0/T_au
		fname = 'Quantum_Kubo_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
		slope1, ic1, t_trunc1, OTOC_trunc1 = find_OTOC_slope(path+'/Datafiles/'+fname,1.0,1.75)
		plot_1D(ax,'./Datafiles/'+fname, label=r'$T=%.2f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)	
		#plt.plot(t_trunc2,slope2*t_trunc2+ic2,linewidth=4,color='k')
	ax.legend(fontsize='small',fancybox=True)
	plt.show()
	fig.savefig('plots/Thermal_OTOC_Kubo_low_temps.pdf',format='pdf',bbox_inches='tight',dpi=file_dpi)
	fig.savefig('plots/Thermal_OTOC_Kubo_low_temps.png',format='png',bbox_inches='tight',dpi=file_dpi)
print('quantum')
if(0):#calculate Quantum OTOC
	for times in all_T:	
		OTOC_arr*=0.0
		T_au = times*Tc
		beta = 1.0/T_au
		print(beta) 
		Cstan= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
		Cstan/=Cstan[0]
		fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
		store_1D_plotdata(t_arr,Cstan,fname,'{}/Datafiles'.format(path))
if(0):#read and plot the Quantum plot high_T	
	for times,c in zip(high_T,colors):#rgbk
		T_au = times*Tc
		beta = 1.0/T_au
		fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
		slope2, ic2, t_trunc2, OTOC_trunc2 = find_OTOC_slope(path+'/Datafiles/'+fname,1.0,1.75)
		print(np.shape(t_trunc2))
		plot_1D(ax,'./Datafiles/'+fname, label=r'$T=%.1f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1)	
		#plt.plot(t_trunc2,slope2*t_trunc2+ic2,linewidth=4,color='k')
	ax.legend(fontsize='small',fancybox=True)
	plt.show()
	fig.savefig('plots/Thermal_OTOC_high_temps.pdf',format='pdf',bbox_inches='tight',dpi=file_dpi)
	fig.savefig('plots/Thermal_OTOC_high_temps.png',format='png',bbox_inches='tight',dpi=file_dpi)
if(1):#read and plot the Quantum plot low_T	
	for times,c in zip(low_T,colors):#rgbk
		T_au = times*Tc
		beta = 1.0/T_au
		fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
		slope2, ic2, t_trunc2, OTOC_trunc2 = find_OTOC_slope(path+'/Datafiles/'+fname,1.0,1.75)
		print(np.shape(t_trunc2))
		plot_1D(ax,'./Datafiles/'+fname, label=r'$T=%.2f\,T_\textup{c}$'%times,color=c, log=True,linewidth=1,alpha=0.5)	
		#plt.plot(t_trunc2,slope2*t_trunc2+ic2,linewidth=4,color='k')
	ax.legend(fontsize='small',fancybox=True)
	plt.show()
	fig.savefig('plots/Thermal_OTOC_low_temps.pdf',format='pdf',bbox_inches='tight',dpi=file_dpi)
	fig.savefig('plots/Thermal_OTOC_low_temps.png',format='png',bbox_inches='tight',dpi=file_dpi)


if(0):#calculate KUBO OTOC
	#Cstan= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
	#OTOC_arr*=0.0
	Ckubo= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr)
	fname = 'Quantum_Kubo_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	store_1D_plotdata(t_arr,Ckubo,fname,'{}/Datafiles'.format(path))
	slope1, ic1, t_trunc1, OTOC_trunc1 = find_OTOC_slope(path+'/Datafiles/'+fname,1.2,2.0)
	#fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	#store_1D_plotdata(t_arr,Cstan,fname,'{}/Datafiles'.format(path))

	#slope2, ic2, t_trunc2, OTOC_trunc2 = find_OTOC_slope(path+'/Datafiles/'+fname,1.0,1.75)
	

	#qqkubo = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qq1','kubo',OTOC_arr)
	#print('t=0 value', qqkubo[0])
	
	#n = 2
	#c_mc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)	
	#plt.plot(t_arr,np.log(abs(c_mc)),linewidth=2)	
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
	plt.legend()
	plt.show()

	path = os.path.dirname(os.path.abspath(__file__))
	fname = 'Quantum_OTOC_{}_T_{}Tc_basis_{}_n_eigen_{}'.format(potkey,times,basis_N,n_eigen)
	store_1D_plotdata(t_arr,OTOC_arr,fname,'{}/Datafiles'.format(path))

if(0):#HUSIMI
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
	
