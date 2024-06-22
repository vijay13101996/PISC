import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well, morse, quartic, asym_double_well, harmonic1D
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from scipy.optimize import curve_fit

ngrid = 600

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
	
if(1): #Double well potential
	L = 10.0#6.0
	lb = -L
	ub = L
	m = 0.5

	lamda = 2.0
	g = 0.08
	pes = double_well(lamda,g)
	
	Tc = lamda*(0.5/np.pi)    
	times = 0.4
	T_au = times*Tc

	potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
	Tkey = 'T_{}Tc'.format(times)	

	print('g, Vb,D', g, lamda**4/(64*g), 3*lamda**4/(64*g))

if(0): #Asymmetric double well
	L = 6.0#6.0
	lb = -L
	ub = L
	m = 0.5

	#1.5: 0.035,0.075,0.13
	#2.0: 0.08,0.172,0.31

	lamda = 2.0
	g = 0.08
	k = 0.04
	pes = asym_double_well(lamda,g,k)
	
	Tc = lamda*(0.5/np.pi)    
	times = 10.0#0.8
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

	T_au = 8.0	
	potkey = 'quartic_a_{}'.format(a)
	Tkey = 'T_{}'.format(T_au)	

if(0): # Harmonic potential
    L = 8
    lb = -L
    ub = L
    m = 1.0

    w = 1.0
    pes = harmonic1D(m,w)

    T_au = 4.0
    potkey = 'harmonic_w_{}'.format(w)
    Tkey = 'T_{}'.format(T_au)

beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 

DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

#print('vals',vals[:5],vecs.shape)
#print('delta omega', vals[1]-vals[0])
if(0): # Plots of PES and WF
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

t_arr = np.linspace(-0.1,0.1,201)
C_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'qq_TCF'#'OTOC'#
enskey = 'Kubo'#

corrcode = {'OTOC':'xxC','qq_TCF':'qq1','qp_TCF':'qp1'}
enscode = {'Kubo':'kubo','Standard':'stan'}	

if(0):
    for i in range(len(t_arr)):
        lamda = t_arr[i]
        #C_arr[i] = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_elts(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,0.0,beta,n_eigen,corrcode[corrkey],lamda,C_arr[i]) 
        C_arr[i] = OTOC_f_1D_omp_updated.otoc_tools.wightman_corr_elts(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,0.0,beta,n_eigen,corrcode[corrkey],lamda,C_arr[i])

    fname = 'Wightman_Im_qqTCF_{}_{}_{}'.format(potkey,enskey,Tkey)
    path = os.path.dirname(os.path.abspath(__file__))	
    store_1D_plotdata(t_arr,C_arr,fname,'{}/Datafiles'.format(path))

    fig,ax = plt.subplots()
    plt.plot(t_arr,C_arr)

    #fig.savefig('/home/vgs23/Images/Im_qqTCF.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)	
    plt.show()

if(1):
    times_arr = [0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]#,2.0,2.5,3.0,3.5,4.0]
    #[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
    b1_arr = []
    for times in times_arr:
    #for times in [1.0]:#[0.5,1.0,2.0,3.0,4.0]: #,5.0,6.0,7.0,8.0]: #[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]: #
        Tkey = 'T_{}Tc'.format(times)
        T = times*Tc

        data = read_1D_plotdata('{}/Datafiles/Wightman_Im_qqTCF_{}_{}_{}.txt'.format(path,potkey,enskey,Tkey))
        #data = read_1D_plotdata('{}/Datafiles/Im_qqTCF_{}_{}_{}.txt'.format(path,potkey,enskey,Tkey))
        
        t_arr = data[:,0]
        C_arr = data[:,1]

        #Plot first derivative
        der = T*np.gradient(C_arr,t_arr[1]-t_arr[0])
        derder = np.gradient(der,t_arr[1]-t_arr[0])

        #Fit derivative to linear function
        popt,pcov = curve_fit(lambda x,l: l*x,t_arr,der) 

        b1_arr.append(popt[0])

        #plt.scatter(times,abs(der[0]))
        #plt.plot(t_arr,der,label='T = {}Tc, der'.format(times))
        #plt.plot(t_arr,C_arr,label='T = {}Tc'.format(times))
    
    #plt.legend()
    #plt.show()


    popt,perr = curve_fit(lambda x,l,c: l*x+c,times_arr,b1_arr)

    plt.scatter(times_arr,b1_arr)
    plt.plot(times_arr,b1_arr)
    plt.plot(times_arr,popt[0]*np.array(times_arr) + popt[1])
    plt.plot(times_arr,np.pi*np.array(times_arr)*Tc)
    
    print('popt',popt,np.pi*Tc)
    plt.show()


