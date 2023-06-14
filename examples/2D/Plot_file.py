import numpy as np
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope
from matplotlib import pyplot as plt
import os
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from scipy.signal import savgol_filter
 
dim=2

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb#9.375 

z = 1.0
 
Tc = 0.5*lamda/np.pi
times = 2.0
T = times*Tc
beta=1/T

m = 0.5
N = 1000
dt_therm = 0.01
dt = 0.002
time_therm = 40.0
time_total = 5.0

nbeads = 1

potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
	
tarr = np.arange(0.0,time_total,dt)
OTOCarr = np.zeros_like(tarr) +0j

#Path extensions
path = os.path.dirname(os.path.abspath(__file__))	
#path = '/scratch/vgs23/PISC/examples/2D'
qext = '{}/quantum/Datafiles/'.format(path)
cext = '{}/cmd/Datafiles/'.format(path)
Cext = '{}/classical/Datafiles/'.format(path)
rpext = '{}/rpmd/Datafiles/'.format(path)

#Simulation specifications
corrkey = 'OTOC'#'qq_TCF'#
beadkey = 'nbeads_{}_'.format(nbeads)
Tkey = 'T_{}Tc'.format(times)
syskey = 'Selene'

fig,ax = plt.subplots()

if(1):
	dt=0.01
	wb=1.0
	#tarr = np.arange(0.0,100.0,0.01)
	for c,sty in zip([0.0,0.25*wb, 0.35*wb, 0.45*wb, 0.55*wb,0.65*wb,0.75*wb],['-','--','-.',':','-','--','-.']):
		ext = 'Quantum_Kubo_qq1_Tanimura_SB_D_6.050000000000001_alpha_0.31622776601683794_VLL_-0.75_VSL_0.75_cb_{}_beta_8.0_neigen_20_basis_50'.format(c)	
		ext = qext+ext
		data = np.loadtxt("{}.txt".format(ext),dtype=complex)
		tarr = data[:,0]
		carr = (data[:,1])
		if(1):
			tau = 13
			n_order = 2
			delta = np.power( (np.abs(tarr)) / tau, n_order)
			carr*=np.exp(-delta)#(1+np.exp(delta))
			#plt.plot(tarr,carr,label='c={}'.format(c))	
			#carr[5000:] = 0.0
			#print('carr', len(carr))
			FFT = np.fft.fft(np.fft.fftshift(carr))*dt
			FFT = np.abs(np.fft.fftshift(FFT))
			freq = np.fft.fftfreq(len(tarr), dt)
			#print('freq',freq,freq[1]-freq[0]) 
			freq *= 2.0 * np.pi
			freq = np.fft.fftshift(freq)
			#FFT = savgol_filter(FFT, 7, 3)
			
			#tarr = np.arange(0,200.01,0.1)
			#sig = np.cos(tarr)
			#FFT = np.abs(np.fft.fft(sig)*0.1)
			#freq = np.fft.fftfreq(len(tarr),0.1)*2*np.pi
			a=FFT
			maxima = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
			tmax = tarr[maxima]
			tmax = tmax[tmax > 0.0]
			tmax = tmax[tmax <= 5.0]
			wmax = freq[maxima]
			wmax = wmax[tmax > 0.0]
			wmax = wmax[tmax <= 5.0]
			
			print('FFT max', tmax,wmax)
			plt.scatter(freq[maxima],FFT[maxima])

			plt.title(r'Linear spectra (Fourier transform of $<q(0)q(t)>$)')	
			plt.plot(freq, FFT, sty,label='c={},Quantum'.format(c))#,color='g')
			plt.xlim([-5,5])
			#plt.show()
	

if(0):
	wb = 1.0
	dt = 0.01
	tarr = np.arange(0.0,100.0,0.01)
	nbeads = 16
	for c,sty in zip([0.0,0.25*wb, 0.35*wb, 0.45*wb, 0.55*wb,0.65*wb,0.75*wb],['-','--','-.',':','-','--','-.']):
		ext = 'RPMD_thermal_qq_TCF_Tanimura_SB_D_6.050000000000001_alpha_0.31622776601683794_VLL_-0.75_VSL_0.75_cb_{}_T_0.125_nbeads_{}_dt_0.01'.format(c,nbeads)
		ext = rpext+ext
		data = np.loadtxt("{}.txt".format(ext),dtype=complex)
		#tarr = data[:,0]
		carr = (data[:,1])
		#plt.plot(tarr,carr,label='c={}'.format(c))
		if(1):
			tau = 15
			n_order = 2
			delta = np.power( (np.abs(tarr)) / tau, n_order)
			carr*=np.exp(-delta)#(1+np.exp(delta))
			FFT = np.fft.fft(np.fft.fftshift(carr))*dt
			FFT = np.abs(np.fft.fftshift(FFT))
			freq = np.fft.fftfreq(len(tarr), dt)
			#print('freq',freq,freq[1]-freq[0]) 
			freq *= 2.0 * np.pi
			freq = np.fft.fftshift(freq)
				
			#tarr = np.arange(0,200.01,0.1)
			#sig = np.cos(tarr)
			#FFT = np.abs(np.fft.fft(sig)*0.1)
			#freq = np.fft.fftfreq(len(tarr),0.1)*2*np.pi
		
			plt.title(r'Linear spectra (Fourier transform of $<q(0)q(t)>$)')	
			plt.plot(freq, FFT,sty,label='c={},RPMD'.format(c),color='k')
			plt.xlim([-5,5])
			#plt.show()
	
	nbeads = 1
	for c,sty in zip([0.0,0.25*wb, 0.35*wb, 0.45*wb, 0.55*wb,0.65*wb,0.75*wb],['-','--','-.',':','-','--','-.']):
		ext = 'RPMD_thermal_qq_TCF_Tanimura_SB_D_6.050000000000001_alpha_0.31622776601683794_VLL_-0.75_VSL_0.75_cb_{}_T_0.125_nbeads_{}_dt_0.01'.format(c,nbeads)
		ext = rpext+ext
		data = np.loadtxt("{}.txt".format(ext),dtype=complex)
		#tarr = data[:,0]
		carr = (data[:,1])
		#plt.plot(tarr,carr,label='c={}'.format(c))
		if(1):
			tau = 15
			n_order = 2
			delta = np.power( (np.abs(tarr)) / tau, n_order)
			carr*=np.exp(-delta)#(1+np.exp(delta))
			FFT = np.fft.fft(np.fft.fftshift(carr))*dt
			FFT = np.abs(np.fft.fftshift(FFT))
			freq = np.fft.fftfreq(len(tarr), dt)
			#print('freq',freq,freq[1]-freq[0]) 
			freq *= 2.0 * np.pi
			freq = np.fft.fftshift(freq)
				
			#tarr = np.arange(0,200.01,0.1)
			#sig = np.cos(tarr)
			#FFT = np.abs(np.fft.fft(sig)*0.1)
			#freq = np.fft.fftfreq(len(tarr),0.1)*2*np.pi
		
			plt.title(r'Linear spectra (Fourier transform of $<q(0)q(t)>$)')	
			plt.plot(freq, FFT,sty,label='c={},classical'.format(c),color='r')
			plt.xlim([-5,5])
			#plt.show()


	

if(0):
	dt=0.002
	corrkey = 'OTOC'
	enskey = 'thermal'
	Tkey = 'T_{}Tc'.format(times)

	ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
	ext = rpext+ext
	plot_1D(ax,ext, label=r'$RPMD \; 2D \; z={}, T=1T_c$'.format(z),color='k', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,2.3,3.3)
	ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
	print('slope',slope)

	ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,1,dt)
	ext = rpext+ext
	#plot_1D(ax,ext, label=r'$RPMD \; 2D \; z={}, T=1T_c$'.format(z),color='c', log=True,linewidth=1)
	#slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.0,2.0)
	#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
	#print('slope',slope/2)

if(0):
	dt=0.005
	corrkey = 'OTOC'
	enskey ='mc'#'thermal'
	Tkey = 'T_{}Tc'.format(times)

	ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
	ext = rpext+ext
	plot_1D(ax,ext, label=r'$RPMD \; 2D \; z={}, T=T_c$'.format(z),color='c', log=True,linewidth=1)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)
	#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')
	print('slope',slope/2)

	if(1):	
		#rpext = '/home/vgs23/PISC/examples/1D/rpmd/Datafiles/'	
		#potkey = 'inv_harmonic_lambda_{}_g_{}'.format(2.0,0.08)
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,0.5)	
		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+ext
		print('f',ext)
		plot_1D(ax,ext, label=r'$RPMD\; 2D \; z=0.5, T=T_c$',color='m', log=True,linewidth=1)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)
		#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	if(1):
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,0.0)	
		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+ext
		print('f',ext)
		plot_1D(ax,ext, label=r'$RPMD\; 2D \; z=0.0, T=T_c$',color='b', log=True,linewidth=1)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)	
		
	if(0):
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,1.5)	
		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+ext
		print('f',ext)
		plot_1D(ax,ext, label=r'$RPMD\; 2D \; z=1.5, T=T_c$',color='y', log=True,linewidth=1)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)
		
	if(0):
		Tkey = 'T_0.95Tc'
		potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,1.0)	
		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+ext
		print('f',ext)
		plot_1D(ax,ext, label=r'$RPMD\; 2D \; z=1.0, T=0.95T_c$',color='olive', log=True,linewidth=1)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)
		
	if(1):
		Tkey = 'T_1.0Tc'
		rpext = '/home/vgs23/PISC/examples/1D/rpmd/Datafiles/'	
		potkey = 'inv_harmonic_lambda_{}_g_{}'.format(2.0,0.08)
		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,dt)
		ext = rpext+ext
		print('f',ext)
		plot_1D(ax,ext, label=r'$RPMD \; 1D, T=T_c$',color='k', log=True,linewidth=1)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,3.5,4.5)
		#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	
	if(1):
		for z,c in zip([0.0,0.5,1.0,1.5],['y','r','g','lime']):
			potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)		
			ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,20,50)
			ext = qext+ext
			plot_1D(ax,ext, label=r'$Quantum, z={}$'.format(z),color=c, log=True,linewidth=1)
		
if(0):
	corrkey = 'OTOC'
	enskey ='thermal'
	Tkey = 'T_{}Tc'.format(times)

	ext = 'Classical_{}_{}_{}_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,dt)
	ext = Cext+ext
	plot_1D(ax,ext, label=r'$Classical, T=10T_c$',color='c', log=True,linewidth=1.2)
	slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,1.,2.)
	print('slope',slope/2)
	#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

	if(1):
		ext = 'RPMD_{}_{}_{}_{}_nbeads_{}_dt_{}'.format(enskey,corrkey,potkey,Tkey,nbeads,0.002)
		ext = rpext+ext
		plot_1D(ax,ext, label=r'$RPMD \; z={}, T=10T_c$'.format(z),color='b', log=True,linewidth=1)
	
	if(1):
		ext = 'Quantum_Kubo_OTOC_{}_beta_{}_neigen_{}_basis_{}'.format(potkey,beta,50,100)
		ext = qext+ext
		plot_1D(ax,ext, label=r'$Quantum, T=10T_c$',color='k', log=True,linewidth=2.3)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(ext,0.73,1.45)
		print('slope',slope/2)
		#ax.plot(t_trunc, slope*t_trunc+ic,linewidth=3,color='k')

if(0):
	for times,c in zip([0.7,0.9,1.0],['r','g','b']):	
		Tkey = 'T_{}Tc'.format(times)
		ext ='RPMD_mc_{}_{}_{}_nbeads_{}_dt_{}'.format(corrkey,potkey,Tkey,nbeads,dt)
		extclass = rpext + ext
		print('fname',extclass)
		slope, ic, t_trunc, OTOC_trunc = find_OTOC_slope(extclass,1,2)
		print('2pi/beta', 2*np.pi*T,slope/2)	
		plot_1D(ax,extclass, label=r'$RPMD,T={}Tc,\; N_b={}, \lambda={:.3f}$'.format(times,nbeads,np.real(slope/2)),color=c, log=True,linewidth=1)	
		ax.plot(t_trunc, slope*t_trunc+ic,linewidth=2,color='k')
	
	plt.title('RPMD microcanonical OTOCs for the 2D double well')	

plt.legend()
plt.show()

