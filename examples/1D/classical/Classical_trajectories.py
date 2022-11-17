import numpy as np
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.double_well_potential import double_well
from PISC.potentials.eckart import eckart
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import os
import matplotlib
#from matplotlib import rc
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
lamda = 2.0
g = 0.08
m=0.5
a = -lamda/(8*g)**0.5
b = -a
Vb = lamda**4/(64*g)
dt=0.01

beta = 2.0
tau0 = 100.0
print('therm energy', 1/beta, 'Vb', Vb)

pes = double_well(lamda,g)
ens = Ensemble(beta=beta,ndim=1)
motion = Motion(dt = dt,symporder=4) 
rng = np.random.default_rng(1) 

E = Vb + 0.05
qcart = np.array([[[0.5]]])#
V = pes.potential(qcart)
print('V', V)

pcart = np.zeros_like(qcart) + (2*m*(E-V))**0.5 # (2*m*Vb)**0.5

rp = RingPolymer(qcart=qcart,pcart=pcart,m=m,mode='rp')
rp.bind(ens,motion,rng)
pes.bind(ens,rp)	

therm = PILE_L(tau0=tau0,pile_lambda=10.0) 
therm.bind(rp,motion,rng,ens)

propa = Symplectic_order_IV()
propa.bind(ens, motion, rp, pes, rng, therm)

sim = RP_Simulation()
sim.bind(ens,motion,rng,rp,pes,propa,therm)

time_total = 10
time_total = time_total
nsteps = int(time_total/dt)	

qarr=[]
Mqqarr=[]
tarr=[]
pathname = os.path.dirname(os.path.abspath(__file__))

print('Vb', Vb)
fig,ax = plt.subplots(2, sharex=True,gridspec_kw={'hspace': 0})
	
for i in range(nsteps):
		sim.step(mode="nve",var='monodromy',pc=False)
		q = rp.q[0,0,0]
		Mqq = rp.Mqq[0,0,0,0,0]	
		if(i%2==0):
			Mqqarr.append(Mqq**2)	
			qarr.append(q.copy())
			tarr.append(sim.t)

ax[0].plot(tarr,qarr)
ax[1].plot(tarr,np.log(Mqqarr)/tarr)
plt.show()
	
if(0):
	for i in range(nsteps):
		sim.step(mode="nvt",var='pq',pc=True)
		q = np.mean(rp.q[0,0,0])
		if(i%2==0):
			qarr.append(q.copy())
			tarr.append(sim.t)


	fname = 'Classical_trajectory_beta_{}_tau0_{}'.format(beta,tau0)
	store_1D_plotdata(tarr,qarr,fname,'{}/Datafiles'.format(pathname))

fname = 'Classical_trajectory_beta_{}_tau0_{}'.format(beta,tau0)	
data = read_1D_plotdata('{}/Datafiles/{}.txt'.format(pathname,fname))
tarr = np.array(data[:,0])
qarr = np.array(data[:,1])

if(0):
	tst = np.abs(tarr - 670).argmin()
	tend = np.abs(tarr-713).argmin()

	abarr = [676,679,691,693,697,701,705,707]

	abind = []
	for i in range(8):
		abind.append(np.abs(tarr-abarr[i]).argmin())

	plt.plot(tarr[tst:abind[0]],qarr[tst:abind[0]],color='b')
	plt.plot(tarr[abind[0]:abind[1]],qarr[abind[0]:abind[1]],color='r')
	plt.plot(tarr[abind[1]:abind[2]],qarr[abind[1]:abind[2]],color='b')
	plt.plot(tarr[abind[2]:abind[3]],qarr[abind[2]:abind[3]],color='r')
	plt.plot(tarr[abind[3]:abind[4]],qarr[abind[3]:abind[4]],color='b')
	plt.plot(tarr[abind[4]:abind[5]],qarr[abind[4]:abind[5]],color='r')
	plt.plot(tarr[abind[5]:abind[6]],qarr[abind[5]:abind[6]],color='b')
	plt.plot(tarr[abind[6]:abind[7]],qarr[abind[6]:abind[7]],color='r')
	plt.plot(tarr[abind[7]:tend],qarr[abind[7]:tend],color='b')

	plt.show()

	print('tst,ted',tst,tend)

		
#plt.plot(tarr[:len(tarr)//6],qarr[:len(tarr)//6])
#plt.show()

if(1):
	product = False
	Zarr = []

	tarr =tarr[:len(qarr)//6]
	qarr= qarr[:len(qarr)//6]	

	for i in range(len(tarr)):
		if(abs(qarr[i]-b)<5e-2):
			#print('t',tarr[i])
			product = True 
		if(abs(qarr[i]-a)<5e-2):
			product=False
		if(product is False):
			if(qarr[i]>a):
				Zarr.append(qarr[i]-a)
			else:
				Zarr.append(0.0)
		else:
			if(qarr[i]<b):
				Zarr.append(qarr[i]-b)
			else:
				Zarr.append(0.0)

	#tarr = tarr[::40]
	#qarr = qarr[::40]
	#Zarr = Zarr[::40]
	
if(0):
	fig,ax = plt.subplots(2, sharex=True,gridspec_kw={'hspace': 0})
	ax0 = ax[0]
	ax1 = ax[1]
	x1, x2, y1, y2 = 665, 720, -2.23, 2.23

	ax1.set_ylabel(r'$Z_t(\omega)$')
	ax0.set_ylabel(r'$X_t(\omega)$')
	ax1.set_xlabel(r'$t$')
	
	ax1.yaxis.set_label_coords(-0.07,0.5)
	ax0.yaxis.set_label_coords(-0.07,0.5)
	ax1.xaxis.set_label_coords(0.5,-0.15)
	
	ax0.set_xlim(x1,x2)
	ax1.set_xlim(x1,x2)
	ax0.plot(tarr,qarr,color='midnightblue')
	
	pos = np.where(np.abs(np.diff(Zarr)) >= 0.1)[0]+1
	x = np.insert(tarr, pos, np.nan)
	y = np.insert(Zarr, pos, np.nan)

	ax1.plot(x,y,color='orangered')

	ax0.set_yticks([a,0.0,b]) 
	ax0.set_yticklabels([r'$a$',r'$0$',r'$b$']) 

	up = b-a #+ 0.9
	ax1.set_yticks([0.0,up,-up])
	ax1.set_yticklabels([r'0',r'$b-a$',r'$-(b-a)$']) 

	aa = abs(qarr - a)
	bb = abs(qarr - b)
	a_arr = np.where(aa<2e-2)
	b_arr = np.where(bb<2e-2)

	a_arr = a_arr[:][0]
	b_arr = b_arr[:][0]
	#print('tarr',a_arr)
	#tarr = np.real(tarr)
	#tarr = tarr[tarr>=665]
	#tarr = tarr[tarr<=725]
	
	tai = [676.13,691.10,697.87,705.16]
	tbi = [678.98,693.35,701.27,707.19]

	ta1,ta2,ta3,ta4 = tai
	tb1,tb2,tb3,tb4 = tbi
	tau1 = 682.22
	sigma4 = 707.19
	ai = []
	bi = []
	for i in range(4):
		ai.append(np.abs(tarr - tai[i]).argmin())
		bi.append(np.abs(tarr - tbi[i]).argmin())
		print(tarr[ai[i]],tarr[bi[i]])
	
	a1,a2,a3,a4 = ai
	b1,b2,b3,b4 = bi
	ax0.plot(tarr[ai[0]:bi[0]],qarr[ai[0]:bi[0]],color='g',linewidth=2)
	ax0.plot(tarr[ai[1]:bi[1]],qarr[ai[1]:bi[1]],color='g',linewidth=2)
	ax0.plot(tarr[ai[2]:bi[2]],qarr[ai[2]:bi[2]],color='g',linewidth=2)
	ax0.plot(tarr[ai[3]:bi[3]],qarr[ai[3]:bi[3]],color='g',linewidth=2)
	
	fmt = matplotlib.ticker.StrMethodFormatter("{x}")
	ax0.xaxis.set_major_formatter(fmt)
		
	ax0.set_xticks([ta1,tb1,tau1,ta2,tb2,ta3,tb3,ta4,tb4])
	#ax0.set_xticklabels([r"$a_1$", r"$b_1$", r"$a_2$", r"$b_2$"])#[r"$a_1$",r"$b_1$",r"$a_2$",r"$b_2$"])
	ax0.set_xticklabels([r"$a_1$",r"$b_1$""\n""$(\sigma_1)$",r"$\tau_1$",r"$a_2$",r"$b_2$", r"$a_3$",r"$b_3$",r"$a_4$",r"$b_4$""\n""$(\sigma_4)$"])#[r"$a_1$",r"$b_1$",r"$a_2$",r"$b_2$"])	
	
	ax0.axhline(y=a,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')
	ax0.axhline(y=b,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')

	ax1.axhline(y=up,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')
	ax1.axhline(y=-up,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')

	for i in range(4):
		ax0.axvline(x=tai[i],ymin=0.0, ymax = 1.0,linestyle='--',color='gray')
		ax1.axvline(x=tai[i],ymin=0.0, ymax = 1.0,linestyle='--',color='gray')
		ax0.axvline(x=tbi[i],ymin=0.0, ymax = 1.0,linestyle='--',color='gray')
		ax1.axvline(x=tbi[i],ymin=0.0, ymax = 1.0,linestyle='--',color='gray')

	ax0.axvline(x=tau1,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
	ax1.axvline(x=tau1,ymin=0.0, ymax = 1.0,linestyle='--',color='k')

	ax0.axvline(x=sigma4,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
	ax1.axvline(x=sigma4,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
	
	ax0.axvline(x=tb1,ymin=0.0, ymax = 1.0,linestyle='--',color='k')
	ax1.axvline(x=tb1,ymin=0.0, ymax = 1.0,linestyle='--',color='k')



	fig.set_size_inches(9, 5)
	fig.savefig('/home/vgs23/Images/Crossings_zoomed.eps', dpi=100)


	
	if(0):
		for i in a_arr:
			#print('tarr',tarr[i],qarr[i])
			if( tarr[i]>660 and tarr[i]<725 ):
				print('a',tarr[i])

		for i in b_arr:
			if(tarr[i]>660 and tarr[i]<725 ):
				print('b',tarr[i])

if(0):	
	print('a', a)
	fig,ax = plt.subplots(2, sharex=True,gridspec_kw={'hspace': 0})
	ax0 = ax[0]
	ax1 = ax[1]
	
	ax0.set_yticks([a,0.0,b]) 
	ax0.set_yticklabels([r'$a$',r'$0$',r'$b$']) 

	up = b-a #+ 0.9
	ax1.set_yticks([0.0,up,-up])
	ax1.set_yticklabels([r'0',r'$b-a$',r'$-(b-a)$']) 

	a1 = 0.0
	b1 = 707#723.0
	a2 = 3158#3161.0
	b2 = 6045#6087.0

	ax0.plot(tarr,qarr,color='midnightblue')
	ax1.plot(tarr,Zarr,color='orangered')
	ax1.set_ylabel(r'$Z_t(\omega)$')
	ax0.set_ylabel(r'$X_t(\omega)$')
	ax1.set_xlabel(r'$t$')
	
	ax1.xaxis.get_major_ticks()[1].tick1line.set_markersize(0)
	ax1.xaxis.get_major_ticks()[4].tick1line.set_markersize(0)

	fmt = matplotlib.ticker.StrMethodFormatter("{x}")
	ax0.xaxis.set_major_formatter(fmt)

	ax0.set_xticks([a1,(b1-a1)/2,b1,a2,(b2-a2)/2+a2,b2]) 
	#ax0.set_xticklabels([r"$a_1$", r"$b_1$", r"$a_2$", r"$b_2$"])#[r"$a_1$",r"$b_1$",r"$a_2$",r"$b_2$"])
	ax0.set_xticklabels([r"$\tau_0$",r"$\cdot\cdot\cdot$", r"$\sigma_4$", r"$\tau_4$",r"$\cdot\cdot\cdot$", r"$\sigma_7$"])#[r"$a_1$",r"$b_1$",r"$a_2$",r"$b_2$"])	
	ax0.tick_params(axis='x',length=0)
	
	axins = inset_axes(ax0, width="25%", height="37%", loc=4,borderpad=1.1)#([0.5, 0.5, 0.47, 0.47])	
	x1, x2, y1, y2 = 665, 720, -2.23, 2.23
	axins.plot(tarr,qarr,color='midnightblue')
	mark_inset(ax0, axins, loc1=2, loc2=3, fc="none",lw=1.25, ec="0.25")
	axins.set_xlim(x1, x2)
	#axins.set_ylim(y1, y2)
	axins.set_yticks([])
	axins.set_xticks([])
	axins.set_xticklabels([])
	axins.set_yticklabels([])

	pos = np.where(np.abs(np.diff(Zarr)) >= 0.1)[0]+1
	x = np.insert(tarr, pos, np.nan)
	y = np.insert(Zarr, pos, np.nan)

	axins = inset_axes(ax1, width="25%", height="37%", loc=1,borderpad=1.1)#([0.5, 0.5, 0.47, 0.47])	
	x1, x2, y1, y2 = 665, 720, -2.23, 2.23
	mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", lw=1.25,ec="0.25")	
	axins.plot(x,y,color='orangered')
	axins.set_xlim(x1, x2)
	#axins.set_ylim(y1, y2)
	axins.set_yticks([])
	axins.set_xticks([])
	axins.set_xticklabels([])
	axins.set_yticklabels([])
	axins.set_xlabel('')
	axins.xaxis.label.set_visible(False)
	
	ax1.yaxis.set_label_coords(-0.07,0.5)
	ax0.yaxis.set_label_coords(-0.07,0.5)
	
	if(0):
		rc('text', usetex=True)
		rc('font', family='serif')

	matplotlib.rcParams.update({'font.size': 12, 'font.family': 'serif','font.serif':'Times New Roman'})

	ax1.annotate('Downcrossing', xy=(0.0,0.00), xytext=(0.17,0.25) , textcoords ='axes fraction')
	ax1.annotate('Upcrossing', xy=(0.0,0.00), xytext=(0.48,0.75) , textcoords ='axes fraction')

	ax0.annotate('Downcrossing', xy=(0.0,0.00), xytext=(0.17,0.38) , textcoords ='axes fraction')
	ax0.annotate('Upcrossing', xy=(0.0,0.00), xytext=(0.48,0.45) , textcoords ='axes fraction')
	
	for x in [a1,b1,a2,b2]:
		ax0.axvline(x=x,ymin=0.0, ymax = 1.0,linestyle='--',color='gray')
		ax1.axvline(x=x,ymin=0.0, ymax = 1.0,linestyle='--',color='gray')

	ax0.axhline(y=a,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')
	ax0.axhline(y=b,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')

	ax1.axhline(y=up,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')
	ax1.axhline(y=-up,xmin=0.0, xmax = 1.0,linestyle='--',color='gray')

	fig.set_size_inches(9, 5)
	fig.savefig('/home/vgs23/Images/Crossings.eps', dpi=100)

#plt.xlabel(r'$t$')
#plt.show()	
