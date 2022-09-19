import numpy as np
import PISC
from PISC.engine.integrators import Symplectic_order_II, Symplectic_order_IV, Runge_Kutta_order_VIII
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.potentials.Henon_Heiles import henon_heiles
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.instanton import instantonize
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import time
import os 

L = 5.0
lbx = -L
ubx = L
lby = -L
uby = L
m = 0.5
ngrid = 100
ngridx = ngrid
ngridy = ngrid
lamda = 3.0#0.5
g = 0.1#25

potkey = 'henon_heiles_lamda_{}_g_{}'.format(lamda,g)

T_au = 0.15#1.2*0.1713
beta = 1.0/T_au 

pes = henon_heiles(lamda,g)

N = 1000
dt_therm = 0.01
dt = 0.005
time_therm = 20.0
time_total = 10.0#15.0
time_relax = 10.0

rngSeed = 1
N = 1
dim = 2
nbeads = 1
T = 1.0	

xgrid = np.linspace(lbx,ubx,200)
ygrid = np.linspace(lby,uby,200)
x,y = np.meshgrid(xgrid,ygrid)

#hesgrid = 0.25*(omega**4 + 4*g0*omega**2*(x**2+y**2) - 48*g0**2*x**2*y**2)
#plt.contour(x,y,hesgrid,colors='g',levels=np.arange(0.0,0.1,0.1))	

#fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
#path = '/home/vgs23/PISC/examples/2D/quantum'#os.path.dirname(os.path.abspath(__file__))
#vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
#vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

potgrid = pes.potential_xy(x,y)
plt.contour(x,y,potgrid,colors='k',levels=np.arange(0,20.0,0.2))#levels=vals[:10])
plt.contour(x,y,potgrid,colors='m',levels=np.arange(-10,0.0,1.0))#levels=vals[:10])
#plt.show()

point = 1/lamda*np.array([1,1/3**0.5])#[1.582,1.582]#[1.442,1.442]
nbeads = 15
dim = 2
inst = instantonize(stepsize=1e-4, tol=1e-8)
qcart = np.zeros((1,dim,nbeads))
#qcart[0,:,0] = point
rp = RingPolymer(qcart=qcart,m=m,nmats=nbeads)
ens = Ensemble(beta=beta,ndim=dim)
motion = Motion(dt = dt,symporder=2) 
rng = np.random.default_rng(rngSeed) 

rp.bind(ens, motion, rng)
pes.bind(ens, rp)
pes.update()
inst.bind(pes,rp)

if(0):
	hesspes = (rp.ddpot+pes.ddpot)[0,:,:,0,0]
	vals,vecs = np.linalg.eigh(hesspes)
	print('hess',hesspes)
	print('vals, vecs',vals,vecs)
	print('dethess',np.linalg.det(hesspes))

eigvec = np.array([1.0, -1.0])
rp_init = inst.saddle_init(np.array(point),0.6,eigvec)
rp = RingPolymer(qcart=rp_init,m=m,nmats=nbeads)
rp.bind(ens, motion, rng)
pes.bind(ens, rp)
pes.update()

inst.bind(pes,rp)

#plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:],color='r')	
vals,vecs = inst.diag_hess()
eigdir =0 

print('eigval',vals[eigdir])
q_init = rp.q[0,:,0]
step = inst.grad_desc_step()
if(1): # Gradient descent helps move away from the classical saddle to a lower energy RP configuration below crossover temperature.
	count=0
	while(step!="terminate"):# and abs(rp.q[0,:,0]-q_init).sum()<1e-1):
		inst.slow_step_update(step)
		step = 	inst.grad_desc_step()
		if(0):#count%1000==0):
			np.set_printoptions(suppress=True)
			eigval = np.linalg.eigvalsh(inst.hess)#[eigdir]	
			print('eigval',np.around(eigval,2))	
			plt.scatter(count,(pes.pot+rp.pot).sum(),color='r')
			plt.scatter(count,abs(inst.grad).max(),color='k')
			plt.pause(0.05)
			#print('grad',abs(inst.grad).max())
			#print('pot',(pes.pot+rp.pot).sum())
		if(count%20000==0):
			plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:])
			plt.pause(0.2)
			#print('count',count)
		if(0):#count%10==0):
			#print('dethess', np.linalg.det(inst.hess))	
			np.set_printoptions(suppress=True)
			eigval = np.around(np.linalg.eigvalsh(inst.hess),4)[:nbeads]	
			if((eigval>0.0).any()):
				plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:])
				plt.pause(0.2)
				print('eigval',eigval)
			if(eigval[0]>0.0):
				break
			#print(count),#[eigdir])
			#print('step', np.around(np.linalg.norm(step),5))
			
		count+=1
	print('count',count)
	print('gradient',np.around(inst.grad,4))
	plt.scatter(rp.qcart[0,0,:],rp.qcart[0,1,:],color='m',s=30)

plt.show()