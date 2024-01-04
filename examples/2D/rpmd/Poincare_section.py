import numpy as np
import PISC
from PISC.engine.Poincare_section import Poincare_SOS
from PISC.potentials.Coupled_harmonic import coupled_harmonic
from PISC.potentials.Quartic_bistable import quartic_bistable
from matplotlib import pyplot as plt
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
#from Saddle_point_finder import separatrix_path, find_minima
import time
import os
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False
### Potential parameters
m = 0.5
N = 20
dt = 0.005

g = 0.1
omega = 1.0
E = 4.0
#Tc = 0.404671903954416 #for E = 5 g = 0.1
#Tc = 0.789382847166302 #for E = 20 g = 0.1
Tc = 0.35224120803940745 #for E = 4 g = 0.1

potkey = 'coupled_harmonic_2D_omega_{}_g_{}'.format(omega, g)

### Temperature is only relevant for the ring-polymer Poincare section
times = 3.0
T = times * Tc
Tkey = 'T_{}Tc'.format(times) 

pes = coupled_harmonic(omega, g)

pathname = os.path.dirname(os.path.abspath(__file__))

xg = np.linspace(-8, 8, int(1e2)+1)
yg = np.linspace(-8, 8, int(1e2)+1)

xgrid, ygrid = np.meshgrid(xg, yg)
potgrid = pes.potential_xy(xgrid, ygrid)

qlist = []

### 'nbeads' can be set to >1 for ring-polymer simulations.
nbeads = 8
#nbeads = 1
PSOS = Poincare_SOS('RPMD', pathname, potkey, Tkey)
PSOS.set_sysparams(pes, T, m, 2)
PSOS.set_simparams(N, dt, dt, nbeads = nbeads, rngSeed = 1)     
PSOS.set_runtime(50.0, 500.0)
if(1):
        #xg = np.linspace(xmin-0.1,xmin+0.1,int(1e2)+1)
        #yg = np.linspace(ymin-0.1,ymin+0.1,int(1e3)+1)

        #xg = np.linspace(0,2*xmin,int(1e2)+1)
        #yg = np.linspace(-2*abs(ymin),4*abs(ymin),int(1e3)+1)

        xg = np.linspace(-8, 8, int(1e2)+1)
        yg = np.linspace(-8, 8, int(1e2)+1)

        xgrid, ygrid = np.meshgrid(xg, yg)
        potgrid = pes.potential_xy(xgrid, ygrid)

        qlist = PSOS.find_initcondn(xgrid, ygrid, potgrid, E)
        PSOS.bind(qcartg = qlist, E = E, sym_init = True)
        if(0): ## Plot the trajectories that make up the Poincare section
                xg = np.linspace(-8,8,int(1e2)+1)
                yg = np.linspace(-5,10,int(1e2)+1)
                xgrid,ygrid = np.meshgrid(xg,yg)
                potgrid = pes.potential_xy(xgrid,ygrid)

                ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/5))
                PSOS.run_traj(0,ax) #(1,2,3,4,8,13 for z=1.25), (2,3) 
                plt.show()
        
        if(1): ## Collect the data from the Poincare section and plot. 
                X, PX, Y = PSOS.PSOS_X(y0 = 0.0)
                #print('X:', X)
                #print('Y:', Y)
                #print('PX:', PX)
                plt.scatter(X, PX, s = 1)
        
        #plt.title('Classical')
        plt.title('RPMD T = 3$T_c$')
        #plt.title(r'$\alpha={}$, E=$V_b$+$3\omega_m/2$'.format(alpha) )#$N_b={}$'.format(nbeads))
        plt.xlabel(r'x')
        plt.ylabel(r'$p_x$')
        
        plt.show()
        #fname = 'Classical_Poincare_Section_x_px_{}_E_{}'.format(potkey,E)
        #store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))
                                
if(0): ## Collect the data from the Poincare section and plot. 
        Y,PY,X = PSOS.PSOS_Y(x0=xmin)
        plt.scatter(Y,PY,s=1)   
        #PSOS.set_simparams(N,dt,dt,nbeads=nbeads,rngSeed=1)
        #PSOS.set_runtime(50.0,500.0)
        #PSOS.bind(qcartg=qlist,E=E)#pcartg=plist)#E=E)
        #Y,PY,X = PSOS.PSOS_Y(x0=0.0)
        plt.title(r'PSOS, $N_b={}$'.format(nbeads))
        #plt.scatter(Y,PY,s=2)
        plt.show()
        #fname = 'Poincare_Section_x_px_{}_T_{}'.format(potkey,T)
        #store_1D_plotdata(X,PX,fname,'{}/Datafiles'.format(pathname))

if(0): ## Initial conditions chosen along the 'transition path' from minima to saddle point
        qlist,vecslist = separatrix_path(m,D,alpha,lamda,g,z)
        plist = []
        print('qlist', qlist[0])
        for i in range(len(qlist)):
                x,y = qlist[i]
                V = pes.potential_xy(x,y)

                K = E-V
                p = (2*m*K)**0.5
                
                eigdir = 1      
                px = vecslist[i,eigdir,0]*p
                py = vecslist[i,eigdir,1]*p
        
                #print('E',V+(px**2+py**2)/(2*m), vecslist[i,eigdir])   
                plist.append([px,py])                   

        plist = np.array(plist)
        rng = np.random.default_rng(0)
        ind = rng.choice(len(qlist),N)  # Choose N points at random from the qlist,plist
        ind = [236]
        print('ind',ind)
        qlist = qlist[ind,:,np.newaxis]
        plist = plist[ind,:,np.newaxis]

        if(0): # Trajectory initialized on barrier-top
                qlist = np.array([[1e-2,0.0]])
                plist = np.array([[0.0,0.0]])
                qlist = qlist[:,:,np.newaxis]
                plist = plist[:,:,np.newaxis]
                print('p',qlist.shape,plist.shape)      

        ### Choice of initial conditions
        
if(0): ## Initial conditions are chosen by scanning through the PES. 
        ind = np.where(potgrid<E)
        xind,yind = ind

        for x,y in zip(xind,yind):
                #x = i[0]
                #y = i[1]
                #ax.scatter( xgrid[x,y],ygrid[x,y])#xgrid[x][y] , ygrid[x][y] )
                qlist.append([xgrid[x,y],ygrid[x,y]])
        #plt.show()
        qlist = np.array(qlist)
        #ind = [599]
        #qlist = qlist[ind,:]
        print('qlist',qlist.shape)


