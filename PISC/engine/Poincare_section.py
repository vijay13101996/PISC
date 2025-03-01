import numpy as np
from PISC.engine.beads import RingPolymer
from PISC.engine.ensemble import Ensemble
from PISC.engine.motion import Motion
from PISC.engine.thermostat import PILE_L
from PISC.engine.simulation import RP_Simulation
from PISC.engine.integrators import Symplectic
from PISC.engine.gen_mc_ensemble import generate_rp
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt

class Poincare_SOS(object):
    def __init__(self,method,pathname,potkey,Tkey,folder_name='Datafiles'):
        self.method = method
        self.pathname = pathname
        self.potkey = potkey
        self.Tkey = Tkey
        self.X = []
        self.Y = []
        self.PX = []
        self.PY = []
        self.folder_name = folder_name
                    
    def set_sysparams(self,pes,T,m,dim):
        self.pes = pes
        self.T = T
        self.beta=1/T
        self.m = m
        self.dim = dim 
        
    def set_simparams(self,N,dt_ens,dt,nbeads=1,rngSeed=0): 
        self.N = N
        self.dt_ens = dt_ens
        self.dt = dt
        self.nbeads = nbeads
        self.rngSeed = rngSeed

    def set_runtime(self,time_ens,time_run):
        self.time_ens = time_ens
        self.time_run = time_run
    
    def bind(self,qcartg,pcartg=None,E=None,specific_traj=None,sym_init=False):
        # Specific trajectories could be chosen by specifying the 'ind' 
        self.ens = Ensemble(beta=self.beta,ndim=self.dim)
        self.motion = Motion(dt = self.dt,symporder=2) 
        self.rng = np.random.default_rng(self.rngSeed) 
        self.therm = PILE_L(tau0=0.1,pile_lambda=100.0) 
        self.propa = Symplectic()
        self.sim = RP_Simulation()
        self.E = E
        
        # If E is specified, the 'gen_mc_ensemble' function initializes a mc ensemble, 'ergodizes' the phase space and
        # returns the initial conditions pcartg, qcartg to use for plotting the Poincare section.   
        if(E is not None):  
            generate_rp(self.pathname,self.m,self.dim,self.N,self.nbeads,self.ens,self.pes,self.rng,self.time_ens,self.dt,self.potkey,self.rngSeed,E,qcartg)
            qcartg = read_arr('Microcanonical_rp_qcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,self.beta,self.potkey,self.rngSeed),"{}/{}".format(self.pathname,self.folder_name))
            pcartg = read_arr('Microcanonical_rp_pcart_N_{}_nbeads_{}_beta_{}_{}_seed_{}'.format(self.N,self.nbeads,self.beta,self.potkey,self.rngSeed),"{}/{}".format(self.pathname,self.folder_name)) 
        
        if(specific_traj is not None):
            qcartg = qcartg[specific_traj]
            pcartg = pcartg[specific_traj]
        if(sym_init):
            qc = np.repeat(qcartg,2,axis=0)
            pc = np.repeat(pcartg,2,axis=0)
            qc[::2,0] = -qc[::2,0]
            pc[::2,0] = -pc[::2,0]
            qcartg = qc
            pcartg = pc
                
        self.rp = RingPolymer(qcart=qcartg,pcart=pcartg,m=self.m)   
        self.sim.bind(self.ens,self.motion,self.rng,self.rp,self.pes,self.propa,self.therm)

    def find_initcondn(self,xgrid,ygrid,potgrid,E):
        ind = np.where(potgrid<E)
        xind,yind = ind
        qlist=[]
        #fig,ax = plt.subplots(1)
        for x,y in zip(xind,yind):
            #x = i[0]
            #y = i[1]
            #ax.scatter( xgrid[x,y],ygrid[x,y])#xgrid[x][y] , ygrid[x][y] )
            qlist.append([xgrid[x,y],ygrid[x,y]])
        #plt.show()
        qlist = np.array(qlist)
        return qlist
    
    def run_traj(self,ind,ax):          
        nsteps = int(self.time_run/self.motion.dt)
        for i in range(nsteps):
            self.sim.step(mode="nve",var='pq')  
            x = self.rp.qcart[ind,0,:]
            px = self.rp.pcart[ind,0,:]
            y = self.rp.qcart[ind,1,:]
            py = self.rp.pcart[ind,1,:]
            if(i%1e1==0):
                #ax.scatter(self.sim.t,y)
                ax.scatter(x,y,s=7,color='r')
                plt.pause(0.05) 
            
    def PSOS_Y(self,x0):
        prev = self.rp.q[:,0,0] - x0
        curr = self.rp.q[:,0,0] - x0
        count=0

        nsteps = int(self.time_run/self.motion.dt)
        #print('E,kin,pot',self.rp.kin+self.pes.pot,self.rp.kin,self.pes.pot)
        Y_list = []
        PY_list = []
        X_list = []
        for i in range(nsteps):
            self.sim.step(mode="nve",var='pq')  
            x = self.rp.q[:,0,0]/self.rp.nbeads**0.5
            px = self.rp.p[:,0,0]/self.rp.nbeads**0.5
            y = self.rp.q[:,1,0]/self.rp.nbeads**0.5
            py = self.rp.p[:,1,0]/self.rp.nbeads**0.5
            curr = x-x0
            ind = np.where( (prev*curr<0.0) & (px>0.0))
            Y_list.extend(y[ind])
            PY_list.extend(py[ind])
            X_list.extend(x[ind])
            prev = curr
            count+=1

        self.Y.extend(Y_list)
        self.PY.extend(PY_list)

        return Y_list,PY_list,X_list
    
    def PSOS_X(self,y0):
        prev = self.rp.q[:,1,0]/self.rp.nbeads**0.5 - y0
        curr = self.rp.q[:,1,0]/self.rp.nbeads**0.5 - y0
        count=0

        nsteps = int(self.time_run/self.motion.dt)
        #print('E,kin,pot',np.sum(self.rp.pcart**2/(2*self.m),axis=1)+self.pes.pot,self.rp.kin,self.pes.pot)
        X_list = []
        PX_list = []
        Y_list = []
    
        for i in range(nsteps):
            self.sim.step(mode="nve",var='pq')  
            x = self.rp.q[:,0,0]/self.rp.nbeads**0.5
            px = self.rp.p[:,0,0]/self.rp.nbeads**0.5
            y = self.rp.q[:,1,0]/self.rp.nbeads**0.5
            py = self.rp.p[:,1,0]/self.rp.nbeads**0.5
            cent_E = np.sum(self.rp.p[:,:,0]**2/self.rp.nbeads,axis=1) + self.pes.potential(self.rp.q[:,:,0]/self.rp.nbeads**0.5)
            #print('t, cent E',self.sim.t,cent_E[0])     
            curr = y-y0
            #Rg = np.sum((self.rp.qcart-self.rp.q[:,:,0])**2, axis=1)
            ind = np.where( (prev*curr<0.0) & (py<0.0))# & (cent_E>0.95*self.E))# & (cent_E>0.8*self.E))
            #if(len(np.array(ind)[0])>0):
            #   print('py',py[ind])
            X_list.extend(x[ind])
            PX_list.extend(px[ind])
            Y_list.extend(y[ind])
            prev = curr
            count+=1
            
        print('X',np.array(X_list).shape)
        self.X.extend(X_list)
        self.PX.extend(PX_list)

        return X_list,PX_list,Y_list            

    def PSOS_X_gyr(self,y0,gyr_min,gyr_max):
        """ Same as PSOS_X but with a filter of the radius of gyration.
        Returns an extra np array with all radii of gyration, so these can be used for histogramming/ passing to other functions."""
        prev = self.rp.q[:,1,0] - y0
        curr = self.rp.q[:,1,0] - y0
        count=0

        nsteps = int(self.time_run/self.motion.dt)
        therm_steps=int(30/self.motion.dt)
        #print('E,kin,pot',np.sum(self.rp.pcart**2/(2*self.m),axis=1)+self.pes.pot,self.rp.kin,self.pes.pot)
        X_list = []
        PX_list = []
        Y_list = []
        gyr_list_np=np.zeros((nsteps,2*self.N))
        for i in range(nsteps):
            self.sim.step(mode="nve",var='pq')  
            x = self.rp.q[:,0,0]/self.rp.nbeads**0.5
            px = self.rp.p[:,0,0]/self.rp.nbeads**0.5
            y = self.rp.q[:,1,0]/self.rp.nbeads**0.5
            py = self.rp.p[:,1,0]/self.rp.nbeads**0.5
            
            gyr_x=np.mean((x-self.rp.qcart[:,0,:])**2,axis=1)
            gyr_y=np.mean((y-self.rp.qcart[:,1,:])**2,axis=1)
            gyr_tot=np.sqrt(gyr_x+gyr_y)
            gyr_list_np[i,:]=gyr_tot[:]

            curr = y-y0
            ind = np.where( (prev*curr<0.0) & (py<0.0)& (gyr_min<np.max(gyr_list_np,axis=0))&(np.max(gyr_list_np,axis=0)<gyr_max))
            nsteps = int(self.time_run/self.motion.dt)
            
            #Initial 10% of the steps discarded, ring polymer relaxation is needed to get an estimate of the max_Rg
            if(10*i>=nsteps and i>therm_steps):
                X_list.extend(x[ind])
                PX_list.extend(px[ind])
                Y_list.extend(y[ind])
            prev = curr
            count+=1
        
        gyr_list_max= np.zeros(2*self.N)
        for s in range(2*self.N):
            gyr_list_max[s]=np.max(gyr_list_np[:,s])
        
        print('shape(X):',np.array(X_list).shape,'(at x: sign of y changes and py<0)')
        self.X.extend(X_list)
        self.PX.extend(PX_list)
        return X_list,PX_list,Y_list, gyr_list_np
        
    def store_data(self,coord): 
        key = [self.method,'Poincare_section',self.potkey,self.Tkey,'{}'.format(self.N)]
        fext = '_'.join(key)    
        fname = ''.join([fext])

        if(coord=='x'): 
            store_1D_plotdata(self.X,self.PX,fname,'{}/{}'.format(self.pathname,self.folder_name))  

        if(coord=='y'): 
            store_1D_plotdata(self.Y,self.PY,fname,'{}/{}'.format(self.pathname,self.folder_name))
