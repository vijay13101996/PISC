import numpy as np
import time

class Simulation(object):
    """
    This is the base class for defining path-integral
    simulations.
    """

    def __init__(self):
        self.t = 0.0
        self.nstep = 0

    def bind(self,ens,motion,rng,rp,pes,propa,therm,
             pes_fort=False,propa_fort=False,transf_fort=False):
        self.ens = ens
        self.motion = motion
        self.rng = rng
        self.rp = rp
        self.pes = pes
        self.propa = propa
        self.therm = therm
 
        # Bind the components

        # Declare fortran 'views' of variables if either of the fortran flags are true
        self.rp.bind(self.ens, self.motion, self.rng, fort=pes_fort or transf_fort or propa_fort)
        self.pes.bind(self.ens, self.rp, pes_fort=pes_fort, transf_fort=transf_fort)
        self.pes.update(update_hess=True)
        self.therm.bind(self.rp,self.motion,self.rng,self.ens)  
        self.propa.bind(self.ens, self.motion, self.rp,
                        self.pes, self.rng, therm,fort=propa_fort)
        self.dt = self.motion.dt
        self.order = self.motion.order
        self.pes_fort = pes_fort
        self.propa_fort = propa_fort
        self.transf_fort = transf_fort

    def rebind(self):
        """ Rebind the components when one (or a few) of them has changed """
        self.bind(self.ens,self.motion,self.rng,self.rp,
                  self.pes,self.propa,self.therm,
                  self.pes_fort,self.propa_fort,self.transf_fort)

class RP_Simulation(Simulation):
    """
    This is the base class for defining ring-polymer based path-integral simulations.
    """
    def __init__(self):
        super(RP_Simulation,self).__init__()

    def bind(self,ens,motion,rng,rp,pes,propa,therm,
             pes_fort=False, propa_fort=False, transf_fort=False):
        super(RP_Simulation,self).bind(ens,motion,rng,rp,pes,propa,therm,
            pes_fort=pes_fort, propa_fort=propa_fort, transf_fort=transf_fort)
        
    def NVE_pqstep(self):#,update_hess=False):
        """ Constant energy step to propagate position and momentum """
        self.propa.pq_step()#update_hess=update_hess)
        self.rp.mats2cart()

    def NVE_pqstep_RSP(self):#,update_hess=False):
        """ Constant energy step to propagate position and momentum using RSP algorithm """
        self.propa.pq_step_RSP()#update_hess=update_hess)
        self.rp.mats2cart() 
            
    def NVE_Monodromystep(self):
        """ Constant energy step to propagate position, momentum and monodromy matrices """
        self.propa.Monodromy_step()
        self.rp.mats2cart()

    def NVE_varstep(self):
        """ Constant energy step to propagate position, momentum and tangent variables """
        self.propa.var_step()
        self.rp.mats2cart()
    
    def NVT_pqstep(self,pc):#,centmove=True,update_hess=False):
        """ Constant temperature step to propagate position and momentum """
        self.propa.O(pc)
        self.propa.pq_step()#centmove,update_hess=update_hess)
        self.propa.O(pc)
        self.rp.mats2cart()

    def NVT_pqstep_RSP(self,pc):#,centmove=True,update_hess=False):
        """ Constant temperature step to propagate position and momentum using RSP algorithm """
        self.propa.O(pc)
        self.propa.pq_step_RSP()#centmove,update_hess=update_hess)
        self.propa.O(pc)
        self.rp.mats2cart()

    def NVT_Monodromystep(self,pc):
        """ Constant temperature step to propagate position, momentum and monodromy matrices """
        self.propa.O(pc)
        self.propa.Monodromy_step()
        self.propa.O(pc)
        self.rp.mats2cart()

    def NVT_const_theta(self,pc):
        """ Constant phase step to propagate position, momentum and monodromy matrices """
        # (Work in progress)
        self.propa.O(pc)
        self.propa.b_cent()
        self.propa.Monodromy_step()
        self.propa.b_cent()
        self.propa.O(pc)
        self.rp.mats2cart()

    def step(self, ndt=1, mode="nvt",var='pq',RSP=False,pc=None):#,centmove=True,update_hess=False):
        """ Propagate the dynamics of the system for ndt steps """
        if mode == "nve":
            if(var=='pq' or var=='fd_monodromy'):
                if var=='fd_monodromy':
                    #Updating the finite difference variables for finite difference monodromy dynamics
                    self.propa.fd = True
                    self.propa.rebind()
                if RSP is False:
                    for i in range(ndt):
                        self.NVE_pqstep()#update_hess=update_hess)
                else:
                    for i in range(ndt):
                        self.NVE_pqstep_RSP()#update_hess=update_hess)
            elif(var=='monodromy'):
                self.propa.update_hess = True #Updating the hessian is necessary for monodromy dynamics
                self.propa.rebind()
                for i in range(ndt):
                    self.NVE_Monodromystep()
            elif(var=='variation'):
                self.propa.update_hess = True #Updating the hessian is necessary for tangent dynamics
                self.propa.rebind() 
                for i in range(ndt):
                    self.NVE_varstep() 
        elif mode == "nvt":
            if(var=='pq' or var=='fd_monodromy'):
                if var=='fd_monodromy':
                    #Updating the finite difference variables is necessary for finite difference monodromy dynamics
                    self.propa.fd = True
                    self.propa.rebind()
                    pc = False # Continuity of the monodromy matrix will be broken if pc=True
                if RSP is False:
                    for i in range(ndt):
                        self.NVT_pqstep(pc)#,centmove,update_hess=update_hess)
                else:
                    for i in range(ndt):
                        self.NVT_pqstep_RSP(pc)#,centmove,update_hess=update_hess)
            elif(var=='monodromy'):
                self.propa.update_hess = True #Updating the hessian is necessary for monodromy dynamics
                self.propa.rebind()
                for i in range(ndt):
                    self.NVT_Monodromystep(pc)
        elif mode == 'nvt_theta':
            self.NVT_const_theta(pc)
        self.t += ndt*self.dt
        self.nstep += ndt

class Matsubara_Simulation(Simulation):
    """ Base class for defining Matsubara dynamics simulations """
    # Mean-field Matsubara dynamics needs to be implemented with a separate class
    def __init__(self):
        super(Matsubara_Simulation,self).__init__()
    
    def bind(self,ens,motion,rng,rp,pes,propa,therm):
        super(Matsubara_Simulation,self).bind(ens,motion,rng,rp,pes,propa,therm)
    
    def NVE_pqstep(self):
        self.propa.pq_step_nosprings()
        self.rp.mats2cart() 
                
    def NVE_Monodromystep(self):
        self.propa.Monodromy_step_nosprings()
        self.rp.mats2cart()
        
    def NVT_pqstep(self,pc):#,centmove=True):
        self.propa.O(pc)
        self.propa.pq_step_nosprings()#centmove)
        self.propa.O(pc)
        self.rp.mats2cart()

    def NVT_Monodromystep(self,pc):
        self.propa.O(pc)
        self.propa.Monodromy_step_nosprings()
        self.propa.O(pc)
        self.rp.mats2cart()

    def step(self, ndt=1, mode="nvt",var='pq',pc=None,centmove=True):
        if mode == "nve":
            if(var=='pq'):
                for i in range(ndt):
                    self.NVE_pqstep()               
            elif(var=='monodromy'):
                self.propa.update_hess=True #Updating the hessian is necessary for monodromy dynamics
                self.propa.rebind() 
                for i in range(ndt):
                    self.NVE_Monodromystep()

        elif mode == "nvt":
            if(var=='pq'):
                for i in range(ndt):
                    self.NVT_pqstep(pc,centmove)
            elif(var=='monodromy'):
                self.propa.update_hess=True #Updating the hessian is necessary for monodromy dynamics
                self.propa.rebind()
                for i in range(ndt):
                    self.NVT_Monodromystep(pc)
        self.t += ndt*self.dt
        self.nstep += ndt


