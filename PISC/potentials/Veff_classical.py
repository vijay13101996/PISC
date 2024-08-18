import numpy as np
from PISC.potentials.base import PES

class Veff_classical_1D_LH(PES):
        """
        The local harmonic ansatz for the effective classical potential
        """

        def __init__(self,pes,beta,m,grad=None,hess=None,tol=1.0,renorm='harm'):
                super(Veff_classical_1D_LH).__init__()
                self.beta = beta
                self.m=m
                self.pes = pes
                self.grad = grad
                self.hess = hess
                self.tol = tol
                self.renorm = renorm

        def bind(self,ens,rp):
                super(Veff_classical_1D_LH,self).bind(ens,rp)

        def potential(self,q): 
                if(self.hess is None):
                        hess = self.pes.ddpotential(q)/self.m
                else:
                        hess = self.hess(q)/self.m

                if(self.grad is None):
                        grad = self.pes.dpotential(q)/self.m
                else:
                        grad = self.grad(q)/self.m
       
                wa=0.0
                ka = grad
                Vc = self.pes.potential(q)#self.m**2*hess*q**2/2 #
                if(0): #VGS renormalisation
                    if (hess < self.tol):
                        hess = self.tol
                    wa = np.sqrt(hess)
                    xi = 0.5*self.beta*wa
                    qu = xi/np.tanh(xi)
                if(1): #Liu and Miller renormalisation
                    if (hess >0):
                        wa = np.sqrt(hess)
                        xi = 0.5*self.beta*wa
                        qu = xi/np.tanh(xi)
                    else:
                        print('there')
                        wa = np.sqrt(-hess)
                        xi = 0.5*self.beta*wa
                        qu = np.tanh(xi)/xi

                pref = self.m*ka**2/(2*hess)

                if(self.renorm=='harm'):# Harmonic renormalisation (i.e. gives normalised distribution for a harmonic potential)
                    pref = Vc   
                    #Vnc = pref*(np.tanh(xi)/xi -1) - 0.5*np.log(np.tanh(xi))/self.beta - 0.5*np.log(self.m*wa/np.pi)/self.beta
                    Vnc = pref*(1/qu -1) - 0.5*np.log(1/qu)/self.beta - 0.5*np.log(self.m*wa**2/np.pi)/self.beta
                elif(self.renorm=='PF'):# Normalised to the partition function (i.e. when integrated over q, gives the partition function)
                    Vnc = pref*(np.tanh(xi)/xi -1) + (-0.5*np.log(np.tanh(xi)/xi) + np.log(np.sinh(xi)/xi))/self.beta
                elif(self.renorm=='NCF'):
                    pref = Vc
                    #Vnc =  pref*(np.tanh(xi)/xi -1) - 0.5*np.log(np.tanh(xi))/self.beta + 0.5*np.log(xi)/self.beta
                    Vnc = pref*(1/qu -1) - 0.5*np.log(1/qu)/self.beta 
                return Vc+Vnc

        def distribution(self,q):
                if(self.hess is None):
                        hess = self.pes.ddpotential(q)/self.m
                else:
                        hess = self.hess(q)/self.m

                if(self.grad is None):
                        grad = self.pes.dpotential(q)/self.m
                else:
                        grad = self.grad(q)/self.m
       
                ka = grad
                Vc = self.pes.potential(q)
                if (hess < self.tol):
                    hess = self.tol
                wa = np.sqrt(hess)
                xi = 0.5*self.beta*wa

                prefactor = np.sqrt(np.tanh(xi))*np.sqrt(self.m*wa/np.pi)
                return prefactor*np.exp(-self.beta*(Vc - 0.5*self.m*ka**2/hess) - np.tanh(xi)*self.m*ka**2/wa**3 ) 

        def dpotential(self,q):
                return None

        def ddpotential(self,q):
                return None

class Veff_classical_2D_LH(PES):
    def __init__(self,pes,beta,m,grad=None,hess=None,tol=1.0,renorm='harm'):
        super(Veff_classical_2D_LH).__init__()
        self.beta = beta
        self.m=m
        self.pes = pes
        self.grad = grad
        self.hess = hess
        self.tol = tol
        self.renorm = renorm

    def bind(self,ens,rp):
        super(Veff_classical_2D_LH,self).bind(ens,rp)

    def potential(self,q):
        if(self.hess is None):
            hess = self.pes.ddpotential(q)/self.m
        else:
            hess = self.hess(q)/self.m

        if(self.grad is None):
            grad = self.pes.dpotential(q)/self.m
        else:
            grad = self.grad(q)/self.m

        wa1 = 0.0
        wa2 = 0.0

        ka = grad
        Vc = self.pes.potential(q)
       
        hess1,hess2 = np.linalg.eigvals(hess[0,:,:,0])
        
        if (hess1 < self.tol):
            hess1 = self.tol
        if (hess2 < self.tol):
            hess2 = self.tol
        
        wa1 = np.sqrt(hess1)
        wa2 = np.sqrt(hess2)
        xi1 = 0.5*self.beta*wa1
        xi2 = 0.5*self.beta*wa2

        pref = Vc

        if(self.renorm=='NCF'): # Only NCF renormalisation is implemented for 2D
            Vnc =  pref*(np.tanh(xi1)/xi1 + np.tanh(xi2)/xi2 - 1)  - 0.5*(np.log(np.tanh(xi1)) + np.log(np.tanh(xi2)))/self.beta + 0.5*(np.log(xi1*xi2))/self.beta
        return Vnc

    def potential_xy(self,x,y):
        q = np.zeros((1,2,1))
        q[:,0,:] = x
        q[:,1,:] = y
        return self.potential(q)


class Veff_classical_1D_GH(PES):
    """
    The global harmonic ansatz for the effective classical potential
    """
    def __init__(self,pes,beta,m,hess=None):
        super(Veff_classical_1D_GH).__init__()  
        self.beta = beta 
        self.m=m
        self.pes = pes
        self.hess = hess
        
    def bind(self,ens,rp):
        super(Veff_classical_1D_GH,self).bind(ens,rp)
        
    def potential(self,q):
        #This provides an option to pass a customised hessian field if required
        if(self.hess is None):
            hess = self.pes.ddpotential(q)/self.m
        else:
            print('here')
            hess = self.hess(q)/self.m
        wa = 0.0
        Vc = self.pes.potential(q)
        if (hess >0):
            wa = np.sqrt(hess)
            ka = 2*grad
            xi = 0.5*self.beta*wa
            Vnc = m*ka**2*(np.tanh(xi)/xi -1)/(8*wa**2) + 0.5*self.m*hess*q**2*(np.tanh(xi)/xi - 1) - 0.5*np.log(2*self.m*xi*np.tanh(xi)/(np.pi*self.beta))/self.beta
            ret = 0.5*self.m*hess*q**2*(np.tanh(xi)/xi - 1) - 0.5*np.log(2*self.m*xi*np.tanh(xi)/(np.pi*self.beta))/self.beta
        elif (hess <0): 
            wa = np.sqrt(-hess)
            xi = 0.5*self.beta*wa
            # Potential is defined in terms of tan when the hessian is negative.        
            ret = self.pes.potential(q) + 0.5*self.m*hess*q**2*(np.tan(xi)/xi - 1) - 0.5*np.log(2*self.m*xi*np.tan(xi)/(np.pi*self.beta))/self.beta
        elif (hess==0):
            ret = Vc 

        return ret

    def dpotential(self,q):
        return None

    def ddpotential(self,q):
        return None

def convolution(pot,qgrid,q,beta,m):
    integral = 0.0
    dq = qgrid[1]-qgrid[0]
    for i in range(len(qgrid)):
        integral += np.sqrt(6*m/(np.pi*beta))*pot(qgrid[i])*np.exp(-6*m*(qgrid[i]-q )**2/beta)*dq
    return integral
    
class Veff_classical_1D_FH(PES):
    """
    Feynman-Hibbs effective potential defined as a convolution.
    """
    def __init__(self,pes,beta,m,qgrid):
        super(Veff_classical_1D_FH).__init__()  
        self.beta = beta 
        self.m=m
        self.pes = pes
        self.qgrid = qgrid
        
    def bind(self,ens,rp):
        super(Veff_classical_1D_FH,self).bind(ens,rp)
        
    def potential(self,q):
        # Eq. 11.23 in Feynman-Hibbs book
        veffq = convolution(self.pes.potential, self.qgrid, q, self.beta, self.m)
        return veffq    
    
    def dpotential(self,q):
        return None

    def ddpotential(self,q):
        return None

class Veff_classical_1D_FK(PES):
    """
    Feynman-Kleinert approximate potential, computed without the
    variational minimisation step and truncated at the quadratic term.
    """
    
    def __init__(self,pes,beta,m,hess=None):
        super(Veff_classical_1D_FK).__init__()  
        self.beta = beta 
        self.m=m
        self.pes=pes
        self.hess=hess
        
    def bind(self,ens,rp):
        super(Veff_classical_1D_FK,self).bind(ens,rp)
        
    def potential(self,q):
        #This provides an option to pass a customised hessian field if required
        if(self.hess is None):
            hess = self.pes.ddpotential(q)/self.m
        else:
            hess = self.hess(q)/self.m
        wa = 0.0
        if (hess >0):
            wa = np.sqrt(hess)
            xi = 0.5*self.beta*wa
            # Eq. 3.36 in text
            return self.pes.potential(q) + np.log(np.sinh(xi)/xi)/self.beta
        elif (hess <0):     
            wa = np.sqrt(-hess)
            xi = 0.5*self.beta*wa
            # Potential modified for negative hessian
            return self.pes.potential(q) + np.log(np.sin(xi)/xi)/self.beta
        elif (hess==0):
            return self.pes.potential(q)
    
    def dpotential(self,q):
        return None

    def ddpotential(self,q):
        return None


