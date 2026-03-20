import numpy as np
from PISC.potentials.base import PES
from matplotlib import pyplot as plt

class cmd_pmf(PES):
    def __init__(self,lamda,g,times,nbeads,pathname):
        super(cmd_pmf).__init__()
        self.lamda = lamda
        self.g = g
        self.times = times
        self.Tc = 0.5*self.lamda/np.pi

        fname = '{}/Datafiles/CMD_PMF_LONGER_inv_harmonic_T_{}Tc_nbeads_{}.txt'.format(pathname,times,nbeads)
        data = np.loadtxt(fname,dtype=complex)
        self.qgrid = np.real(data[:,0])
        self.fgrid = np.real(data[:,1])

        self.coeff = np.polyfit(self.qgrid,self.fgrid,10)
        self.poly = np.poly1d(self.coeff)
        
        self.polyint = np.polyint(self.coeff,k=0)
        self.polypot = np.poly1d(self.polyint)

        #plt.plot(self.qgrid,self.fgrid)
        #plt.plot(self.qgrid,self.poly(self.qgrid))
        #plt.show()
        #exit()

    def bind(self,ens,rp,pes_fort=False,transf_fort=False):
        super(cmd_pmf,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
    
    def potential(self,q):
        q = q[:,0,:]
        return self.polypot(q)

    def dpotential(self,q):
        return self.poly(q)

    def ddpotential(self,q):
        return self.poly.deriv()(q)

    
