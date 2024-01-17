import numpy as np
from PISC.potentials.base import PES

class PotAdder(PES):
    """
    Class that adds a potential to a custom function.
    This is most helpful when an additional 'filter' (that depends only 
    on the position) needs to be applied to a canonical ensemble.
    """
    def __init__(self, pes, func):
        self.pes = pes
        self.func = func

    def bind(self,ens,rp,pes_fort=False,transf_fort=False):
        super(PotAdder,self).bind(ens,rp,pes_fort=pes_fort,transf_fort=transf_fort)
    
    def potential(self, q):
        return self.pes.potential(q) + self.func.potential(q)

    def dpotential(self, q):
        return self.pes.dpotential(q) + self.func.dpotential(q)

    def ddpotential(self, q):
        return self.pes.ddpotential(q) + self.func.ddpotential(q)
