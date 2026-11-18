"""
This module contains necessary definitions for 
implementing path-integral dynamics.
"""

import numpy as np

class Motion(object):

    def __init__(self,dt=1e-2,symporder=2):
        self.dt=dt
        self.order = symporder
        if(self.order==2):
            """ Velocity Verlet splitting scheme """""
            self.qdt = np.array([1.0,0.0])*self.dt
            self.pdt = np.array([0.5,0.5])*self.dt
        elif(self.order==4):
            """ 4th order splitting scheme from: 
                "Semiclassical dynamics in up to 15 coupled vibrational degrees of freedom", 
                J. Chem. Phys. 106, 4832-4839 (1997)
            """
            self.qdt = np.array([0.2113248654,0.57735026919,-0.57735026919,0.78867513459])*self.dt
            self.pdt = np.array([0.0,0.53867513459,0.5,-0.03867513459])*self.dt
