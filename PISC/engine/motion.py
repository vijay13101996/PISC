"""
This module contains necessary definitions for 
implementing path-integral dynamics.
"""

import numpy as np

class Motion(object):

	def __init__(self,dt,symporder):
		self.dt=dt
		self.order = symporder
		if(self.order==2):
			self.qdt = 1.0*self.dt#np.array([1.0,0.0])*self.dt
			self.pdt = 0.5*self.dt#np.array([0.5,0.5])*self.dt
		elif(self.order==4):
			self.qdt = np.array([0.2113248654,0.57735026919,-0.57735026919,0.78867513459])*self.dt
			self.pdt = np.array([0.0,0.53867513459,0.5,-0.03867513459])*self.dt
