import numpy as np#

"""
(Pseudo-)code for a trial run of MF Matsubara simulation of OTOC.


1. Set beta, nbeads, nmats,  
2. Set time of propagation, timestep, order of the symplectic integrator.
3. Declare all class objects and invoke 'bind'
4. Thermalize using 2nd order symplectic NVT steps.
5. Run 4th order symplectic code for the desired time and record Mqq at 
   regular intervals of time. 

"""

 
