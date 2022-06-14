#import sys
#sys.path.append("/home/lm979/Desktop/Programs/")
#sys.path.append("/home/lm979/Desktop/Programs/mylib")
import numpy as np
from numpy import linalg as LA

from mylib.oneD import Potentials_and_DVR as PnDVR

def DVR_initialize(pot_name,m,hbar):
    if(pot_name=='PiB'):### PiB
        ##self consistency: HO and PiB are known
        grid = np.linspace(-3,3,1500,endpoint=False)[1:] #in principle: gridspacing one to 5 grid points per de broglie wavelength and where one expects wavefkt to still be
        H=PnDVR.pot(grid=grid, potential=PnDVR.pot_1d_well)+PnDVR.kin_inf(grid=grid,hbar=hbar,m=m)
        N_tsteps=400
        t=np.linspace(0,0.7,N_tsteps)
        log_OTOC_T=False
        log_OTOC_mic=False
    elif(pot_name=='HO'):### Harmonic Oscillator
        ##self consistency: HO and PiB are known
        grid = np.linspace(-25,25,1500,endpoint=False)[1:] #in principle: gridspacing one to 5 grid points per de broglie wavelength and where one expects wavefkt to still be
        spring=1
        H=PnDVR.pot(grid=grid, potential=lambda x : PnDVR.pot_harm_osz(x=x,k=spring,x0=0))+PnDVR.kin_inf(grid=grid,hbar=hbar,m=m)
        N_tsteps=300
        t=np.linspace(0,7,N_tsteps)# HO
        log_OTOC_T=False
        log_OTOC_mic=False
    elif(pot_name=='DW'):###Double Well
        #DW: Check for highest EV intersection with y axis, then check if energies change using more grdpoints
        #given parameters: N=50 E=136, N=100 E=519, N=150 E=1154: checked: need 11,14,17 
        grid = np.linspace(-25,25,2500,endpoint=False)[1:] #in principle: gridspacing one to 5 grid points per de broglie wavelength and where one expects wavefkt to still be
        H=PnDVR.pot(grid=grid, potential=lambda x : PnDVR.pot_double_well(x=x,labda= 2, g=1/50))+PnDVR.kin_inf(grid=grid,hbar=hbar,m=m)
        N_tsteps=100
        t=np.linspace(0,4,N_tsteps)# DW
        log_OTOC_T=True
        log_OTOC_mic=True

    else:
        print('Choose Valid potential') # throw with try and catch much more fancy

    return H, grid, t, log_OTOC_mic, log_OTOC_T
def DVR_all_inclusive(pot_name,grid,t,m,hbar):
    if(pot_name=='PiB'):### PiB
        H=PnDVR.pot(grid=grid, potential=PnDVR.pot_1d_well)+PnDVR.kin_inf(grid=grid,hbar=hbar,m=m)
        log_OTOC_T=False
        log_OTOC_mic=False
    elif(pot_name=='HO'):### Harmonic Oscillator
        spring=1
        H=PnDVR.pot(grid=grid, potential=lambda x : PnDVR.pot_harm_osz(x=x,k=spring,x0=0))+PnDVR.kin_inf(grid=grid,hbar=hbar,m=m)
        log_OTOC_T=False
        log_OTOC_mic=False
    elif(pot_name=='DW'):###Double Well
        H=PnDVR.pot(grid=grid, potential=lambda x : PnDVR.pot_double_well(x=x,labda= 2, g=1/50))+PnDVR.kin_inf(grid=grid,hbar=hbar,m=m)
        log_OTOC_T=True
        log_OTOC_mic=True
    else:
        print('Choose Valid potential') # throw with try and catch much more fancy
    w,v = LA.eigh(H)
    return w,v, log_OTOC_mic, log_OTOC_T