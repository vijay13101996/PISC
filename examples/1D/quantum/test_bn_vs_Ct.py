import numpy as np
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from Krylov_WF_tools import coh_st_coeff, fix_vecs, comp_Ot, Cn, av_O, find_coeff_wf, verify_Ct, get_phi_t


tarr = np.linspace(0, 100, 500)
beta = 1.0


ncoeff = 50
barr = np.arange(1,ncoeff+1)*np.pi/beta


phi_t = get_phi_t(tarr, barr)

plt.plot(tarr, phi_t[0], label='$\phi(t)$ n=0')
plt.show()

