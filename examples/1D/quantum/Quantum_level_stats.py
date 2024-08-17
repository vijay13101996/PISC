import numpy as np
from matplotlib import pyplot as plt
from PISC.potentials import quartic
from PISC.dvr.dvr import DVR1D

ngrid = 2000

lb = -20.0
ub = 20.0
m = 1

pes = quartic(1.0)


DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
vals,vecs = DVR.Diagonalize() 


diff = np.diff(vals)

plt.hist(diff,bins=40,density=True)
plt.show()




