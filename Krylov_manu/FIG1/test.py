import numpy as np
from matplotlib import pyplot as plt

n,bn = np.loadtxt('/home/vgs23/PISC/Krylov_manu/FIG1/FIG1_Exponential_beta1.txt', unpack=True)
plt.plot(n, bn, 'o', label='Data Points')

n,bn = np.loadtxt('/home/vgs23/PISC/Krylov_manu/FIG1/FIG1_Gaussian_beta1.txt', unpack=True)
plt.plot(n, bn, 's', label='Data Points')

n,bn = np.loadtxt('/home/vgs23/PISC/Krylov_manu/FIG1/FIG1_Powerlaw_beta1.txt', unpack=True)
plt.plot(n, bn, '^', label='Data Points')


plt.xlabel('n')
plt.ylabel('beta_n')
plt.title('Plot of beta_n vs n')
plt.legend()
plt.grid()
plt.show()
