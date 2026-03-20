import numpy as np
from matplotlib import pyplot as plt
import glob

direc = '/home/vgs23/PISC/Krylov_manu/FIG3/'
# List all text files in the directory by using glob
file_lst =  glob.glob(direc + '*.txt')

#print(file_lst)
for file in file_lst:
    
    # Load from line 2 onwards
    data = np.loadtxt(file, skiprows=1)
    n = data[:, 0]
    b_n = data[:, 1]
    plt.plot(n, b_n)#, label=file.split('/')[-1])


plt.xlabel('n')
plt.ylabel('beta_n')
plt.title('Plot of beta_n vs n')
plt.legend()
plt.grid()
plt.show()
