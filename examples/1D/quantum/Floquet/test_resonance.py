import numpy as np
from matplotlib import pyplot as plt


N = 20
lamda = 10.0
a_arr = np.zeros(N) + 0j

f = np.zeros((N, N)) + 0j

for j in range(N):
    f[j, j] = j^2
    if(j>0):
        f[j, j-1] = lamda
    if(j<N-1):
        f[j, j+1] = lamda

#Solve the eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eig(f)

for i in range(N):
    plt.plot(np.real(eigenvectors[:, i]), label=f'Eigenvector {i+1}')

    plt.title('Eigenvectors of the Matrix')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()



