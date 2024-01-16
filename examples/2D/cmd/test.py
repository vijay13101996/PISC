import numpy as np
from PISC.potentials.DW_harm_2D import DW_harm
from matplotlib import pyplot as plt


x = np.linspace(-5,5,100)
y = np.linspace(-10,10,100)

X,Y = np.meshgrid(x,y)

m = 0.5
w = 10
lamda = 2.0
g = 0.08

Vb = lamda**4/(64*g)

z = 0.0

dw = DW_harm(m,w,lamda,g,z)

V = dw.potential_xy(X,Y)

plt.contour(X,Y,V,levels=np.linspace(0,1.05*Vb,100))
#plt.contour(X,Y,V,levels=np.linspace(-1,0,100))
plt.show()


