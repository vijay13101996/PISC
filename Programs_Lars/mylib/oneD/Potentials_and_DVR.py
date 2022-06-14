import numpy as np
def pot_1d_well(x):
    V=0
    if(x>1):
        V+=100000   
    if(x<0):
        V+=100000
    return V

def pot_harm_osz(x,k,x0):
    return 0.5*k* (x-x0)**2

def pot_double_well(x, labda=2,g=1/50):
    return -(1/4) * labda**2 * x**2 +g*x**4 + labda**4 / (64 *g)
def kin_inf(grid,hbar=1,m=1):#(-inf,inf) intervall
    N_grid=len(grid)
    T = np.zeros((N_grid,N_grid)) 
    for k in range(N_grid):
        for l in range(N_grid):
            if(k==l):
                T[k,k]= np.pi**2 /3
            if(k!=l):
                T[k,l]= (-1)**(k-l) *2/((k-l)**2)
    T *= hbar**2 / (2*m* (grid[2]-grid[1])**2 )
    return T

def pot(grid, potential):#input: grid and generic potential
    N_grid=len(grid)
    V = np.zeros((N_grid,N_grid)) 
    for k in range(N_grid):
        V[k,k]= potential(grid[k])
    return V


def C_T(C,w,beta=1): #C is microcanonical OTOC
    Z=0 
    sum=np.zeros_like(C[1,:])#get Thermal OTOC for each time t
    for n in range(len(C[:,1])):
        Z+=np.exp(-beta *w[n]) #partition sum
    for n in range(len(C[:,1])):
        for time in range(len(C[1,:])):
            sum[time] += np.exp(-beta *w[n])*C[n,time] #wights of all microcanonical OTOCS
    return sum/Z

