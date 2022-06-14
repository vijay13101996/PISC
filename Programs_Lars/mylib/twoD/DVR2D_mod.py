#import sys
#sys.path.append("/home/lm979/Desktop/Programs/")
#sys.path.append("/home/lm979/Desktop/Programs/mylib")

import numpy as np
#from numpy import linalg as LA
import time
def pot_2D_PiB(x,y):
    V=0
    if(x>1 or x<0 or y<0 or y>1):
        V+=10000000   
    return V

def pot_2D_HO(x,y,m=1/2,omegax=1, omegay=1,x0=0,y0=0):
    return 0.5*m*omegax**2 *(x-x0)**2 +0.5*m*omegay**2 *(y-y0)**2

def pot_2D_CHO(x,y,g=1/10,omega=1,m=1/2):
    return 0.5*m* omega**2 * (x**2+y**2) + g * x**2 * y**2

def pot_2D_DWMorse(x,y,c_const, D,alpha_morse, g=1/50,labda=2):
    DW= -(1/4) * labda**2 * x**2 +g*x**4 + labda**4 / (64 *g)
    eaq = np.exp(-alpha_morse*y)
    Morse= D*(1-eaq)**2
    coupling= c_const*x**2 * y**2
    return Morse+DW+coupling
    

def pot2D(xy, potential2D):#=pot_2D_HO):#input: grid and generic potential
    time0=time.perf_counter()
    N_x= len(xy[0,:,1])
    N_y = len(xy[1,1,:])
    N_tot= N_x*N_y
    V = np.zeros((N_tot,N_tot)) 
    
    for nx in range(N_x):
        for ny in range(N_y):
            cntr=N_y*nx+ny
            V[cntr,cntr]=potential2D(xy[0,nx,ny],xy[1,nx,ny])
    time1=time.perf_counter()
    print("V matrix generated in %.3f seconds!"% (time1-time0))
    return V

def kin_inf_2D(xy,hbar=1,m=0.5):#(-inf,inf) intervall
    time0=time.perf_counter()
    N_x= len(xy[0,:,1])
    N_y = len(xy[1,1,:])
    N_tot= N_x*N_y
    T = np.zeros((N_tot,N_tot))
    konst_x=hbar**2 / (2*m* (xy[0,1,0]-xy[0,0,0])**2 )#for same grid sizing can be put at the end.. speedup necessary?
    konst_y=hbar**2 / (2*m* (xy[1,0,1]-xy[1,0,0])**2 )
    
    for i1 in range(N_x):#i1=i
        for j1 in range(N_y):#j1=j
            cntr1=N_y*i1+j1
    
            for i2 in range(N_x):#i2=i'
                for j2 in range(N_y):#j2=j'
                    cntr2=N_y*i2+j2
    
                    if(i1==i2):#delta(i,i')
                        #compute T(j,j')
                        if(j1==j2):
                            T[cntr1,cntr2]+=konst_y*np.pi**2/3
                        if(j1!=j2):
                            T[cntr1,cntr2]+=konst_y*((-1)**(j1-j2) *2/((j1-j2)**2))
                    if(j1==j2):#delta(j,j')
                        #compute T(i,i')
                        if(i1==i2):
                            T[cntr1,cntr2]+=konst_x*np.pi**2/3
                        if(i1!=i2):
                            T[cntr1,cntr2]+=konst_x*((-1)**(i1-i2) *2/((i1-i2)**2))
    time1=time.perf_counter()
    print("T matrix generated in %.3f seconds!"% (time1-time0))#Faster if same grid size and probably if if clauses better distributed
    return T

def kin_box_2D(xy,hbar=1,m=0.5,Lx=1,Ly=1):#(-inf,inf) intervall
    time0=time.perf_counter()
    N_x= len(xy[0,:,1])
    N_y = len(xy[1,1,:])
    N_tot= (N_x)*(N_y)
    T = np.zeros((N_tot,N_tot))
    konst_x= hbar**2 * np.pi**2 / (2*2*m*Lx**2)
    konst_y= hbar**2 * np.pi**2 / (2*2*m*Ly**2)
    #konst_x=hbar**2 / (2*m* (xy[0,1,0]-xy[0,0,0])**2 )#for same grid sizing can be put at the end.. speedup necessary?
    #konst_y=hbar**2 / (2*m* (xy[1,0,1]-xy[1,0,0])**2 )
    N_i=Lx/(xy[0,1,0]-xy[0,0,0])
    N_j=Ly/(xy[1,0,1]-xy[1,0,0])

    for i1 in range(N_x):#i1=i
        for j1 in range(N_y):#j1=j

            cntr1=(N_y)*i1+j1
    
            for i2 in range(N_x):#i2=i'
                for j2 in range(N_y):#j2=j'
                    cntr2=(N_y)*i2+j2
    
                    if(i1==i2):#delta(i,i')
                        #compute T(j,j')
                        if(j1==j2):
                            T[cntr1,cntr2]+=konst_y*((2*N_j**2 +1)/3 - 1/(np.sin(np.pi*(j1+1)/N_j)**2) )
                        if(j1!=j2):
                            T[cntr1,cntr2]+=konst_y*((-1)**(j1-j2))* ( 1/(np.sin(np.pi*(j1-j2)/(2*N_j))**2)- 1/(np.sin(np.pi*(j1+j2+2)/(2*N_j))**2) )
                    if(j1==j2):#delta(j,j')
                        #compute T(i,i')
                        if(i1==i2):
                            T[cntr1,cntr2]+=konst_x*((2*N_i**2 +1)/3 - 1/(np.sin(np.pi*(i1+1)/N_i)**2) )
                        if(i1!=i2):
                            T[cntr1,cntr2]+=konst_x*((-1)**(i1-i2))* ( 1/(np.sin(np.pi*(i1-i2)/(2*N_i))**2)- 1/(np.sin(np.pi*(i1+i2+2)/(2*N_i))**2) )
    time1=time.perf_counter()
    print("T matrix generated in %.3f seconds!"% (time1-time0))#Faster if same grid size and probably if "if clauses" better distributed
    return T
