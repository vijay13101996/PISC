import numpy as np

def x_nm(v,n,m,xy):
    N_x= len(xy[0,1,:])
    N_y=len(xy[0,:,1])
    sum_x=0
    v1=v.reshape((N_x,N_y,N_x*N_y))#comment out if want to use loop
    sum_x=np.sum(v1[:,:,n]*v1[:,:,m]*xy[0,:,:])
    if(False):#yields same result
        sum_x=0
        sum_y=0
        for nx in range(N_x):
            for ny in range(N_y):
                cntr=N_y*nx+ny
                sum_x += xy[0,nx,ny]*v[cntr,n]*v[cntr,m]
    return sum_x
def x_nm_2(v,n,m,xy):#also gives the same results
    N_x=len(xy[0,1,:])
    N_y=len(xy[0,:,1])
    dx= xy[0,0,0]-xy[0,1,0]
    dy= xy[1,0,1]-xy[1,0,0]
    sum_x=0
    sum_y=0
    sum_non=0
    for nx in range(N_x):
        for ny in range(N_y):
            cntr=N_y*nx+ny
            sum_non += v[cntr,n]*v[cntr,n]*dx*dy
            sum_x += xy[0,nx,ny]*v[cntr,n]*v[cntr,m]*dx*dy 
    #print(sum_non)
    #print(sum_x)
    #print(sum_x/sum_none)
    return sum_x/sum_non,


def b_nm(t,n,m,X,w): #is in Fortran 
    I=0
    for k in range(len(X[:,1])):
        I+=(X[n,k]*X[k,m])*((w[k]-w[m])*np.exp((w[n]-w[k])*1j*t)-(w[n]-w[k])*np.exp((w[k]-w[m])*1j*t))
    return 0.5*I

def c_n(B,n,epsilon=0.1): #microcanonical OTOC
    I=0
    for m in range(len(B[:,1,0])):#sum over N_trunc
        I += np.abs(B[n,m,:])**2
    if(abs(I[0]-1)>epsilon):
        print(I[0], n)
        I=0
    return I
def C_T(C,w,beta=1): #C is microcanonical OTOC
    Z=0 
    sum=np.zeros_like(C[1,:])#get Thermal OTOC for each time t
    for n in range(len(C[:,1])):
        Z+=np.exp(-beta *w[n]) #partition sum
    for n in range(len(C[:,1])):
        for time in range(len(C[1,:])):
            sum[time] += np.exp(-beta *w[n])*C[n,time] #weights of all microcanonical OTOCS
    return sum/Z