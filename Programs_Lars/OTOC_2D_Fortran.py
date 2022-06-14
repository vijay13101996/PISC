import numpy as np
from numpy import linalg as LA
from mylib.twoD import DVR2D_mod
from mylib.twoD import OTOC2D
from mylib.twoD import PY_OTOC2D
import sys
sys.path.insert(0, "/home/lm979/Desktop/PISC")
from PISC.dvr import dvr
import time
time_0=time.perf_counter()

##################TEST DVR(Print Results/Energies)
#count number of occurences of HO energy levels 
def nbr_occ(w,norm=1,nr_E=50,print_energy_levels=False):
    epsilon=0.05
    cntr=np.zeros(nr_E)
    for l in range(len(cntr)):
        for k in range(len(w)):
            if(abs(w[k]/norm-l)<epsilon):
                cntr[l]+=1
    print(cntr[:])
    if(print_energy_levels==True):
        print(w[:50])


##########Parameters##########
hbar=1 #like in OTOC paper
m=1/2 #like in OTOC paper
L=1
pot_name='PiB'#PiB, HO, CHO
N_trunc=150
print('Potential: %s' % pot_name)

##########DVR##########
if(pot_name=='HO'):
    xy = np.mgrid[-10:10.000001:0.25, -10:10.000001:0.25] #HO until 20 (25)
    #xy= np.mgrid[-7:7.1:0.5, -7:7.1:0.5]#inaccurate for testing
    #xy= np.mgrid[-5:5.1:1, -5:5.1:1]#very inaccurate for testing
    V=DVR2D_mod.pot2D(xy,potential2D=DVR2D_mod.pot_2D_HO)
    T=DVR2D_mod.kin_inf_2D(xy,hbar=hbar, m=m)

if(pot_name=='CHO'):
    xy = np.mgrid[-13.0:13.01:0.25, -13.0:13.01:0.25] #until 13 approx
    #xy = np.mgrid[-12.0:12.01:0.3, -12.0:12.01:0.3] #until 13 approx
    V=DVR2D_mod.pot2D(xy,potential2D=DVR2D_mod.pot_2D_CHO)
    T=np.zeros_like(V)
    if(False):####Still need to implent if fast creation of T is needed
        N_x=len(xy[0,1,:])
        N_y=len(xy[0,:,1])
        N_tot= N_x*N_y
        time_t0= time.perf_counter()
        T=OTOC2D.otoc_2d.calc_t_mat(t=T, xy=xy,hbar=hbar,m=m,len_x=N_x,len_y=N_y,len_tot=N_tot )
        time_t1=time.perf_counter()
        print('T matrix generated in %.3f seconds!' % (time_t1-time_t0))
    else:
        T=DVR2D_mod.kin_inf_2D(xy,hbar=hbar, m=m)

if(pot_name=='PiB'):
    xy = np.mgrid[0:1.0000:0.05, 0:1.0000:0.05] #needs to be super fine if to infinity
    xy=xy[:,1:,1:]
    V=DVR2D_mod.pot2D(xy,potential2D=DVR2D_mod.pot_2D_PiB)
    #T=DVR2D_mod.kin_inf_2D(xy,hbar=hbar, m=m)
    T=DVR2D_mod.kin_box_2D(xy,hbar=hbar, m=m)
if(pot_name=='DWMorse'):
    D = 5.0#10.0
    alpha = 1#1.165#0.81#0.175#0.41#0.255#1.165
    lamda = 1.5#4.0
    g = 0.035#lamda**2/32#4.0
    z = 0.5#1.25#2.3	

    xy = np.mgrid[-12.0:12.01:0.4, -12.0:12.01:0.4] #until ?
    xy = np.mgrid[-12.0:12.01:0.4, -12.0:12.01:0.4] #until ?
    if(False):
        alpha_morse=1
        D=5
        c_const=0.1
        morse_pot=lambda a,b: DVR2D_mod.pot_2D_DWMorse(a,b,c_const=c_const,D=D,alpha_morse=alpha_morse)
        V=DVR2D_mod.pot2D(xy,potential2D=morse_pot)
    from PISC.potentials.Quartic_bistable import quartic_bistable
    pes = quartic_bistable(alpha,D,lamda,g,z)
    V=DVR2D_mod.pot2D(xy,potential2D=pes.potential_xy)
    T=DVR2D_mod.kin_inf_2D(xy,hbar=hbar, m=m)


##########DVR Diagonalization##########
N_x=len(xy[0,1,:])
N_y=len(xy[0,:,1])
N_tot= N_x*N_y

H=T+V
print()

print('Diagonalization (%i x %i) in ... ' % (N_tot,N_tot))
time_0=time.perf_counter()

w,v = LA.eigh(H) 
#from scipy.sparse import linalg as LA2
#w,v = LA2.eigsh(H,k=150, which='SM')
#VJsDVR=dvr.DVR2D(80,80,-10,10,-10,10,m=1/2, potential=DVR2D_mod.pot_2D_HO)
#vals, vecs = VJsDVR.Diagonalize()

time_1=time.perf_counter()
print('.... t = %f seconds! ' % (time_1-time_0))

#################OTOC functions
N_x=len(xy[0,:,0])
N_y=len(xy[1,0,:])

##########Generate X##########
time_x0=time.perf_counter()

X=np.zeros((N_trunc+1,N_trunc+1))
for n in range(len(X[:,0])):
    for m in range(len(X[:,0])):
        X[n,m]= PY_OTOC2D.x_nm(v=v,n=n,m=m,xy=xy)

time_x1=time.perf_counter()
print("Generated X in %.3f seconds!" % (time_x1-time_x0))

##########Generate B##########
if(pot_name=='PiB'):
    t=np.linspace(0,0.7,400)#PiB
elif(pot_name=='HO'):
    t=np.linspace(0,6.0,600)#HO
elif(pot_name=='CHO'):
    t=np.linspace(0,15.0,400)#CHO
elif(pot_name=='DWMorse'):
    t=np.linspace(0,10.0,400)#DWMorse
elif(pot_name=='Quartic_bistable'):
    #implement VJ's code and play with coupling strength... also make contour plot 
    print('use VJs code!')


B=np.zeros((N_trunc+1,N_trunc+1,len(t)),dtype=complex)
B=OTOC2D.otoc_2d.get_b_omp(B,X, w[:N_trunc+1],t,N_trunc+1, len(t))
if(False):#not OMP
    for n in range(len(X[:,0])):
        for m in range(len(X[:,0])):
            B[n,m,:] = PY_OTOC2D.b_nm(t=t, n=n,m=m, X=X,w=w)
B=B[:-1,:-1,:]

time_b1=time.perf_counter()
print("Generated B in %.3f seconds!" % (time_b1-time_x1))

##########Generate C (microcanonical)########## 
max_n=N_trunc-30 #compare to paper, CHO: N_trunc=253 and max_n=150
###cannot be as high as N Trunc
C_mc= np.zeros((max_n, len(t)))
for n in range(max_n):
    C_mc[n,:]= PY_OTOC2D.c_n(B=B,n=n,epsilon=0.1)#epsilon to check that the first value is correct (1+-epsilon)
    if(C_mc[n,0]==0):
        max_n=n-1
        break
C_mc=C_mc[:max_n,:]
print(max_n)

time_c1=time.perf_counter()
print("Generated C in %.3f seconds!" % (time_c1-time_b1))

if(True):#Plot MC OTOC
    import matplotlib.pyplot as plt
    #from matplotlib.ticker import ScalarFormatter
    fig, ax = plt.subplots()
    ax.plot(t,np.zeros_like(C_mc[2,:])+1, '--')#depends if log or not
    #for n in (1,2,5,10):#PiB
    for n in (1,5,9,11,13,20):#DWMorse
    #for n in (1,5,min(max_n-3,50),min(max_n-2,100),min(max_n-1,150)): 
        ax.plot(t,C_mc[n-1,:], label = "n = %i" % n) #n-1 because enumeration in paper starts at 1
    ax.set_ylabel('OTOC')
    #ax.set_yscale('log')
    #ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('t')
    #ax.set_xlim([0,4])
    ax.legend()
    plt.show()
    #plt.savefig('expl.png')


if(True):#Plot Thermal OTOC
    import matplotlib.pyplot as plt
    #from matplotlib.ticker import ScalarFormatter
    fig, ax = plt.subplots()
    ax.plot(t,np.zeros_like(C_mc[2,:])+1, '--')#depends if log or not
    #for T in (1,2,3,4,5): #CHO
    for T in (1,5,9,30):#DW
    #for T in (0.5,1,10,40,100): #HO, doesn't matter since all the same
    #for T in (1,20,50,100,200):#PiB 
        beta=1/T
        ax.plot(t,PY_OTOC2D.C_T(C=C_mc,w=w, beta=beta), label = "T = %.2f" %T)
        #ax.plot(t,np.log(C_T(C=C_mc,w=w, beta=beta)), label = "T = %.2f" %T)
    ax.set_ylabel('OTOC')
    #ax.set_yscale('log')
    #ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('t')
    #ax.set_xlim([0,4])
    ax.legend()
    plt.show()
    #plt.savefig('expl.png')
