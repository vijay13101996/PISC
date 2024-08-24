import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt

ngrid = 1000

L=10#4*np.sqrt(1/(4+np.pi))#10
lb=0
ub=L

lbc = 0
ubc = L

m=0.5

print('L',L)

potkey = '1D_Box_m_{}_L_{}'.format(m,np.around(L,2))


anal = True
#anal = False

def potential(x):
    if(x<lbc or x>ubc):
        return 1e12
    else:
        #print('x',x)
        return 0.0#x**4

neigs = 550
potential = np.vectorize(potential)

#xgrid = np.linspace(lb,ub,ngrid)
#plt.plot(xgrid,potential(xgrid))
#plt.ylim([0,1000])
#plt.show()

#----------------------------------------------------------------------

DVR = DVR1D(ngrid, lb, ub,m, potential)
if(not anal):
    vals,vecs = DVR.Diagonalize(neig_total=neigs)


x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

#----------------------------------------------------------------------

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:]/L)

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

#----------------------------------------------------------------------

print('vals',vals_anal[-1])

if(not anal):
    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

if(anal):
    print('Using analytical pos_mat, vals')
    vals = vals_anal
    O = O_anal
else:
    print('Using numerical pos_mat, vals')
    vals = vals
    O = pos_mat

i = 4
j = 4
#print('O_anal',O_anal[i,j])
#print('O',O[i,j])

#print('O_anal',np.around(O_anal[:10,:10],3))
#print('O',np.around(O[:10,:10],3),'vals',vals[-1])

#exit()

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

T_arr = [5,10,20,30,40,100]#np.arange(1.,30.05,2.)#[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
mun_arr = []
mu0_harm_arr = []
bnarr = []
nmoments = 60
ncoeff = 200

for T_au in T_arr:
    
    Tkey = 'T_{}'.format(T_au)

    beta = 1.0/T_au 

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')
    bnarr.append(barr)

    mun_arr.append(even_moments)


mun_arr = np.array(mun_arr)
bnarr = np.array(bnarr)
print('mun_arr',mun_arr.shape)
print('bnarr',bnarr.shape)

store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))
#exit()

if(1):
    narr = np.arange(1,nmoments//2+1)
    nlogn = narr#*np.log(narr)
    for i in [0]:#,1,2,3]:#4,6,8,10,12,14,16]:
        plt.scatter(nlogn,np.log(mun_arr[i,1:]),label='T={}'.format(np.around(T_arr[i],2)))
        #plt.plot(nlogn, (3*narr-2)*np.log((2*np.pi*T_arr[i])),label='T={}'.format(np.around(T_arr[i],2)))
        #Fit to a line
        #p = np.polyfit(nlogn,np.log(mun_arr[i,5:]),1)
        #print('p',p,nlogn,np.log(mun_arr[i,10:]))
        #plt.plot(nlogn,p[0]*nlogn+p[1],label='T={}'.format(np.around(T_arr[i],2)))
        
        #plt.scatter(narr,4*narr*np.log(narr),label='nlog(n)')

    #plt.xlim([10,nmoments//2])
    
    plt.title(r'$neigs={}$'.format(neigs))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mu_{2n}$')
    #plt.ylim([0.0,2000])
    plt.legend()    
    plt.show()
    exit()

if(0):
    for i in [0,1,2,3]:#range(0,len(T_arr),2):
        plt.scatter(np.arange(ncoeff),bnarr[i,:200],label='T={}'.format(T_arr[i]))
        #plt.scatter(np.log(np.arange(1,nmoments//2+1)),np.log(mun_arr[i,1:]),label='T={}'.format(T_arr[i]))

    #plt.xlim([10,nmoments//2])
    
    if(anal):
        plt.title(r'$neigs={},{}$'.format(neigs,'analytical'))
    else:
        plt.title(r'$neigs={},{}$'.format(neigs,'numerical'))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$b_n$')
    #plt.ylim([0.0,700])
    plt.legend()    
    plt.show()
    exit()

slope_arr = []
mom_list = range(0,nmoments//2+1)
for i in mom_list:#,5,6,7,8,9,10]:#,3,4]:#range(0,ncoeff,2):
    #plt.scatter(T_arr,bnarr[:,i],label='n={}'.format(i))
    
    #plt.scatter((T_arr),(mun_arr[:,i]),label='n={}'.format(2*i))
    plt.scatter(np.log(T_arr),np.log(mun_arr[:,i]),label='n={}'.format(2*i))

    lT_arr = np.log(T_arr)
    lmun_arr = np.log(mun_arr[:,i])

    p = np.polyfit(lT_arr,lmun_arr,1)

    slope_arr.append(p[0])
    
    plt.plot(lT_arr,p[0]*lT_arr+p[1],label='n={}'.format(2*i))


    # Fit mun_arr[:,i] to a T_arr**(i/2)
    #p = np.polyfit(T_arr,mun_arr[:,i],i/2)
    #print('p',p)
    #plt.plot(T_arr,p[0]*T_arr**(i/2),label='n={}'.format(2*i))

#plt.scatter(T_arr,np.array(mu0_harm_arr),label='n=0, harm',color='black')
plt.xlabel(r'$log(T)$')
plt.ylabel(r'$log(\mu_{2n})$')
plt.legend()
plt.show()

exit()

plt.scatter(mom_list,np.array(slope_arr))
plt.xlabel(r'$n$')
plt.ylabel(r'$slope$')
plt.show()

if(0):
    for i in [1,2,3]:#range(1):#nmoments//2+1):
        # Fit times_arr vs mun_arr[:,i] to a line
        p = np.polyfit(times_arr,mun_arr[:,i],1)
        print('p',p)
        plt.plot(times_arr,p[0]*times_arr+p[1],label='n={}'.format(2*i))
        plt.scatter(times_arr,mun_arr[:,i],label='n={}'.format(2*i))
    #plt.scatter(times_arr,mun_arr)
    plt.legend()
    plt.show()
    

#ncoeffs = 20
#barr = np.zeros(ncoeffs)
#barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 'wgm')

#b0 = np.sqrt(1/(2*m*w*np.sinh(0.5*beta*w)))

#print('barr',barr,b0)

#plt.scatter(np.arange(ncoeffs),barr)
#plt.show()


