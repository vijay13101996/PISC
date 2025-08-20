import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det


ngrid = 1000

L=10 
lb=0
ub=L

lbc=0
ubc=L

m=0.5

print('L',L)

potkey = '1D_Box_m_{}_L_{}'.format(m,np.around(L,2))

anal = True

#----------------------------------------------------------------------
def potential(x):
    if(x<lbc or x>ubc):
        return 1e12
    else:
        #print('x',x)
        return 0.0#x**4

neigs = 300
potential = np.vectorize(potential)


DVR = DVR1D(ngrid, lb, ub,m, potential)

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

print('Using analytical pos_mat, vals')
vals = vals_anal
O = O_anal

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

T_arr = [1]#np.arange(10.0,40.01,5.0)
mun_arr = []
mu0_harm_arr = []
bnarr = []
nmoments = 60
ncoeff = 100

mu_all_arr = []

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
    mu_all_arr.append(moments)

mun_arr = np.array(mun_arr) #contains mu_0, mu_2, mu_4, ...
bnarr = np.array(bnarr)
print('mun_arr',mun_arr.shape)
print('bnarr',bnarr.shape)

if(1):
    for i in [0]:#,1,2,3,4,5,6]:#,6,8,10,12,14,16]:
        plt.scatter(np.arange(ncoeff),bnarr[i,:],label='T={}'.format(T_arr[i]))
    
    #plt.xlim([10,nmoments//2])
    
    #plt.title(r'$neigs={},L={}$'.format(neigs,ub))
    
    #plt.title(r'$\omega={},a={},b={},n={}$'.format(omega,a,b,n_anharm))
    plt.xlabel(r'$n$')
    plt.ylabel(r'$b_n$')
    #plt.ylim([0.0,2000])
    plt.legend()    
    plt.show()
    exit()

store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))


start = 5
narr = np.arange(start,nmoments//2+1)
nlognarr = narr*np.log(narr)
logmun_arr = np.log(mun_arr)[:,start:]

term1 = logmun_arr - 2*nlognarr[np.newaxis,:]

gamma = 0.57721566
b = 5*np.pi/2#7.765

slope_arr = np.zeros(len(T_arr))
offset_arr = np.zeros(len(T_arr))

for i in range(len(T_arr)):
    T = T_arr[i]

    logmun_anal = 2*nlognarr + (gamma + 2*np.log(T))*narr - b - np.log(T)*np.pi/2
    print('logmun_anal',logmun_anal[15:20],'logmun_num',logmun_arr[i,15:20])


    term1_num = term1[i,:]
    K1T = gamma + 2*np.log(T)
    K2T = -b - np.log(T)*np.pi/2
    term1_anal = K1T*narr + K2T

    #print('term1_num',term1_num[15:20],'term1_anal',term1_anal[15:20])

    #plt.scatter(narr,term1[i,:],label='T_{}'.format(T_arr[i]))

    p = np.polyfit(narr,term1[i,:],1)
    slope = p[0]
    offset = p[1]
    #print('T',T,'slope',slope,'offset',offset)
    slope_arr[i] = slope
    offset_arr[i] = offset

    slope_anal = K1T
    offset_anal = K2T
    #print('slope_num',slope,'slope_anal',slope_anal)
    #print('offset_num',offset,'offset_anal',offset_anal,'\n')
    
    #plt.plot(narr,slope_anal*narr+offset_anal,label='T={}'.format(T))
    #plt.plot(narr,term1_anal,label='T_{}_anal'.format(T_arr[i]))

#plt.show()
#exit()

logT_arr = np.log(T_arr)

#Fit slope_arr vs logT_arr to a line
p = np.polyfit(logT_arr,slope_arr,1)
K1_s = p[0]
K1_o = p[1]

print('K1_s',K1_s,'K1_o',K1_o)

p = np.polyfit(logT_arr,offset_arr,1)
K2_s = p[0]
K2_o = p[1]

print('K2_s',K2_s,'K2_o',K2_o)

#plt.scatter(logT_arr,slope_arr)
plt.scatter(logT_arr,offset_arr)
plt.plot(logT_arr,K2_s*logT_arr+K2_o)
plt.show()
