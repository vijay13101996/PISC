import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det


ngrid = 2000

L=40 #4*np.sqrt(1/(4+np.pi))#10
lb=0
ub=L

lbc=0
ubc=L

m=1.0#0.5

print('L',L)

potkey = 'TEMP_1D_Box_m_{}_L_{}'.format(m,np.around(L,2))

anal = True
#anal = False

def potential(x):
    if(x<lbc or x>ubc):
        return 1e12
    else:
        #print('x',x)
        return 0.0#x**4

neigs = 400
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

if(0):
    for k in [0,3,5]:#np.arange(0,10,2):
        plt.plot(abs(np.diag(O,k))[:],label='k={}'.format(k))
    plt.legend()
    plt.show()
    exit()


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

T_arr = [0.01]#
mun_arr = []
mu0_harm_arr = []
bnarr = []
nmoments = 100
ncoeff = 200

mu_all_arr = []

On = np.zeros((neigs,neigs))
nmat = 10 

ip = 'fta'

for T_au in T_arr:
    
    Tkey = 'T_{}'.format(T_au)

    beta = 1.0/T_au 

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, ip)
    bnarr.append(barr)

    if(0):
        for nmat in range(10,200,10):#range(100):
            barr_mat = np.zeros(ncoeff)
            barr_mat, On = Krylov_complexity.krylov_complexity.compute_on_matrix(O, L, barr, beta, vals, ip, On, nmat+1) 
        
            On2 = np.matmul(On,On.T)
            logOn = np.log(np.abs(On))
            logOn2 = np.log(np.abs(On2))
            plt.imshow(np.abs(logOn2))#vmax=1e4)
            plt.show()

            print('trace', nmat, np.trace(On2), np.linalg.norm(On)**2)
            #trace = np.trace(On2)
            #plt.scatter(nmat,np.log(trace))
            #plt.title(r'$O_{:d}$'.format(nmat))
        plt.legend()
        plt.show()
        exit()

    mun_arr.append(even_moments)
    mu_all_arr.append(moments)

mun_arr = np.array(mun_arr)
bnarr = np.array(bnarr)
print('mun_arr',mun_arr.shape)
print('bnarr',bnarr.shape)


store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))

print(np.arange(nmoments//2+1), mun_arr[0,-1])

#plt.scatter(np.arange(nmoments//2+1),np.log(mun_arr[0,:]))
plt.scatter(np.arange(ncoeff),bnarr[0,:])
plt.show()

exit()

if(0):
    narr = np.arange(1,nmoments//2+1)
    nlogn = narr*np.log(narr)
    
    logmun_arr = np.log(mun_arr)[:,1:]

    slope_arr = []

    for i in [5]:
        T = T_arr[i]
        gamma = 0.57721566490
        temp = ((narr*T)**(2*narr))*np.exp(gamma*narr)/(T**(np.pi/2)*np.exp(5*np.pi/2))
        plt.plot(narr, np.log(temp),label='T={}'.format(np.around(T,2)))
        plt.plot(narr, logmun_arr[i,:],label='T={}'.format(np.around(T,2)))


    plt.show()
    exit()



    for i in range(len(T_arr)):#[0,1,2,3]:#4,6,8,10,12,14,16]:
        temp = logmun_arr[i,:] - 2*nlogn
        plt.scatter(narr[5:],temp[5:],label='T={}'.format(np.around(T_arr[i],2)))

        #Fit to a line
        p = np.polyfit(narr[5:30],temp[5:30],1)
        slope_num = p[0]
        off_num = p[1]

        gamma = 0.57721566490
        slope_anal = 2*np.log(T_arr[i]) + gamma
        off_anal = -0.5*np.pi*np.log(T_arr[i]) - np.exp(1)**2

        #print('slope_anal',slope_anal,'slope_num',slope_num)
        print('off_anal',off_anal,'off_num',off_num,'eee') 
 
        plt.plot(narr[5:],slope_anal*narr[5:]+p[1],label='T={}'.format(np.around(T_arr[i],2)))
        slope_arr.append(p[1])
        

        #plt.plot(narr[5:],slope*np.ones(len(narr[5:])),label='slope={}'.format(np.around(slope,2)))

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

    plt.scatter((T_arr),slope_arr)
    #plt.plot(T_arr, 2*np.log(T_arr) - 0.54)
    plt.xscale('log')
    
    #Fit slope_arr to log(T_arr)
    p = np.polyfit(np.log(T_arr),slope_arr,1)
    plt.plot(T_arr,p[0]*np.log(T_arr)+p[1],label='slope={}'.format(p[0]))
    print('p',p)

    plt.xlabel(r'$T$')
    plt.ylabel(r'$slope$')
    plt.show()
    exit()

if(1):
    slope_arr = []
    for i in [0]:#range(0,len(T_arr),2):
        #plt.scatter(np.arange(ncoeff),bnarr[i,:200],label='T={}'.format(T_arr[i]),s=3)
        plt.scatter((np.arange(1,nmoments//2+1)),(mun_arr[i,1:]),label='T={}'.format(T_arr[i]))
        
        # Find slope
        #slope = np.polyfit((np.arange(5,50)),(bnarr[i,5:50]),1)
        #slope_arr.append(slope[0])
    
        #print('slope',slope, np.pi*T_arr[i])
        #plt.plot(np.arange(5,50),slope[0]*np.arange(5,50)+slope[1],lw=2.5)

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

    plt.scatter(T_arr,slope_arr)
    plt.plot(T_arr, np.pi*np.array(T_arr))
    plt.xlabel(r'$T$')
    plt.ylabel(r'$slope$')
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


