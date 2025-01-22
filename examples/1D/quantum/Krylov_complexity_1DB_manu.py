import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12


ngrid = 1000

L=10 #4*np.sqrt(1/(4+np.pi))#10
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

neigs = 100
potential = np.vectorize(potential)

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

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

LO = np.zeros((neigs,neigs))
LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

T_arr = np.array([1,2,3,4,5])
mun_arr = []
mu0_harm_arr = []
bnarr = []
nmoments = 100
ncoeff = 120

mu_all_arr = []

On = np.zeros((neigs,neigs))
nmat = 10 

for T_au in T_arr: 
    Tkey = 'T_{}'.format(T_au)

    beta = 1.0/T_au 

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, 'wgm', 0.5, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5,'wgm')
    bnarr.append(barr)

    mun_arr.append(even_moments)
    mu_all_arr.append(moments)

mun_arr = np.array(mun_arr)
bnarr = np.array(bnarr)
mu_all_arr = np.array(mu_all_arr)
print('mun_arr',mun_arr.shape)
print('bnarr',bnarr.shape)

store_arr(T_arr,'T_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(mun_arr,'mun_arr_{}_neigs_{}'.format(potkey,neigs))
store_arr(bnarr,'bnarr_{}_neigs_{}'.format(potkey,neigs))


#plt.scatter(np.arange(nmoments//2+1),np.log(mun_arr[0,:]))
#plt.scatter(np.arange(nmoments+1),np.log(mu_all_arr[0,:]))

fig,ax = plt.subplots(1,2,sharex=False,sharey=False)
fig.subplots_adjust(wspace=0.3,hspace=0.3)


ax[0].annotate(r'$(a)$', xy=(0.02, 0.9), xytext=(0.02, 0.9), textcoords='axes fraction', fontsize=xl_fs)
ax[1].annotate(r'$(b)$', xy=(0.02, 0.9), xytext=(0.02, 0.9), textcoords='axes fraction', fontsize=xl_fs)

for i in range(len(T_arr)):
    ax[0].scatter(np.arange(ncoeff),bnarr[i,:],s=10,label=r'$T={}$'.format(T_arr[i]))
    
    #Find slope
    p = np.polyfit(np.arange(ncoeff)[4:17],bnarr[i,4:17],1)
    print('p',p)

    ax[1].scatter(T_arr[i],p[0],zorder=4)

#Draw horizontal line at y=Emax/2
ax[0].hlines(0.5*vals[-1],0,ncoeff,linestyles='dashed',lw=2,color='0.3')

ax[0].set_xlabel(r'$n$',fontsize=xl_fs)
ax[0].set_ylabel(r'$b_n$',fontsize=yl_fs)
#ax[0].legend(fontsize=le_fs)
ax[0].set_xticks(np.arange(0,ncoeff,25))
ax[0].set_ylim([0,vals[-1]*0.6])
ax[0].set_xlim([0,110])
ax[0].tick_params(axis='both', which='major', labelsize=tp_fs)

T_arr = np.arange(0.0,6.0,0.01)
ax[1].plot(T_arr, np.pi*np.array(T_arr),lw=2.,color='black')
ax[1].set_xlabel(r'$T$',fontsize=xl_fs)
ax[1].set_ylabel(r'$\alpha_T$',fontsize=yl_fs)
ax[1].set_xticks(np.arange(1,6,1))
ax[1].set_xlim([0.0,6])
ax[1].set_ylim([0.0,6*np.pi])

ax[1].tick_params(axis='both', which='major', labelsize=tp_fs)


#Annotate alpha_T = pi*T at 45 degrees angle in the line plot
ax[1].annotate(r'$\alpha_T = \pi k_B T$', xy=(2.1, 3*np.pi), xytext=(2.1, 3*np.pi), textcoords='data', fontsize=xl_fs-2,rotation=55)

fig.set_size_inches(7,3.5)	
#fig.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),fontsize=le_fs-2,ncol=5)
fig.legend(fontsize=le_fs-2,loc=(0.11,0.91),ncol=5)
#fig.savefig('/home/vgs23/Images/bn_vs_n_1DB.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)
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


