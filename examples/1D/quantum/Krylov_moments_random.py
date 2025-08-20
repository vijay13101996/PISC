import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
import argparse
import matplotlib as mpl

neigs = 800
L = 40
omega = 1.0

neigs_arr = np.arange(1,neigs+1)

harm_vals = omega*neigs_arr - omega/2
box_vals = np.pi**2*neigs_arr**2/(2*L**2)
pow_vals = 0.05*neigs_arr**1.5

vals = harm_vals
print('vals',vals[:5])

O = np.zeros((neigs,neigs)) + 1j*0.0

k_diag = 800
for i in range(neigs):
    for j in range(i,neigs):
        #if(abs(i-j)%2==1): 
            if(abs(i-j)<=k_diag):
                O[i,j] =  1.0#np.random.uniform(0,10)
                #O[i,j] = 1.0 #+ 0.2*np.random.normal(0,1)
                O[j,i] = O[i,j]

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

#O = O_anal

liou_mat = np.zeros((neigs,neigs))
L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

def compute_moments(O, vals, T_au, lamda, ip, nmoments, label=None):
    On = np.zeros((neigs,neigs))

    print('T, lamda, ip',T_au,lamda,ip)
    Tkey = 'T_{}'.format(T_au)

    beta = 1.0/T_au 

    #Normalize the operator
    b0 = 0.0
    b0 = Krylov_complexity.krylov_complexity.compute_ip(O,O,beta,vals,lamda,b0,ip)
    O/=np.sqrt(b0)

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, ip, lamda, moments)
    even_moments = moments[0::2]

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, lamda, ip)
    
    return moments, barr

def mom_to_bn(even_moments):
    ncoeff = len(even_moments) 
    bnarr = np.zeros(ncoeff)
    Marr = np.zeros((ncoeff,ncoeff)) # index l is the row, index j is the column
    Marr[:,0] = even_moments # The M matrix is to be filled from left to the diagonal, first column is the moments

    bnarr[0] = 1.0 # We assume that the operator is normalized

    for l in range(1,ncoeff): # b0 and M[0,0] are already set, and the first row is filled until the diagonal.
        for j in range(1,l+1): # First column is already filled, so we start from j=1
            if (j==1): # Fill the first column
                Marr[l,j] = Marr[l,j-1]/bnarr[j-1]**2 # M[:,-1] is set to zero by default
            else:
                Marr[l,j] = Marr[l,j-1]/bnarr[j-1]**2 - Marr[l-1,j-2]/bnarr[j-2]**2
        
        bnarr[l] = np.sqrt(Marr[l,l]) # The diagonal element is the next b

    return bnarr

nmoments = 51
ncoeff = nmoments//2+1
moments = np.zeros(nmoments+1)

T_arr = [2,3,4,5,6]#[25,30,35,40]#np.arange(1.0,5.01,1.0)#[1.0,2.0,3.0,4.0,5.0]

lamda = 0.5
T = 40.0
#for T in T_arr:

lamda_arr = [0.5]#np.arange(0.2,0.51,0.1)
slope_arr = []

#for lamda in lamda_arr:

def compute_slope(lamda):   
    slope_arr = []
    for T in T_arr:
        mun_arr, bnarr = compute_moments(O, vals, T, lamda, 'asm', nmoments)

        print('nmoments',nmoments)

        even_moments = mun_arr[0::2][1:] # Skip the first moment, which is the normalization
        narr = np.arange(1,len(even_moments)+1) 

        log_em = np.log(even_moments)
        log_n = np.log(narr)
        nlog_n = narr*log_n

        #Fitting log_em = 2*nlog_n + diff(n; T,lamda)
        #NOTE: diff measures the growth of the moments with n as a function of temperature and lamda
        diff = log_em - 2*nlog_n

        #Fit diff(n; T,lamda) to a line vs n so that diff(n; T,lamda) = s(T,lamda)*n + i(T,lamda)
        #NOTE: The intercept is temperature independent so that i(T,lamda) = i(lamda)
        s = 20
        e = 40 #nmoments//2
        fit = np.polyfit(narr[s:e],diff[s:e],1)
        print('fit',fit, fit[0])

        slope_arr.append(fit[0])

        #plt.plot(narr[s:],fit[0]*narr[s:]+fit[1],label='$\lambda$ = {}'.format(lamda))
        #plt.plot(narr[s:e],fit[0]*narr[s:e]+fit[1],label='T = {}'.format(T))
        #plt.scatter(narr[s:e],diff[s:e])#,label='even_moments')
        #plt.plot(narr,log_em,label='T = {}'.format(T))
        plt.scatter(narr,bnarr[1:],label='bnarr')

    return slope_arr

if(1):
    for T in [1,5,10]:#,20,30]:
        mun_arr, bnarr = compute_moments(O, vals, T, lamda, 'asm', nmoments)

        bn_mom = mom_to_bn(mun_arr[0::2])
        #plt.scatter(np.arange(0,ncoeff),bnarr,label='bnarr')
        #plt.scatter(np.arange(0,ncoeff),bn_mom,label='bn_mom')
        
        plt.plot(np.log(mun_arr[0::2]),label='mun_arr,T = {}'.format(T))
    
    #plt.ylim([0,vals[-1]*0.6])
    plt.legend()
    plt.show()

    exit()

intercept_arr = []

for lamda in lamda_arr:
    slope_arr = compute_slope(lamda)

    log_T = np.log(T_arr)
    slope_arr = np.array(slope_arr)

    #Fit s(T,lamda) to a line vs log(T) so that s(T,lamda) = sT(lamda)*log(T) + iT(lamda)
    #NOTE: The slope sT is lamda independent so that sT(lamda) = sT = 2 (numerically)
    fit = np.polyfit(log_T,slope_arr,1)
    print('\n fit slope',fit,'\n')

    #plt.plot(np.log(lamda_arr),np.log(slope_arr))
    #plt.scatter(log_T,slope_arr)
    #plt.plot(log_T,fit[0]*log_T+fit[1])

    intercept_arr.append(fit[1])

plt.legend()
plt.show()

plt.plot(np.log(0.5/lamda_arr),intercept_arr)
plt.plot(np.log(0.5/lamda_arr),2*np.log(0.5/lamda_arr)+intercept_arr[0]-2*np.log(0.5/lamda_arr)[0])

plt.show()
exit()


