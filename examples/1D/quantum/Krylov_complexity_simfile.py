import numpy as np
import matplotlib.pyplot as plt
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from Krylov_complexity_harm_pow import compute_bn, comput_On_matrix, compute_moments
import matplotlib
from math import comb
import mpmath
#---------------------------------------------- DVR and plot parameters

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False


xl_fs = 16 
yl_fs = 16
tp_fs = 14

le_fs = 16
ti_fs = 12

L = 40
lb = -L
ub = L
ngrid = 1000

#---------------------------------------------- Potential parameters

m = 1.0
omega = 1.0
a = 0.0
b = 0.0

pes = mildly_anharmonic(m,a,b,w=omega)

#potkey = 'L_{}_MAH_w_{}_a_{}_b_{}_n_{}'.format(L,omega,a,b,n_anharm)

DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
neigs = 500
vals,vecs = DVR.Diagonalize(neig_total=neigs)

#---------------------------------------------- Compute position matrix 

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

#pos_mat = np.zeros((neigs,neigs)) 
#pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)
#O = (pos_mat_anal)

def comp_pos_mat(i,j):
    if(i==j):
        return 0.0
    elif(i-j==1):
        return np.sqrt(1/(2*m*omega))*np.sqrt(j+1)
    elif(j-i==1):
        return np.sqrt(1/(2*m*omega))*np.sqrt(i+1)
    else:
        return 0.0

pos_mat_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        pos_mat_anal[i,j] = comp_pos_mat(i,j)

vals_anal = np.zeros(neigs)
for i in range(neigs):
    vals_anal[i] = omega*(i+0.5)

#---------------------------------------------- Operator construction

#Compute powers of an operator
def pow_n(O,n):
    On = O
    for i in range(1,n):
        On = np.matmul(On,O)
    return On

#Construct a matrix which is 1 along every odd off-diagonal element
def Un(neigs,d_max):
    Un = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(i,neigs):
            if(abs(i-j)%2==1 and abs(i-j)<d_max):
                Un[i,j] = 1
                Un[j,i] = Un[i,j]
    return Un

def un(neigs):
    un = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(i,neigs):
                un[i,j] = 1
                un[j,i] = un[i,j]
    return un
#----------------------------------------------
O = pos_mat_anal
vals = vals_anal

ncoeff = 100
T_au = 1.0

#fig,ax = plt.subplots(1,1)#,figsize=(4,4))

N = 200
U2 = Un(neigs,2)
U4 = Un(neigs,6)
print('U3',U4[:10,:10])
k=5
O = Un(neigs, N)#pow_n(U2,k)#Un(neigs,N)#pow_n(O,N)

def T(i,j,k):
    tmp = k - abs(i-j)
    ret = 0.0
    if(tmp<0 or tmp%2==1):
        return ret
    else:
        ind = int(tmp/2)
        ret = comb(k,ind)
 
    if(i+j<k):
        tmp2 = (k - (i+j))//2
        ret-= comb(k,tmp2-1)
    return ret

q = 2
for i in range(10):
    print('T',T(2,i,k))

Omat = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        if(k-abs(i-j)>=0):
            Omat[i,j] = T(i,j,k)
print('Omat \n',Omat[:10,:10])

print('O \n',O[:10,:10])

print('diff',(O-Omat)[:10,:10])

u = un(neigs)
#print('U2',U2)

print('u',u)

print('O \n',O[:10,:10])
narr, mun_arr = compute_moments(O,vals,T_au,80)
beta = 1/T_au
xi = beta*omega/2
pref = 2*np.exp(-2*xi)*(1+1/np.tanh(xi))*np.sinh(xi)

def A(n,k):
    val=0.0
    for i in range(0,k+1):
        val+= (-1)**i*comb(n+1,i)*(k+1-i)**n
    return val

def E_n(n):
    arr = np.zeros(n)
    for i in range(n):
        arr[i] = A(n,i)
    return arr

n = 3 

print('mun_arr',mun_arr[n])

z = np.exp(-xi)

if(1): #mu for O =u
    num = 4*omega**(2*n)*np.sinh(xi)*z**2
    denom = (z-1)**(2*n+2)*(z+1)
    pcoeff = E_n(2*n)
    print('pcoeff',pcoeff)
    poly = np.poly1d(pcoeff)

    mu_anal = poly(z)*num/denom
    print('mu_anal',mu_anal)

if(1): #mu for O = UN
    term1 = 2*z*(2*omega)**(2*n)
    term2 = mpmath.lerchphi(z**2,-2*n,0.5)
    term3 = 0.0#z**(N)*mpmath.lerchphi(z**2, -2*n, N/2+0.5)
    mu_anal = term1*(term2-term3)
    print('mu_anal',mu_anal)

coeffarr0,bnarr0 = compute_bn(O,vals,vecs,T_au,ncoeff)
print('bnarr',bnarr0[:5]**2)

for T_au in [1.0]:
    coeffarr,bnarr = compute_bn(O,vals,vecs,T_au,ncoeff)
    #ax.scatter(coeffarr[1:],bnarr[1:],label='T={}'.format(T_au),s=4)


if(0):
    b0 = []
    for i in range(1,50,5):
        On = pow_n(O,i)
        coeffarr,bnarr = compute_bn(On,vals,vecs,T_au,ncoeff)
        print('bnarr',bnarr[:i+1])
        ax.scatter(coeffarr[1:],bnarr[1:],label='n={}'.format(i),s=10)
        b0.append(bnarr[1])

if(0):
    for d in [20,40]:
        O = Un(neigs,d)
        coeffarr,bnarr = compute_bn(O,vals,vecs,T_au,ncoeff)
        #ax.scatter(coeffarr[1:],bnarr[1:],label='d={}'.format(d),s=4)
        
        ax.scatter(coeffarr[1:],np.log(bnarr[1:]),label='d={}'.format(d),s=4)

nmat = 2

barr, On = comput_On_matrix(O,T_au,vals,nmat)


#ax.scatter(np.arange(nmat+1),barr,label='On',s=4)

ax.set_xlabel(r'$n$',fontsize=xl_fs)
ax.set_ylabel(r'$b_{n}$',fontsize=yl_fs)
ax.set_ylim([0,vals[neigs-1]/2])

#ax.axhline(vals[N-1]/2,color='black',ls='--')

#ax.plot(coeffarr,np.pi*coeffarr*T_au,color='r',label=r'$\pi n T$')

plt.show()

#coeffarr,bnarr = compute_bn(O,vals,vecs,T_au,ncoeff)





