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

xl_fs = 13
yl_fs = 13
tp_fs = 12

le_fs = 11#9.5
ti_fs = 13


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

fig, ax = plt.subplots()

def plot_ax_2c(ax):

    T_arr = np.array([1,2,3,4,5])
    axin = ax.inset_axes([0.6, 0.19, 0.38, 0.38])

    for i in range(len(T_arr)):
        ax.scatter(np.arange(ncoeff),bnarr[i,:],s=10,label=r'$T={}$'.format(T_arr[i]))

        #### Save data to txt files
        np.savetxt('FIG2c_bn_T_{}_{}.txt'.format(T_arr[i],potkey), np.column_stack((np.arange(ncoeff), bnarr[i,:])), header='n b_n')

        #Find slope
        p = np.polyfit(np.arange(ncoeff)[4:17],bnarr[i,4:17],1)
        print('p',p)

        axin.scatter(T_arr[i],p[0],zorder=4,s=10)

    #Draw horizontal line at y=Emax/2
    ax.hlines(0.5*vals[-1],0,ncoeff,linestyles='dashed',lw=2,color='0.3')

    ax.set_xlabel(r'$n$',fontsize=xl_fs)
    #ax.set_ylabel(r'$b_n$',fontsize=yl_fs)
    #ax[0].legend(fontsize=le_fs)
    ax.set_xticks(np.arange(0,ncoeff,25))
    ax.set_ylim([0,vals[-1]*0.6])
    ax.set_xlim([0,110])
    #ax.tick_params(axis='both', which='major', labelsize=tp_fs)

    T_arr = np.arange(0.0,6.0,0.01)
    axin.plot(T_arr, np.pi*np.array(T_arr),lw=2.,color='black')
    axin.set_xlabel(r'$T$',fontsize=xl_fs-5)
    axin.set_ylabel(r'$\alpha_T$',fontsize=yl_fs-5)
    axin.set_xticks(np.arange(1,6,1))
    axin.set_xlim([0.0,6])
    axin.set_ylim([0.0,6*np.pi])

    axin.tick_params(axis='both', which='major', labelsize=tp_fs-4)

    #Annotate alpha_T = pi*T at 45 degrees angle in the line plot
    axin.annotate(r'$\alpha_T = \pi k_B T$', xy=(1.3, 2.*np.pi), xytext=(0.8, 2.*np.pi), textcoords='data', fontsize=xl_fs-5,rotation=50)

#plot_ax_2c(ax)

#fig.set_size_inches(3.5,3.5)	
#fig.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),fontsize=le_fs-2,ncol=5)
#fig.legend(fontsize=le_fs-2,loc=(0.11,0.91),ncol=5)
#fig.savefig('/home/vgs23/Images/bn_vs_n_1DB_1panel.pdf', dpi=400, bbox_inches='tight',pad_inches=0.0)
#plt.show()


