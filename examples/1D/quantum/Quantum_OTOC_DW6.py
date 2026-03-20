import numpy as np
from PISC.dvr.dvr import DVR1D
#from PISC.husimi.Husimi import Husimi_1D
from PISC.potentials import double_well_P6
from PISC.potentials.triple_well_potential import triple_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
#from PISC.engine import OTOC_f_1D_omp_updated
from matplotlib import pyplot as plt
import os 
from PISC.utils.plottools import plot_1D
from PISC.utils.misc import find_OTOC_slope

ngrid = 600

def wf_expt(xarr,wf,dx):
	return np.sum(abs(wf)**2*xarr**2*dx)

#Sextic double well
L = 200.0
lb = -L
ub = L
dx = (ub-lb)/ngrid

N=200 
eps=1.0
g = np.sqrt(18*np.log(N)/N)/np.sqrt(N)
print('g',g)
m = 2/(g)#*np.sqrt(N))

if(0): #Default
    a = -10
    b = -1
    c = 1
    pes = double_well_P6(a,b,c)

T_au = 0.5
beta = 1.0/T_au 
print('T in au, beta',T_au, beta) 
Tkey = 'T_{}'.format(T_au)  

#potkey = 'double_well_P6_a_{}_b_{}_c_{}'.format(a,b,c)

qgrid = np.linspace(lb,ub,ngrid-1)
t = 0#25

kappa_arr = -np.linspace(2,3,11)
domega_arr = np.zeros((len(kappa_arr),200))
for k in range(len(kappa_arr)):
    kappa = kappa_arr[k]
    r = kappa*g
    a = -eps*t
    b = (r+g)/N#np.sqrt(N)
    c = g/(4*N**2)#1.5)
    
    pes = double_well_P6(a,b,c)
    DVR = DVR1D(ngrid,lb,ub,m,pes.potential)
    vals,vecs = DVR.Diagonalize(neig_total=ngrid-10) 

    potgrid = pes.potential(qgrid)/1e9
    plt.plot(qgrid,potgrid)
    qp = np.sqrt(8*N*(abs(r+g))/(3*g))
    plt.scatter(qp,pes.potential(qp)/1e9)
    #plt.xlim([-100,100])
    plt.ylim([-0.0001,0.0002])
    for i in range(10):
            plt.axhline(y=vals[i]/1e9)
    plt.show()

    q2avg = wf_expt(qgrid,vecs[:,0],DVR.dx)
    print('q2avg',(q2avg))

    diff = 0
    for i in range(1,201,2):
        #print('i',i)
        diff+=vals[i]-vals[i-1]
    

    for i in range(200):
        domega_arr[k,i] = vals[i]

#for i in range(1,20,2):
#    plt.scatter(kappa_arr,domega_arr[:,i])
#    plt.plot(kappa_arr,domega_arr[:,i])

#    plt.scatter(kappa_arr,domega_arr[:,i-1])
#    plt.plot(kappa_arr,domega_arr[:,i-1])
#    plt.show()

#plt.plot(kappa_arr,domega_arr[:,19])
#plt.plot(kappa_arr,domega_arr[:,18])
plt.plot(kappa_arr,domega_arr[:,191]-domega_arr[:,190])
plt.show()

if(0): # Plots of PES and WF
    qgrid = np.linspace(lb,ub,ngrid-1)
    potgrid = pes.potential(qgrid)
    hessgrid = pes.ddpotential(qgrid)
    idx = np.where(hessgrid[:-1] * hessgrid[1:] < 0 )[0] +1
    idx=idx[0]
    print('idx', idx, qgrid[idx], hessgrid[idx], hessgrid[idx-1])
    print('E inflection', potgrid[idx]) 

    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim([-20,20])
    plt.plot(qgrid,potgrid)
    plt.plot(qgrid,abs(vecs[:,10])**2)  
    #plt.plot(qgrid,potgrid1,color='k')
    #plt.plot(qgrid,-0.16*qgrid**2 + 0.32)
    for i in range(20):
            plt.axhline(y=vals[i])
    plt.show()



exit()





x_arr = DVR.grid[1:DVR.ngrid]
basis_N = 50
n_eigen = 30

k_arr = np.arange(basis_N) +1
m_arr = np.arange(basis_N) +1

t_arr = np.linspace(0,30.0,2000)
C_arr = np.zeros_like(t_arr) +0j

path = os.path.dirname(os.path.abspath(__file__))

corrkey = 'qq_TCF'#'OTOC'#'qp_TCF'
enskey = 'Kubo'#'mc'#'Kubo'

corrcode = {'OTOC':'xxC','qq_TCF':'qq1','qp_TCF':'qp1'}
enscode = {'Kubo':'kubo','Standard':'stan'} 


if(1): #Thermal correlators
    if(enskey == 'Symmetrized'):
        C_arr = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,corrcode[corrkey],0.5,C_arr)
    else:
        C_arr = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,corrcode[corrkey],enscode[enskey],C_arr) 
    fname = 'Quantum_{}_{}_{}_{}_basis_{}_n_eigen_{}'.format(enskey,corrkey,potkey,Tkey,basis_N,n_eigen)    
    print('fname',fname)    

if(0): #Microcanonical correlators
    n = 2
    C_arr = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,t_arr,corrcode[corrkey],C_arr)   
    fname = 'Quantum_mc_{}_{}_n_{}_basis_{}'.format(corrkey,potkey,n,basis_N)
    print('fname', fname)   

path = os.path.dirname(os.path.abspath(__file__))   
store_1D_plotdata(t_arr,C_arr,fname,'{}/Datafiles'.format(path))

fig,ax = plt.subplots()
plt.plot(t_arr,((C_arr)))
fig.savefig('/home/vgs23/Images/OTOC_temp.pdf'.format(g), dpi=400, bbox_inches='tight',pad_inches=0.0)
    

plt.show()

if(0): #Stray code
    bnm_arr=np.zeros_like(OTOC_arr)
    OTOC_arr[:] = 0.0
    lda = 0.5
    Z = 0.0
    coefftot = 0.0
    for n in range(0,2):
        Z+= np.exp(-beta*vals[n])       
        for M in range(5):
            bnm =OTOC_f_1D_omp_updated.otoc_tools.quadop_matrix_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,M+1,t_arr,1,'cm',OTOC_arr)
            coeff = 0.0
            if(n!=M):
                coeff =(1/beta)*((np.exp(-beta*vals[n]) - np.exp(-beta*vals[M]))/(vals[M]-vals[n]))
            else:
                coeff = np.exp(-beta*(vals[n]))
            coefftot+=coeff
            if(coeff>=1e-4):
                coeffpercent = coeff*100/0.104
                print('coeff',n,M,coeffpercent)
                plt.plot(t_arr,abs(bnm)**2,label='n,M, % ={},{},{}'.format(n,M,np.around(coeffpercent,2)))
                #print('contribution of b_nm for lambda={},n,m={},{}'.format(lda,n,M), np.exp(-beta*vals[n])*np.exp(-lda*beta*(vals[M]-vals[n])))
            bnm_arr+=coeff*abs(bnm)**2
    bnm_arr/=Z
    print('Z',Z,coefftot)
    plt.plot(t_arr,np.log(bnm_arr),color='m',linewidth=3)
    plt.legend()
    #Clamda = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
    
    #cmc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,3,m_arr,t_arr,'xxC',OTOC_arr)  
    #for i in range(15):
    #   OTOC= (OTOC_f_1D_omp.position_matrix.compute_b_mat_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n,i,t_arr,OTOC_arr))
    #   OTOC_arr += abs(OTOC)**2

    #plt.plot(t_arr,bnm_arr,color='k')
    #plt.plot(t_arr,qqkubo) 
    #plt.plot(t_arr,np.log(abs(Clamda)))
    #plt.plot(t_arr,np.log(abs(cmc)))
    #plt.plot(t_arr,np.log(abs(Cstan)), label='Standard OTOC')  
    #plt.plot(t_arr,np.log(abs(Csym)), label='Symmetrized OTOC')
    #plt.plot(t_arr,np.log(abs(Ckubo)),color='k',label=r'Kubo thermal OTOC, $\lambda_K={:.2f}$'.format(np.real(slope1)))
    #plt.plot(t_arr,np.log(OTOC_arr), linewidth=2,label='Quantum OTOC')
    #plt.plot(t_trunc1,slope1*t_trunc1+ic1,linewidth=4,color='k')
    #plt.plot(t_arr,np.log(abs(Cstan)),color='r',label=r'Standard thermal OTOC, $\lambda_S={:.2f} > 2\pi/\beta$'.format(np.real(slope2)))
    #plt.plot(t_trunc2,slope2*t_trunc2+ic2,linewidth=4,color='r')
    #plt.plot(t_arr,np.imag(Cstan), label='Standard xx TCF')    
    #plt.plot(t_arr,np.imag(Csym), label='Symmetrized xx TCF')
    #plt.plot(t_arr,np.real(Ckubo),color='k',label='Kubo xx TCF')

    #plt.suptitle('Double well potential')
    #plt.title(r'$Kubo \; vs \; Standard \; OTOC \; at \; T=T_c$')  
    #plt.title(r'$xp \; TCF \; behaviour \; at \; T=T_c$')
    #plt.legend()
    #plt.show()

    #F = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxF','stan',OTOC_arr) 
    #qq = OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qq1','kubo',OTOC_arr) 
    #Cstan= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr) 
    #OTOC_arr*=0.0
    #Ckubo= OTOC_f_1D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'qq1','kubo',OTOC_arr)
    #Csym = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
    #Cstan = OTOC_f_1D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.0,OTOC_arr)
    #fname = 'Quantum_{}_{}_{}_{}_basis_{}_n_eigen_{}'.format(enskey,corrkey,potkey,Tkey,basis_N,n_eigen)
    #store_1D_plotdata(t_arr,Ckubo,fname,'{}/Datafiles'.format(path))
    
    n = 2
    print('E', np.around(vals[n],2),vals[n])
    #c_mc = OTOC_f_1D_omp_updated.otoc_tools.corr_mc_arr_t(vecs,m,x_arr,DVR.dx,DVR.dx,k_arr,vals,n+1,m_arr,t_arr,'xxC',OTOC_arr)    
    #fname = 'Quantum_{}_{}_{}_basis_{}_n_eigen_{}_n_{}'.format(enskey,corrkey,potkey,basis_N,n_eigen, n )#np.around(vals[n],2))
    #store_1D_plotdata(t_arr,c_mc,fname,'{}/Datafiles'.format(path))

    #plt.plot(t_arr,qq,lw=2)
    #plt.plot(t_arr,np.log(abs(Csym)),linewidth=2)  
    #plt.show() 



    
