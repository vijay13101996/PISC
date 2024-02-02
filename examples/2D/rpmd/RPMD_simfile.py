import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials import quartic_bistable, Harmonic_oblique, DW_harm
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 
from argparse import ArgumentParser

path = os.path.dirname(os.path.abspath(__file__))
#path = '/scratch/vgs23/PISC/examples/2D/rpmd'

dim=2
m = 0.5

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

def main(times,nbeads,z,pot):
    if(pot=='dw_qb'): # 2D Double well
        alpha = 0.382
        D = 3*Vb
     
        pes = quartic_bistable(alpha,D,lamda,g,z)
        potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        
    if(pot=='dw_harm'): # 2D Double well with harmonic coupling
        w = 2.0
        
        pes = DW_harm(m, w, lamda, g, z)
        potkey = 'DW_harm_2D_m_{}_w_{}_lamda_{}_g_{}_z_{}'.format(m,w,lamda,g,z)

    Tc = 0.5*lamda/np.pi
    T = times*Tc
    Tkey = 'T_{}Tc'.format(times)

    N = 1000
    dt_therm = 0.05
    dt = 0.005
    time_therm = 50.0
    time_total = 5.0

    method = 'RPMD'
    sysname = 'Papageno'      
    corrkey = 'fd_OTOC'#'OTOC'
    enskey = 'thermal'

    print('z,times,nbeads',z,times,nbeads)

    ### -----------------------------------------------------------------------------
    E = T
    xg = np.linspace(-3,3,int(1e2)+1)
    yg = np.linspace(-3,3,int(1e2)+1)
    xgrid,ygrid = np.meshgrid(xg,yg)
    potgrid = pes.potential_xy(xgrid,ygrid)

    ind = np.where(potgrid<E)
    xind,yind = ind

    #fig,ax = plt.subplots(1)
    #ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0,1.01*D,D/30))

    qlist= []
    for x,y in zip(xind,yind):
        #x = i[0]
        #y = i[1]
        #ax.scatter( xgrid[x,y],ygrid[x,y])#xgrid[x][y] , ygrid[x][y] )
        qlist.append([xgrid[x,y],ygrid[x,y]])
    #plt.show()
    qlist = np.array(qlist)
        
    ### ------------------------------------------------------------------------------

    pes_fort=False#True
    propa_fort=True
    transf_fort=False#True

    Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey)
    Sim_class.set_sysparams(pes,T,m,dim)
    Sim_class.set_simparams(N,dt_therm,dt,pes_fort=pes_fort,propa_fort=propa_fort,transf_fort=transf_fort)
    Sim_class.set_methodparams(nbeads=nbeads)
    Sim_class.set_ensparams(tau0=1.0,pile_lambda=100.0,qlist=qlist)
    Sim_class.set_runtime(time_therm,time_total)

    if(propa_fort):
        print('Integrators are implemented in FORTRAN')
    if(transf_fort):
        print('Transformations are implemented in FORTRAN')
    if(pes_fort):
        print('PES is implemented in FORTRAN')

    start_time=time.time()
    func = partial(Sim_class.run_seed)
    seeds = range(2000)
    seed_split = chunks(seeds,20)

    param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
        'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
            
    with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:   
        f.write('\n'+str(param_dict))

    batching(func,seed_split,max_time=1e6)
    print('time', time.time()-start_time)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--times',type=float,default=3.0)
    parser.add_argument('--nbeads',type=int,default=1)
    parser.add_argument('--z',type=float,default=1.0)
    parser.add_argument('--pot',type=str,default='dw_qb')
    args = parser.parse_args()
    main(args.times,args.nbeads,args.z,args.pot)

#---------------------------------------------------------------------------------


if(0): ### Oblique harmonic oscillator
    m = 0.5
    omega1 = 1.0
    omega2 = 1.0
    trans = np.array([[1.0,0.7],[0.2,1.0]])
    
    pes = Harmonic_oblique(trans,m,omega1,omega2)   

    T = 1.0
    
    N = 1000
    dt_therm = 0.05
    dt = 0.02
    time_therm = 50.0
    time_total = 20.0
    nbeads = 1

    potkey = 'TESTharmonicObl'
    Tkey = 'T_{}'.format(T)

if(0):
    # Tanimura's system-bath potential
    m = 1.0
    mb= 1.0
    delta_anh = 0.1
    w_10 = 1.0
    wb = w_10
    wc = w_10 + delta_anh
    alpha = (m*delta_anh)**0.5
    D = m*wc**2/(2*alpha**2)

    VLL = -0.75*wb#0.05*wb
    VSL = 0.75*wb#0.05*wb
    cb = 0.75#*wb#0.65*wb#0.75*wb

    pes = Tanimura_SB(D,alpha,m,mb,wb,VLL,VSL,cb)
                
    TinK = 300
    K2au = 3.1667e-6
    T = 0.125#TinK*K2au
    beta = 1/T

    potkey = 'Tanimura_SB_D_{}_alpha_{}_VLL_{}_VSL_{}_cb_{}'.format(D,alpha,VLL,VSL,cb)
    Tkey = 'T_{}'.format(T)
    print('pot', potkey)

    N = 1000
    dt_therm = 0.05#10.0
    dt = 0.01#2.0
    time_therm = 50.0#60000.0
    time_total = 100.0#20000.0

    nbeads=16


