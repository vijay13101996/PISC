import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials import coupled_quartic
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 
import argparse

def main(beta=1.0,nbeads=4,T=None):
    dim=2

    m=1.0

    g1 = 10
    g2 = 0.1
    pes = coupled_quartic(g1,g2)

    potkey = 'CQ_g1_{}_g2_{}'.format(g1,g2)

    if(T is None):
        T = 1.0/beta
    else:
        beta = 1.0/T
    print('beta,nbeads',beta,nbeads)

    Tkey = 'beta_{}'.format(beta)

    N = 1000
    dt_therm = 0.05
    dt = 0.002
    time_therm = 50.0
    time_total = 5.0

    path = os.path.dirname(os.path.abspath(__file__))

    method = 'RPMD'
    sysname = 'Papageno'      
    corrkey = 'Im_qq_TCF'#'fd_OTOC'#'qq_TCF'#'pq_TCF'#'OTOC'
    enskey = 'thermal'
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

    pes_fort=True
    propa_fort=True
    transf_fort=True

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
    seeds = range(100)
    seed_split = chunks(seeds,20)

    param_dict = {'Temperature':Tkey,'CType':corrkey,'m':m,\
        'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
            
    with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:   
        f.write('\n'+str(param_dict))

    batching(func,seed_split,max_time=1e6)
    print('time', time.time()-start_time)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-b', '--beta', type=float, default=1.0)
    argparser.add_argument('-n', '--nbeads', type=int, default=4)
    argparser.add_argument('-T', type=float, default=None)
    main(**vars(argparser.parse_args()))
