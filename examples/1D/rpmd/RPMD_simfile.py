import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials import double_well,quartic,morse, asym_double_well, harmonic1D
from PISC.engine.PI_sim_core import SimUniverse
import time
import os 
import argparse

dim=1

if(0): #Asymmetric double well potential
    lamda = 2.0
    g = 0.08
    k = 0.04
    pes = asym_double_well(lamda,g,k)
    
    Tc = lamda*(0.5/np.pi)    
    times = 1.0#0.8
    T = times*Tc

    m = 0.5
    N = 1000
    dt_therm = 0.05
    dt = 0.002
    time_therm = 50.0
    time_total = 5.0

    nbeads = 1

    potkey = 'asym_double_well_lambda_{}_g_{}_k_{}'.format(lamda,g,k)
    Tkey = 'T_{}Tc'.format(times)   

    
if(0): #Quartic potential from Mano's paper
    a = 1.0

    pes = quartic(a)

    T = 1.0
    
    m = 1.0
    N = 1000
    dt_therm = 0.05
    dt = 0.02

    time_therm = 50.0
    time_total = 20.0
    
    nbeads = 1

    method = 'RPMD'
    potkey = 'TESTquart'.format(a)
    sysname = 'Selene'      
    Tkey = 'T_{}'.format(T)
    corrkey = 'qq_TCF'
    enskey = 'thermal'

if(0): #Morse
    m=0.5
    D = 9.375
    alpha = 0.382
    pes = morse(D,alpha)
    
    w_m = (2*D*alpha**2/m)**0.5
    Vb = D/3

    print('alpha, w_m', alpha, Vb/w_m)
    T = 3.18
    potkey = 'morse'
    Tkey = 'T_{}'.format(T)
    
    N = 1000
    dt_therm = 0.05
    dt = 0.02
    time_therm = 50.0
    time_total = 5.0

    nbeads = 1

def main(times=0.95,nbeads=16): #Double well potential
    lamda = 2.0
    g = 0.08

    print('Vb', lamda**4/(64*g))

    pes = double_well(lamda,g)

    Tc = 0.5*lamda/np.pi
    T = times*Tc

    m = 0.5
    N = 1000
    dt_therm = 0.05
    dt = 0.005
    time_therm = 50.0
    time_total = 5.0

    potkey = 'inv_harmonic_lambda_{}_g_{}'.format(lamda,g)
    Tkey = 'T_{}Tc'.format(times)

    method = 'RPMD'
    sysname = 'Papageno'        
    corrkey = 'fd_OTOC'
    enskey = 'thermal'

    path = os.path.dirname(os.path.abspath(__file__))

    pes_fort = False
    propa_fort = True
    transf_fort = True

    Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey,ext_kwlist=['FORTRAN'])
    Sim_class.set_sysparams(pes,T,m,dim)
    Sim_class.set_simparams(N,dt_therm,dt,pes_fort=pes_fort,propa_fort=propa_fort,transf_fort=transf_fort)
    Sim_class.set_methodparams(nbeads=nbeads)
    Sim_class.set_ensparams(tau0=1.0,pile_lambda=1.0)
    Sim_class.set_runtime(time_therm,time_total)

    start_time=time.time()
    func = partial(Sim_class.run_seed)
    seeds = range(1)
    seed_split = chunks(seeds,20)

    param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
        'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
            
    with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:   
        f.write('\n'+str(param_dict))

    print('T',times, 'Tc', Tc)
    print('nbeads',nbeads)
    batching(func,seed_split,max_time=1e6)
    print('time', time.time()-start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--times',type=float,help='Temperature in units of Tc',default=0.95)
    parser.add_argument('-n','--nbeads',type=int,help='Number of beads',default=16)
    args = parser.parse_args()
    main(times=args.times,nbeads=args.nbeads)
