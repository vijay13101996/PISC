import numpy as np
from matplotlib import pyplot as plt
import time
import pickle
import multiprocessing as mp
from functools import partial
from PISC.utils.mptools import chunks, batching
from PISC.potentials import quartic_bistable, DW_Morse_harm
from PISC.engine.PI_sim_core import SimUniverse
import time
import os
from argparse import ArgumentParser

dim=2
m = 0.5

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

alpha = 0.382
D = 3*Vb

def main(times,nbeads,z,pot,enskey):
    print('\n')
    print('T:',times,'Tc')
    print('nbeads:',nbeads)
    print('z:',z)
    print('potential:',pot)
    print('ensemble:',enskey)

    Tc = 0.5*lamda/np.pi
    T = times*Tc
    Tkey = 'T_{}Tc'.format(times)

    if(pot=='dw_qb'):# 2D Double well
        pes = quartic_bistable(alpha,D,lamda,g,z)
        potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        print(potkey)

    if(pot=='dw_harm'):# 2D Double well with harmonic coupling 
        pes = DW_Morse_harm(alpha,D,lamda,g,z)
        potkey = 'DW_Morse_harm_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        print(potkey)

    N = 1000
    dt_therm = 0.05
    dt = 0.002
    time_therm = 50.0
    time_total = 5.0

    method = 'RPMD'
    sysname = 'Papageno'		
    corrkey = 'fd_OTOC'

    pes_fort=False#True
    propa_fort=True
    transf_fort=True

    path = '/scratch/vgs23/PISC/examples/2D/rpmd'
    #path = os.path.dirname(os.path.abspath(__file__))

    Sim_class = SimUniverse(method,path,sysname,potkey,corrkey,enskey,Tkey,ext_kwlist=['1D_const'])
    Sim_class.set_sysparams(pes,T,m,dim)
    Sim_class.set_simparams(N,dt_therm,dt,pes_fort=pes_fort,propa_fort=propa_fort,transf_fort=transf_fort)
    Sim_class.set_methodparams(nbeads=nbeads)
    Sim_class.set_ensparams(tau0=1.0,pile_lambda=100.0)
    Sim_class.set_runtime(time_therm,time_total)

    start_time=time.time()
    func = partial(Sim_class.run_seed)
    seeds = range(2000)#,4000)
    seed_split = chunks(seeds,20)

    param_dict = {'Temperature':Tkey,'CType':corrkey,'Ensemble':enskey,'m':m,\
        'therm_time':time_therm,'time_total':time_total,'nbeads':nbeads,'dt':dt,'dt_therm':dt_therm}
            
    with open('{}/Datafiles/RPMD_input_log_{}.txt'.format(path,potkey),'a') as f:	
        f.write('\n'+str(param_dict))

    batching(func,seed_split,max_time=1e6)
    print('time', time.time()-start_time)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--times', type=float, default=3.0, help='Temperature in units of Tc')
    parser.add_argument('-nb', '--nbeads', type=int, default=1, help='Number of beads')
    parser.add_argument('-z', '--z', type=float, default=1.0, help='Coupling strength')
    parser.add_argument('-p', '--pot', type=str, default='dw_qb', help='Potential')
    parser.add_argument('-e', '--ens', type=str, default='thermal', help='Ensemble')
    args = parser.parse_args()

    main(args.times,args.nbeads,args.z,args.pot,args.ens)
