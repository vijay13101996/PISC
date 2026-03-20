import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials.Coupled_quartic import coupled_quartic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from PISC.engine import OTOC_f_2D_omp_updated
import os 
import time 
import argparse
import ast
from matplotlib import pyplot as plt

g1 = 0.25
path = os.path.dirname(os.path.abspath(__file__))

n_eigen = 40
basis_N = 100

ndeg = False #True

def load_data(path,g1,g2):
    potkey = 'coupled_quartic_g1_{}_g2_{}'.format(g1,g2)

    with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey)) as f:    
        for line in f:
            pass
        param_dict = ast.literal_eval(line)

    lbx = param_dict['lbx']
    ubx = param_dict['ubx']
    lby = param_dict['lby']
    uby = param_dict['uby']
    ngridx = param_dict['ngridx']
    ngridy = param_dict['ngridy']
    m = param_dict['m']
    pes = coupled_quartic(g1,g2)

    DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)   
    fname_diag = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngridx)    # Change ngridx!=ngridy

    if(ndeg):
        vals = read_arr('{}_vals_ndeg'.format(fname_diag),'{}/Datafiles'.format(path))
        vecs = read_arr('{}_vecs_ndeg'.format(fname_diag),'{}/Datafiles'.format(path))
    else:
        vals = read_arr('{}_vals'.format(fname_diag),'{}/Datafiles'.format(path))
        vecs = read_arr('{}_vecs'.format(fname_diag),'{}/Datafiles'.format(path))
    
    return vals,vecs,DVR

def main(g2,T):
    start = time.time()
    print('g2',g2)
    print('T',T)

    potkey = 'coupled_quartic_g1_{}_g2_{}'.format(g1,g2)
    pes = coupled_quartic(g1,g2)

    print('pot',potkey)
    
    beta = 1.0/T

    vals,vecs,DVR = load_data(path,g1,g2)
    x_arr = DVR.pos_mat(0)
    k_arr = np.arange(basis_N) +1
    m_arr = np.arange(basis_N) +1

    reg='Kubo'
    t_arr = np.linspace(0.0,5.0,1000)
    OTOC_arr = np.zeros_like(t_arr)+0j 

    if(1):
        if(reg=='Kubo'):
            CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','kubo',OTOC_arr)
        elif(reg=='Standard'):  
            CT = OTOC_f_2D_omp_updated.otoc_tools.therm_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC','stan',OTOC_arr)
        elif(reg=='Symmetric'):
            CT = OTOC_f_2D_omp_updated.otoc_tools.lambda_corr_arr_t(vecs,DVR.m,x_arr,DVR.dx,DVR.dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,'xxC',0.5,OTOC_arr)
       
        if(ndeg):
            fname = 'Quantum_{}_OTOC_{}_T_{}_neigen_{}_basis_{}_ndeg'.format(reg,potkey,T,n_eigen,basis_N)
        else:
            fname = 'Quantum_{}_OTOC_{}_T_{}_neigen_{}_basis_{}'.format(reg,potkey,T,n_eigen,basis_N)
        store_1D_plotdata(t_arr,CT,fname,'{}/Datafiles'.format(path))
   

    if(ndeg):
        dataarr = read_1D_plotdata('{}/Datafiles/Quantum_{}_OTOC_{}_T_{}_neigen_{}_basis_{}_ndeg.txt'.format(path,reg,potkey,T,n_eigen,basis_N))
        t_arr = dataarr[:,0]
        CT = dataarr[:,1]
    else:
        dataarr = read_1D_plotdata('{}/Datafiles/Quantum_{}_OTOC_{}_T_{}_neigen_{}_basis_{}.txt'.format(path,reg,potkey,T,n_eigen,basis_N))
        t_arr = dataarr[:,0]
        CT = dataarr[:,1]

    plt.plot(t_arr,CT)
    plt.xlabel('t')
    plt.ylabel('OTOC')
    plt.show()

    print('Time taken:',time.time()-start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute quantum OTOC for Coupled Quartic potential')
    parser.add_argument('--g2', type=float, help='Coupling constant g2',default=4.5)
    parser.add_argument('--T', type=float, help='Temperature',default=1.0)

    args = parser.parse_args()
    main(args.g2,args.T)

