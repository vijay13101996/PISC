import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials import quartic_bistable, DW_Morse_harm
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
import time 
import argparse

ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

n_eig_tot = 200 #Number of eigenstates to be calculated

#2D double well potential parameters
m = 0.5

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)

D = 3*Vb
alpha = 0.382

def main(z,pot):
    start_time = time.time()
    print('z',z)
    print('pot',pot)
    if(pot=='dw_qb'):
        potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        pes = quartic_bistable(alpha,D,lamda,g,z)
        
        lbx = -6.0
        ubx = 6.0
        lby = -3
        uby = 7.0

    elif(pot=='dw_harm'):
        potkey = 'DW_Morse_harm_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)
        pes = DW_Morse_harm(alpha,D,lamda,g,z)

        lbx = -6.0
        ubx = 6.0
        lby = -2.0
        uby = 7.0

    if(0):
        xg = np.linspace(lbx,ubx,ngridx)
        yg = np.linspace(lby,uby,ngridy)
        xgr,ygr = np.meshgrid(xg,yg)
        plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,3*Vb,1.0))
        plt.show()    
        exit()

    x = np.linspace(lbx,ubx,ngridx+1)
    print('Vs',pes.potential_xy(0,0))
    fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)	
    path = os.path.dirname(os.path.abspath(__file__))

    DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
    print('potential',potkey)	

    #Diagonalization
    param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
    with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:	
        f.write('\n'+str(param_dict))

    vals,vecs = DVR.Diagonalize()

    store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
    print('Time taken:',time.time()-start_time)

    if(0): #Test whether the Wavefunctions look correct
        vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
        vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        n=8
        print('vals[n]', vals[n])
        plt.imshow(DVR.eigenstate(vecs[:,n])**2,origin='lower')

    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D double well potential')
    parser.add_argument('--z','-z', type=float, help='z parameter', default=0.0)
    parser.add_argument('--pot','-p', type=str, help='Potential', default='dw_qb')
    args = parser.parse_args()
    main(args.z,args.pot)
