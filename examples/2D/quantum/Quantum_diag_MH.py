import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials import Morse_harm_2D
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
import time 
import argparse

ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

n_eig_tot = 100 #Number of eigenstates to be calculated

#System parameters
m = 1.0
L = 5.0
lbx = -L
ubx = L
lby = -4
uby = 15
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid

omega = 1.0
xe = 0.05

alpha = np.sqrt(2*m*omega*xe)
D = omega/(4*xe)

def main(z):
    start_time = time.time()
    print('z',z)
      
    potkey = 'Morse_harmonic_m_{}_omega_{}_xe_{}_z_{}'.format(m,omega,xe,z)
    pes = Morse_harm_2D(m,omega,D,alpha,z)

    if(0):
        xg = np.linspace(lbx,ubx,ngridx)
        yg = np.linspace(lby,uby,ngridy)
        xgr,ygr = np.meshgrid(xg,yg)
        plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,1.2*D,1.0))
        plt.show()    
        exit()

    x = np.linspace(lbx,ubx,ngridx+1)
    fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
    path = os.path.dirname(os.path.abspath(__file__))

    DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,pes.potential_xy)
    print('potential',potkey)   

    #Diagonalization
    param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
    with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
        f.write('\n'+str(param_dict))

    if(1):
        vals,vecs = DVR.Diagonalize(neig_total=n_eig_tot)

        store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
        print('Time taken:',time.time()-start_time)

    if(1): #Test whether the Wavefunctions look correct
        vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
        vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        #n=8
        #print('vals[n]', vals[n])
        #plt.imshow(DVR.eigenstate(vecs[:,n])**2,origin='lower')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Morse + Harmonic 2D potential')
    parser.add_argument('--z','-z', type=float, help='z parameter', default=0.0)
    args = parser.parse_args()
    main(args.z)
