import numpy as np
from PISC.dvr.dvr import DVR2D
from PISC.potentials import coupled_quartic
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
from matplotlib import pyplot as plt
import os 
import time 
import argparse

ngrid = 100 #Number of grid points
ngridx = ngrid #Number of grid points along x
ngridy = ngrid #Number of grid points along y

n_eig_tot = 400 #Number of eigenstates to be calculated

#System parameters
m = 0.5
L = 10.0
lbx = -L
ubx = L
lby = -L
uby = L
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
omega = 1.0

def main(g2):
    g1 = 0.25
    start_time = time.time()
    print('g1,g2',g1,g2)
      
    potkey = 'coupled_quartic_g1_{}_g2_{}'.format(g1,g2)
    pes = coupled_quartic(g1,g2)

    if(0):
        xg = np.linspace(lbx,ubx,ngridx)
        yg = np.linspace(lby,uby,ngridy)
        xgr,ygr = np.meshgrid(xg,yg)
        plt.contour(pes.potential_xy(xgr,ygr),levels=np.arange(0,10,1.0))
        plt.show()    
        exit()

    x = np.linspace(lbx,ubx,ngridx+1)
    #print('Vs',pes.potential_xy(0,0))
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
        
        vals = vals[:n_eig_tot]
        vecs = vecs[:,:n_eig_tot]

        # Remove degeneracies in eigenvalues 
        ndec = 3
        vals_ndeg, ind = np.unique(np.round(vals,decimals=ndec),return_index=True)
        vecs_ndeg = vecs[:,ind]

        #vals_ndeg2 = np.round(vals[ind],decimals=ndec)
        #print('vals_ndeg',vals_ndeg,vals_ndeg2)

        store_arr(vecs,'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        store_arr(vals,'{}_vals'.format(fname),'{}/Datafiles'.format(path))
        
        store_arr(vecs_ndeg,'{}_vecs_ndeg'.format(fname),'{}/Datafiles'.format(path))
        store_arr(vals_ndeg,'{}_vals_ndeg'.format(fname),'{}/Datafiles'.format(path))

        print('Time taken:',time.time()-start_time)

    if(1): #Test whether the Wavefunctions look correct
        vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
        vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))
        
        vals_ndeg = read_arr('{}_vals_ndeg'.format(fname),'{}/Datafiles'.format(path))
        vecs_ndeg = read_arr('{}_vecs_ndeg'.format(fname),'{}/Datafiles'.format(path))

        #n=8
        #print('vals[n]', vals[n])
        #plt.imshow(DVR.eigenstate(vecs[:,n])**2,origin='lower')

    vals = vals[:500]
    vals_ndeg = vals_ndeg[:500]

    if(1):
        plt.plot(np.arange(len(vals)),vals)
        plt.scatter(np.arange(len(vals)),vals,s=2,color='r')
        
        plt.plot(np.arange(len(vals_ndeg)),vals_ndeg,color='k')
        plt.scatter(np.arange(len(vals_ndeg)),vals_ndeg,s=2,color='b')

        plt.show()    
        exit()

    if(1):
        diff = np.diff(vals)
        #diff = diff[200:]
        
        plt.hist(diff,bins=80)
        #plt.scatter(np.arange(len(diff)),diff)
        plt.show()
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coupled quartic oscillator potential')
    parser.add_argument('--g2','-g2', type=float, help='g2 parameter', default=0.1)
    args = parser.parse_args()
    main(args.g2)
