import numpy as np
from PISC.engine.instools import find_instanton, find_extrema, inst_init, inst_double
from PISC.utils.readwrite import read_arr, store_arr
from PISC.potentials import quartic_bistable, DW_Morse_harm
from matplotlib import pyplot as plt
import os
import argparse

### Potential parameters
m=0.5
dim=2

lamda = 2.0
g = 0.08
Vb = lamda**4/(64*g)
Tc = 0.5*lamda/np.pi

D = 3*Vb
alpha = 0.382
#print('Vb',Vb, 'D', D)

path = os.path.dirname(os.path.abspath(__file__))

def find_instanton_DW(nbeads,m,pes,beta,potkey,Tkey,ax=None,plt=None,plot=False,step=1e-4,tol=1e-3,nb_start=4,qinit=None,path=None):
    print('Starting')
    sp=np.array([0.0,0.0])
    eigvec = np.array([1.0,0.0])

    nb = nb_start
    if(qinit is None):
            qinit = inst_init(sp,0.1,eigvec,nb)     
    while nb<=nbeads:
            instanton = find_instanton(m,pes,qinit,beta,nb,dim=2,scale=1.0,stepsize=step,tol=tol,plot=False,ax=ax,plt=plt)
            if(path is not None):
                    store_arr(instanton,'Instanton_NEW_{}_{}_nbeads_{}'.format(potkey, Tkey,nb),'{}/Datafiles/'.format(path))    
            print('Instanton config. with nbeads=', nb, 'computed') 
            if(ax is not None and plot is True):
                    print('here')
                    ax.scatter(instanton[0,0],instanton[0,1],label='nbeads = {}'.format(nb))        
                    plt.pause(0.01)
            qinit=inst_double(instanton)
            nb*=2
    print('Exiting',instanton.sum(axis=2))
    if(plot):
            plt.show()      
    return instanton

def main(z, times, nbeads, pot):
    if pot=='dw_qb':
        pes = quartic_bistable(alpha,D,lamda,g,z)
        potkey = 'double_well_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)    
    elif pot=='dw_harm':
        pes = DW_Morse_harm(alpha,D,lamda,g,z)
        potkey = 'DW_Morse_harm_2D_alpha_{}_D_{}_lamda_{}_g_{}_z_{}'.format(alpha,D,lamda,g,z)

    T = times*Tc
    beta = 1/T
    Tkey = 'T_{}Tc'.format(times)

    nbeads=32

    fig,ax = plt.subplots(1)
    xg = np.linspace(-5,5,int(1e2)+1)
    yg = np.linspace(-3,7,int(1e2)+1)
    xgrid,ygrid = np.meshgrid(xg,yg)
    potgrid = pes.potential_xy(xgrid,ygrid)

    #Display where the Instanton is
    #ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0.0,D,D/20))

    instanton = find_instanton_DW(nbeads,m,pes,beta,potkey,Tkey,ax=ax,plt=plt,plot=False,path=path)
    #plt.show()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--z','-z', type=float, help='z',default=2.0)
    argparser.add_argument('--times','-t', type=float, help='times',default=1.0)
    argparser.add_argument('--nbeads','-nb', type=int, help='nbeads',default=32)
    argparser.add_argument('--pot','-p', type=str, help='pot',default='dw_harm')
    args = argparser.parse_args()
    main(args.z, args.times, args.nbeads, args.pot)

if(0):
    fig,ax = plt.subplots(1)
    xg = np.linspace(-5,5,int(1e2)+1)
    yg = np.linspace(-3,7,int(1e2)+1)
    xgrid,ygrid = np.meshgrid(xg,yg)
    potgrid = pes.potential_xy(xgrid,ygrid)

    #Display where the Instanton is
    ax.contour(xgrid,ygrid,potgrid,levels=np.arange(0.0,D,D/20))

    #Gives a slightly unconverged instanton configuration
    instanton = find_instanton_DW(nbeads,m,pes,beta,potkey,Tkey,ax=ax,plt=plt,plot=True,path=path) 

    #instanton = read_arr('Instanton_{}_{}_nbeads_{}'.format(potkey,Tkey,nbeads),'{}/Datafiles'.format(path)) #instanton
                    
    #Gives instanton configuration upto an accuracy of 1e-4
    #inst_opt = find_instanton_DW(32,m,pes,beta,potkey,Tkey,ax,plt,plot=True,step=1e-3,tol=1e-10,nb_start=32,qinit=instanton,path=path)

    plt.show()


