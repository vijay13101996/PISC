import numpy as np
from PISC.engine import Krylov_complexity
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic, double_well
from PISC.utils.readwrite import store_1D_plotdata, read_1D_plotdata, store_arr, read_arr
import scipy
from matplotlib import pyplot as plt
from compute_lanczos_moments import compute_Lanczos_iter, compute_Lanczos_det
import argparse
import matplotlib


def compute_bn(O,vals,vecs,T_au,ncoeff):

    neigs = len(vals)
    beta = 1.0/T_au
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    #moments = np.zeros(nmoments+1)
    #moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, 'asm', 0.5, moments)
    #even_moments = moments[0::2]
    #mun_arr.append(even_moments)
    #mun_arr = np.array(mun_arr)

    bnarr = np.zeros(ncoeff)
    bnarr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, bnarr, beta, vals,0.5, 'wgm')
    
    bnarr = np.array(bnarr)
    coeffarr = np.arange(ncoeff) 
    #Print all values upto zero of bnarr
    #print('bnarr',bnarr)

    return coeffarr,bnarr

def compute_moments(O,vals,T_au,nmoments):
    neigs = len(vals)
    beta = 1.0/T_au
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    moments = np.zeros(nmoments+1)
    moments = Krylov_complexity.krylov_complexity.compute_moments(O, vals, beta, 'asm', 0.5, moments)
    even_moments = moments[0::2]
    n_arr = np.arange(nmoments+1)
    return n_arr, even_moments

def comput_On_matrix(O,beta,vals,nmat,lamda=0.5,ip='wgm'): 
    neigs = len(vals)
    
    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    On = np.zeros((neigs,neigs))
    barr = np.zeros(nmat+1)

    barr, On = Krylov_complexity.krylov_complexity.compute_on_matrix(O, L, barr, beta, vals, lamda, ip, On, nmat+1) 
    return barr, On   

