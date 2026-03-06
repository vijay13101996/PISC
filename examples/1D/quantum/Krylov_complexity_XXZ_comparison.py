import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pickle

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

fkey_NN= 'krylov_coeffs_{}_NN_beta_{}.pkl'
fkey_NNN = 'krylov_coeffs_{}_NNN_beta_{}.pkl'

op_list = ['A', 'B', 'Jz', 'JE']
op_rep = [r'$\hat{A}$', r'$\hat{B}$', r'$\hat{j}_S$', r'$\hat{j}_E$']

NNc = 'green'
NNNc = 'blue'

if(0):
    beta_list = [0.25, 1.0, 5.0]

    fig, axs = plt.subplots(3,4, figsize=(12, 9), sharex=True, sharey=True)

    for i, op in enumerate(op_list):
        for j, beta in enumerate(beta_list):
            try:
                NN_bn = pickle.load(open(fkey_NN.format(op, beta), 'rb'))
                NNN_bn = pickle.load(open(fkey_NNN.format(op, beta), 'rb'))
                print(f'Loaded {op} coefficients for beta={beta}')
            except FileNotFoundError:
                print(f'File not found for {op} at beta={beta}')
                continue
            
            axs[j, i].scatter(NN_bn[0], NN_bn[1], label='NN XXZ', s=5)
            axs[j, i].scatter(NNN_bn[0], NNN_bn[1], label='NNN XXZ', s=5)
            axs[j, i].set_title(r'$\beta={}$'.format(beta), fontsize=tp_fs)
            

    plt.savefig('krylov_coefficients_comparison.pdf', bbox_inches='tight')

if(1):
    beta_list = [1.0,10.0]

    fig, axs = plt.subplots(4,2, figsize=(6, 12), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    for i, op in enumerate(op_list):
        for j, beta in enumerate(beta_list):
            try:
                NN_bn = pickle.load(open(fkey_NN.format(op, beta), 'rb'))
                NNN_bn = pickle.load(open(fkey_NNN.format(op, beta), 'rb'))
                print(f'Loaded {op} coefficients for beta={beta}')
            except FileNotFoundError:
                print(f'File not found for {op} at beta={beta}')
                continue
            if i==0 and j==0: 
                axs[i, j].scatter(NN_bn[0], NN_bn[1], label='NN XXZ', s=5,color=NNc)
                axs[i, j].scatter(NNN_bn[0], NNN_bn[1], label='NNN XXZ', s=5, color=NNNc)
            else:
                axs[i, j].scatter(NN_bn[0], NN_bn[1], s=5, color=NNc)
                axs[i, j].scatter(NNN_bn[0], NNN_bn[1], s=5, color=NNNc)
            if(i==0):
                axs[i, j].set_title(r'$\beta={}$'.format(beta), fontsize=tp_fs)
            if i==3:
                axs[i, j].set_xlabel(r'$n$', fontsize=xl_fs)
            if j==0:
                axs[i, j].set_ylabel(r'$b_n$', fontsize=yl_fs)
            
            axs[i, j].annotate(op_rep[i], xy=(0.5, 0.9), xytext=(0.5, 0.9), textcoords='axes fraction', fontsize=ti_fs, ha='center')
            axs[i, j].set_ylim([-0.5,9.5])
    fig.legend(loc='lower center', ncol=2, fontsize=le_fs)

    plt.savefig('krylov_coefficients_comparison_2.pdf', bbox_inches='tight')
#plt.show()

