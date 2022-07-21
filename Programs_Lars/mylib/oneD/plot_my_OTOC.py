import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

def plot_MC_OTOC(Cn_Fortran, pot_name, t,log_OTOC_mic=True, other_parameters=[]):
    fig, ax = plt.subplots()
    if(log_OTOC_mic==True):
        ax.plot(t,np.zeros_like(Cn_Fortran[2,:]), '--')
    else:
        ax.plot(t,np.zeros_like(Cn_Fortran[2,:])+1, '--')
    if(pot_name=='DW'):
        plot_n = (1,5,9,11,13,20)#like paper
    elif(pot_name == 'HO'):
        plot_n = (1,5,10) # should not make difference
    elif(pot_name == 'PiB'):
        plot_n = (1,2,5,10)#like paper
    else:
        plot_n = (1,5,10)
        print('Potential not known. Unless otherwise specified used default parameters 1,5,10.')
    if(len(other_parameters)!=0):
        plot_n=other_parameters

    for n in plot_n:#DW OTOC from Paper
        if(log_OTOC_mic==True):
            ax.plot(t,np.log(Cn_Fortran[n,:]), label = "n = %i" %n)
        else:
            ax.plot(t,Cn_Fortran[n,:], label = "n = %i" %n)
    if(log_OTOC_mic==True):
        ax.set_ylabel(r'log(C$_n$(t))',size=20)
    else:
        ax.set_ylabel(r'C$_n$(t)',size=20)
        if(pot_name=='PiB'):
            ax.set_yscale('log',base=2) #PiB 
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('t',size=20)
    #ax.set_xlim([0,4])# [0,4]..DW OTOC Paper
    ax.legend()
    plt.show(block=False)

def C_T(C,E,beta=1): #C is microcanonical OTOC
    Z=0 
    sum=np.zeros_like(C[1,:])#get Thermal OTOC for each time t
    for n in range(len(C[:,1])):
        Z+=np.exp(-beta *E[n]) #partition sum
    for n in range(len(C[:,1])):
        for time in range(len(C[1,:])):
            sum[time] += np.exp(-beta *E[n])*C[n,time] #weights of all microcanonical OTOCS
    return sum/Z

def plot_Thermal_OTOC(Cn_Fortran, pot_name,t,Energies, log_OTOC_T=True,other_parameters=[]):
    fig, ax = plt.subplots()
    if(log_OTOC_T==True):
        ax.plot(t,np.zeros_like(Cn_Fortran[1,:]), '--')#depends if log or not
    else:
        ax.plot(t,np.zeros_like(Cn_Fortran[1,:])+1, '--')#depends if log or not
    if(pot_name =='DW'):
        plot_T = (1,5,9,30) 
    elif(pot_name=='HO'):
        plot_T = (1,10,20)# HO doesn't matter since all the same
    elif(pot_name=='PiB'):
        plot_T = (1,20,50,100,200) 
    else:
        plot_T = (1,10,20)
        print('Potential not known. Unless otherwise specified used default parameters 1,10,20.')
    if(len(other_parameters)!=0):
        plot_T=other_parameters
    for T in plot_T:
        beta=1/T
        if(log_OTOC_T==True):
            ax.plot(t,np.log(C_T(C=Cn_Fortran,E=Energies,beta=beta)), label = "T = %.2f" %T)
        else:
            ax.plot(t,C_T(C=Cn_Fortran,E=Energies,beta=beta), label = "T = %.2f" %T)
    if(log_OTOC_T==True):
        ax.set_ylabel(r'log(C(t))',size=20)
    else:
        ax.set_ylabel('C(t)',size=20)
        if(pot_name=='PiB'):
            ax.set_yscale('log',base=2) #PiB
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('t',size=20)
    #ax.set_xlim([0,4])
    #plt.text( x= 5,y=0.08, s=r'$\beta$')
    ax.legend()
    plt.show(block=False)

