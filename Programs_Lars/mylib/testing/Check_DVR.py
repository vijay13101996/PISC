import numpy as np
from matplotlib import pyplot as plt

#count number of occurences of energy levels 
def multiplicity(E,norm=1,nr_E=50,print_energy_levels=False):
    epsilon=0.05
    cntr=np.zeros(nr_E)
    for l in range(len(cntr)):
        for k in range(len(E)):
            if(abs(E[k]/norm-l)<epsilon):
                cntr[l]+=1
    print(cntr[:])
    if(print_energy_levels==True):
        print(E[:50])

def plot_DW_MO_DW_inter(x,y,pot,vals,vals_x,vals_y,z):
    if(True):###option1, just cosmetics
        fig= plt.figure()
        gs = fig.add_gridspec(2,2,hspace=1,wspace=0)
        axs = gs.subplots(sharey='row')
        ###plot potential on x axis
        axs[0,0].plot(x,pot(x,0))
        axs[0,0].set_title('Pot x')
        ###Plot potential on y axis
        axs[0,1].plot(y, pot(0,y)-pot(0,0), 'tab:orange')
        axs[0,1].set_title('Pot y')
        #plot pot of x axis with coupled energies
        axs[1,0].plot(x,pot(x,0))
        axs[1,0].set_title('E_coupled (pot x, z = %.2f)' %z)
        #plot pot of x axis with added energies
        axs[1,1].plot(x,pot(x,0))
        axs[1,1].set_title('E_added (pot x)')
        ####Adjust here if more/less energy levels desured
        plot_to_n=12#
        vals_max=20####for y Axis lim

        #plot energy levels (x)
        for n in range(plot_to_n):
            axs[0,0].plot(x, vals_x[n]*np.ones_like(x), '--',label='n = %i' % n )
            if(vals_x[n]>vals_y[1]+vals_x[1]):
                break
        #plot energy levels (y)
        for n in range(3):
            axs[0,1].plot(y, (vals_y[n])*np.ones_like(y), '--',label='m = %i' % n )
        #plot energy levels (coupled)
        for n in range(plot_to_n):
            axs[1,0].plot(x, vals[n]*np.ones_like(x), '--',label='n = %i' % n )
            if(n%2-1):
                    axs[1,0].text(x[int(len(x)/2)], vals[n], 'n = %i' % n,fontsize=7 )
        #plot energy levels (added)
            #plot sum of n, m=0
        for n in range(plot_to_n):
            if(n==0):
                axs[1,1].plot(x, (vals_x[n]+vals_y[0])*np.ones_like(x), 'k-',linewidth=2.5)
            else:
                axs[1,1].plot(x, (vals_x[n]+vals_y[0])*np.ones_like(x), '--')
            if(n%2-1 and (vals_x[n]+vals_y[0])<vals[20]):
                    axs[1,1].text(x[0], vals_x[n]+vals_y[0] , '(%i,0)' % n,fontsize=7 )
            if(n%2 and (vals_x[n]+vals_y[0])<vals[20]):
                    axs[1,1].text(x[-10], vals_x[n]+vals_y[0], '(%i,0)' % n,fontsize=7 )
            #plot sum of n, m=1
        for n in range(plot_to_n):
            if(n==0):
                axs[1,1].plot(x, (vals_x[n]+vals_y[1])*np.ones_like(x), 'k-',linewidth=2.5)
            else:
                axs[1,1].plot(x, (vals_x[n]+vals_y[1])*np.ones_like(x), '--')
            
            if(n%2-1 and (vals_x[n]+vals_y[1])<vals[20]):
                    axs[1,1].text(x[0],vals_x[n]+vals_y[1], '(%i,1)' % n,fontsize=7 )
            if(n%2 and (vals_x[n]+vals_y[1])<vals[20]):
                    axs[1,1].text(x[-10], vals_x[n]+vals_y[1], '(%i,1)' % n,fontsize=7 )
            #plot sum of n, m=2
        for n in range(plot_to_n):    
            if(n==0):
                axs[1,1].plot(x, (vals_x[n]+vals_y[2])*np.ones_like(x), 'k-',linewidth=2.5)
            else:
               axs[1,1].plot(x, (vals_x[n]+vals_y[2])*np.ones_like(x), '--')
            
            if(n%2-1 and (vals_x[n]+vals_y[1])<vals[20]):
                    axs[1,1].text(x[0],vals_x[n]+vals_y[1], '(%i,2)' % n,fontsize=7 )
            if(n%2 and (vals_x[n]+vals_y[1])<vals[20]):
                    axs[1,1].text(x[-10], vals_x[n]+vals_y[1], '(%i,2)' % n,fontsize=7 )
        for ax in (axs[0,0],axs[1,1]):
            ax.set_ylim([0,vals[vals_max]])
        plt.show()




def plot_V_and_E(x,y,pot,vals,vecs,vals_x,vals_y,z=0,twoDplot=True,threeDplot=True):
    #maybe use vecs (for plotting) 
    #plt.savefig()#if want to save (FONTIZE important for papers) 
    #can do a lot more cosmetics x axis , y  axis title ....
    if(twoDplot==True):
        max_val=20
        fig= plt.figure()
        if(True):###option1, just cosmetics
            gs = fig.add_gridspec(2,2, hspace=1,wspace=0)
            axs = gs.subplots(sharey='row')
        else:###option2, just cosmetics 
            fig, axs = plt.subplots(2,2)
        
        ###Plot potential on x axis
        axs[0,0].plot(x,pot(x,0))
        axs[0,0].set_title('Pot x')
        ###Plot potential on y axis
        axs[0,1].plot(y, pot(0,y)-pot(0,0), 'tab:orange')
        axs[0,1].set_title('Pot y')
        for n in range(10):
            axs[0,0].plot(x, vals_x[n]*np.ones_like(x), '--',label='n = %i' % n )
            if(vals_x[n]>vals_y[1]):
                break
        for n in range(3):
            axs[0,1].plot(y, (vals_y[n])*np.ones_like(y), '--',label='m = %i' % n )
        
        #plot energy levels added up  
        for n in range(10):
            axs[1,0].plot(x, (vals_x[n]+vals_y[0])*np.ones_like(x), 'r--')
            axs[1,0].plot(x, (vals_x[n]+vals_y[1])*np.ones_like(x), 'g--')
            
            axs[1,0].text(x[int(len(x)/2)], (vals_x[n]+vals_y[0]), '(%i,0)' % n,fontsize=7 )
            if(vals[max_val] > vals_x[n]+vals_y[1]):
                axs[1,0].text(x[int(len(x)/2)], (vals_x[n]+vals_y[1]), '(%i,1)' % n,fontsize=7 )

            if((vals_x[n]+vals_y[0])>vals_y[2]):
                break

        axs[1,0].plot(x, (vals_x[1]+vals_y[1])*np.ones_like(x), 'b-',label='n=0,m=1 ')
        #plot energy levels of coupled system
        for n in range(max_val):
            if((vals[n]>(vals_x[1]+vals_y[1])) and n >14 ):
                axs[1,1].plot(x, vals[n]*np.ones_like(x),'-')
                break
            else: 
                axs[1,1].plot(x, vals[n]*np.ones_like(x),'r--' )
                if(n%2-1):
                    axs[1,1].text(x[int(len(x)/2)], vals[n], 'n = %i' % n,fontsize=7 )

        for ax in (axs[0,0], axs[0,1]):
            ax.set_ylim([0,vals[max_val]])
            ax.legend(loc='upper center',ncol=3,fancybox=True,shadow=True,fontsize=7)
        
        axs[1,0].set_ylim([vals[0]-vals_x[0],vals[20]])
        axs[1,0].set_title('Individual Energies added',fontsize= 10)#y=-0.01,fontsize=8)
        axs[1,1].set_title('Coupled system, z = %.2f' % z,fontsize=10)#,y=-0.01,fontsize=8)
        axs[1,0].set_ylabel('E')
        plt.show()
        

    if(threeDplot == True):
        #from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X,Y = np.meshgrid(x[20:-20],y[2:50])
        Z=pot(X,Y)

        #ax.contour3D(X, Y, Z, 100, cmap='binary')#very useful, 3D conture
        ax.plot_wireframe(X, Y, Z, color='black')
        #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


    plt.show()
