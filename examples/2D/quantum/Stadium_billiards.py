import numpy as np
from matplotlib import pyplot as plt
from PISC.dvr.dvr import DVR2D
import os
from PISC.utils.readwrite import store_arr, read_arr
from PISC.engine import Krylov_complexity_2D
from matplotlib.widgets import Slider

if(1): # Parameters used in the paper
    m = 0.5
    r = np.sqrt(1/(4+np.pi))
    a = r

if(0): # Parameters for a more classical stadium billiard
    m = 1.0
    r = 1.0
    a = 1.0

potkey = 'stadium_billiard_m_{}_a_{}_r_{}'.format(m,np.round(a,3),np.round(r,3))

def potential_full(x,y):
    if( (x+a)**2 + y**2 < r**2 or (x-a)**2 + y**2 < r**2):
        return 0.0
    elif(x>-a and x<a and y>-r and y<r):
        return 0.0 
    else:
        return 1e6

def potential_quart(x, y):
    # Potential for a quarter stadium billiard: 0 inside the billiard, infinity outside
    if (0 <= x <= a and 0 <= y <= r):
        return 0    # Inside the rectangular part
    elif((x - a)**2 + (y)**2 <= r**2 and x-a>0 and y>0):
        return 0    # Inside the semicircular part
    else:
        return 1e6  # A large number to represent infinity

if(1): # Compute the boundary of the quarter stadium billiard
    flat_x = np.linspace(0, a, 500)
    curv_x = np.linspace(a, (a + r), 500)
    curv_y = np.sqrt(r**2 - (curv_x - a)**2)

    bdy_x = np.concatenate((flat_x, curv_x))
    bdy_y = np.concatenate((r*np.ones_like(flat_x), curv_y))

if(1): # Plot the full stadium billiard potential
    fig,ax = plt.subplots(figsize=(3,1.5))
    ax.plot(bdy_x, bdy_y, color='g', linestyle='-', lw=2)
    ax.plot(-bdy_x, bdy_y, color='g', linestyle='-', lw=2)
    ax.plot(bdy_x, -bdy_y, color='g', linestyle='-', lw=2)
    ax.plot(-bdy_x, -bdy_y, color='g', linestyle='-', lw=2)

    if(1):#Draw the rectangular billiard on the same plot
        ax.plot([-a-r, a+r], [-r, -r], color='k', linestyle='-', lw=2)
        ax.plot([-a-r, a+r], [r, r], color='k', linestyle='-', lw=2)
        ax.plot([-a-r, -a-r], [-r, r], color='k', linestyle='-', lw=2)
        ax.plot([a+r, a+r], [-r, r], color='k', linestyle='-', lw=2)

    #Remove frame and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.savefig('billiard_box.png', bbox_inches='tight', dpi=300)
    plt.show()
    exit()

def compute_bn(vals, vecs, DVR, beta, ncoeff=70):
    x_arr = DVR.pos_mat(0)
    neigs = len(vals)
    dx = DVR.dx
    dy = DVR.dy

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
    O = pos_mat
    #for n in range(20):
    #    O = O@ pos_mat

    #vals = np.linspace(vals[0], vals[-1], neigs) + np.random.normal(0,1,size=neigs)
    #O = np.random.normal(size=O.shape) #Randomize the operator to avoid any special symmetries.
    #O = np.ones_like(O) #Use the position operator as it is.

    liou_mat = np.zeros((neigs,neigs))
    L = Krylov_complexity_2D.krylov_complexity.compute_liouville_matrix(vals,liou_mat)

    barr = np.zeros(ncoeff)
    barr = Krylov_complexity_2D.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals, 0.5, 'wgm')
        
    return pos_mat, barr

def find_nodes(mat, threshold=1e-3):
    # Find vertical nodes of the matrix mat where the values drop below a certain threshold
    nodes = []
    for i in range(mat.shape[1]):
        #If 75% of the values in the column are below the threshold, consider it a node    
        if np.sum(np.abs(mat[:, i]) < threshold) > 0.7 * mat.shape[0]:
            print('Node found at column index:', i)
            nodes.append(i)
    return nodes

if(0): # Create a grid for the potential    
    x = np.linspace(-a-r, (a + r), 500)
    y = np.linspace(-r, r, 500)
    X, Y = np.meshgrid(x, y)
    _FULLV = np.vectorize(potential)(X,Y)
    plt.imshow(V, extent=(-a-r, a + r, -r, r), origin='lower')
    
    plt.colorbar(label='Potential')
    plt.title('Potential Landscape of the Stadium Billiard')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()

# Create a DVR grid and solve for eigenvalues
#System parameters
L = (a+r)
offset = 0.0
lbx = -L*(1+offset)
ubx = L*(1+offset)
lby = -r*(1+offset)
uby = r*(1+offset)
hbar = 1.0
ngrid = 100
ngridx = ngrid
ngridy = ngrid
dx = (ubx-lbx)/ngridx
dy = (uby-lby)/ngridy

x = np.linspace(lbx,ubx,ngridx+1)
#print('Vs',pes.potential_xy(0,0))
fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey,ngrid)  
path = os.path.dirname(os.path.abspath(__file__))

DVR = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential_quart)
print('potential',potkey)    

n_eig_tot = 500
if(0): #Diagonalization
    param_dict = {'lbx':lbx,'ubx':ubx,'lby':lby,'uby':uby,'m':m,'ngridx':ngridx,'ngridy':ngridy,'n_eig':n_eig_tot}
    #with open('{}/Datafiles/Input_log_{}.txt'.format(path,potkey),'a') as f:    
    #    f.write('\n'+str(param_dict))#print(param_dict,file=f)
    
    vals,vecs = DVR.Diagonalize(n_eig_tot)

    store_arr(vecs[:,:n_eig_tot],'{}_vecs'.format(fname),'{}/Datafiles'.format(path))
    store_arr(vals[:n_eig_tot],'{}_vals'.format(fname),'{}/Datafiles'.format(path))
    plt.imshow(DVR.eigenstate(vecs[:,0])**2,origin='lower')
    plt.show()
    exit()

vals = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
vecs = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

print('E2-E1',vals[1]-vals[0])

if(1): #Quarter stadium billiards
    potkey_quart = 'quart_stadium_billiard_m_{}_a_{}_r_{}'.format(m,np.round(a,3),np.round(r,3))
    
    # Create a DVR grid and solve for eigenvalues
    L = (a+r)
    offset = 0.0
    lbx = -offset
    ubx = L*(1+offset)
    lby = -offset
    uby = r*(1+offset)
    hbar = 1.0
    ngrid = 100
    ngridx = ngrid
    ngridy = ngrid
    dx = (ubx-lbx)/ngridx
    dy = (uby-lby)/ngridy

    x = np.linspace(lbx,ubx,ngridx+1)
    fname = 'Eigen_basis_{}_ngrid_{}'.format(potkey_quart,ngrid)  
    path = os.path.dirname(os.path.abspath(__file__))

    DVR_quart = DVR2D(ngridx,ngridy,lbx,ubx,lby,uby,m,potential_quart)

    n_eig_tot = 200
    
    vals_quart = read_arr('{}_vals'.format(fname),'{}/Datafiles'.format(path))
    vecs_quart = read_arr('{}_vecs'.format(fname),'{}/Datafiles'.format(path))

    if(0):
        n = 18
        for n in range(50):
            wf = DVR_quart.eigenstate(vecs_quart[:,n])
            plt.imshow(abs(wf)**2,origin='lower',extent=(lbx,ubx,lby,uby))
            plt.title('Eigenstate {} of the quarter stadium billiard'.format(n))
            plt.pause(2.0)
        plt.colorbar(label='Probability Density')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        exit()

if(1): #Create a widget to scan through the eigenstates and the position matrix of the quarter stadium billiard
    neigs = 150
    vecs = vecs_quart[:,:neigs]
    DVR = DVR_quart
    
    x_arr = DVR.pos_mat(0)
    dx = DVR.dx
    dy = DVR.dy

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity_2D.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dy, pos_mat)
    pos_mat_cpy = pos_mat.copy()
    #for n in range(20):
    #    pos_mat = pos_mat@ pos_mat_cpy

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax0 = axs[0]
    ax1 = axs[1]

    n = 0
    wf = DVR.eigenstate(vecs[:,n])
    im0 = ax0.imshow(abs(wf)**2,origin='lower',extent=(lbx,ubx,lby,uby))
    ax0.set_title('Eigenstate {} of the quarter stadium billiard'.format(n))
    ax0.set_xlabel('x')
    ax0.set_ylabel('y') 

    #Draw the boundary of the quarter stadium billiard
    ax0.plot(bdy_x, bdy_y, color='r', linestyle='--', label='Boundary of the quarter stadium billiard')

    im1 = ax1.imshow(np.log(abs(pos_mat)))
    ax1.set_title('Position Matrix in the Eigenbasis (n_eig={})'.format(neigs))
    ax1.set_xlabel('i')
    ax1.set_ylabel('j')
    line = ax1.axvline(x=n, color='r', linestyle='--', label='Current Eigenstate Index')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Eigenstate Index', 0, neigs-1, valinit=0, valstep=1)

    def update(val):
        n = int(slider.val)
        wf = DVR.eigenstate(vecs[:,n])
        im0.set_data(abs(wf)**2)
        ax0.set_title('Eigenstate {} of the quarter stadium billiard'.format(n))
        line.set_xdata([n])
        ax1.set_title('Position Matrix in the Eigenbasis (n_eig={})'.format(neigs))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    exit() 





if(1): #Krylov complexity of the full stadium billiards
    x_arr = DVR.pos_mat(0)
    neigs = n_eig_tot
    ncoeff = 70
    beta = 1/10
    narr = [25,30,36,41,49,74,84,94]

    bounce_scar_nlist = [3,4,9,17,19,31,34,43,48,53,60,62,66,70,76,79,82,86,96]

    nlist= [i for i in narr]
    for neigs in [150]:#,150]:#,250,350]:#10,20,50,100]:
        vals_temp = vals[:neigs]
        vecs_temp = vecs[:,:neigs]

        print('vals',vals[:10])
        print('vals_quart',vals_quart[:10])
        pos_mat, barr = compute_bn(vals_temp, vecs_temp, DVR, beta, ncoeff)
        pos_mat_quart, barr_quart = compute_bn(vals_quart[:], vecs_quart[:,:], DVR_quart, beta, ncoeff)
        
        if(0):
            nodes = find_nodes(pos_mat_quart)
            #for node in nodes:
            #    plt.axvline(x=node, color='r', linestyle='--', label='Vertical Node at index {}'.format(node))
            plt.imshow(np.log(abs(pos_mat_quart)))
            plt.colorbar(label='Position Matrix Elements')
            plt.title('Position Matrix in the Eigenbasis (n_eig={})'.format(neigs))
            plt.xlabel('Eigenstate Index')
            plt.ylabel('Eigenstate Index')
            plt.show()
        
        plt.plot(barr, label='n_eig={}'.format(neigs),marker='o',markersize=5)
        plt.plot(barr_quart, label='Quarter stadium billiard',marker='x',markersize=5)

    plt.xlabel('Lanczos iteration')
    plt.ylabel('b_n')
    plt.title('Lanczos coefficients for the stadium billiard')
    plt.legend()
    plt.show()
    exit()


