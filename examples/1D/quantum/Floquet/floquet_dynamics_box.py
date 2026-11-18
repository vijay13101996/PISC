from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct, compute_ipr
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
import numpy as np
import matplotlib
import scipy
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation

m = 1.0
L = 2.0
ngrid = 2**9 
neigs = ngrid-1

lb = 0
ub = L

def pot(x):
    if(x>L or x<0):
        return 1e10
    else:
        return 0.0

potkey = 'L_{}_1D_box_0'.format(L)
DVR = DVR1D(ngrid, lb, ub,m, pot)
vals,vecs = DVR.Diagonalize(neig_total=neigs)

x_arr = DVR.grid[1:ngrid]
dx = x_arr[1]-x_arr[0]

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid-1))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:ngrid]/L)

vals = vals_anal
vecs = vecs_anal

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)

pos_mat = O_anal - L/2*np.diag(np.ones(neigs))

#--------------------------------------------------------------------------------------

nph = 2
IPR_maxlist_2p = [5.0, 7.5, 10.0, 12.25, 14.75,17.25, 19.75, 22.25, 24.75, 27.25, 29.75]
IPR_maxlist_1p = [3.75,6.25, 8.75, 11.25, 13.75, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5] 

g_arr = np.linspace(0.0, 2.0, 21)
#Extend g_arr to include 2-4
g_arr = np.concatenate((g_arr, np.linspace(2.0, 4.0, 21)))

w_F_2p = np.around((vals[6]-vals[4])/2, decimals=3)
w_F_1p = np.around((vals[1]-vals[0]), decimals=3)
w_F_arr = np.arange(0.5, 30.01, 0.25)

N = 500
key = 'pos_mat'
if(nph==1):
    n = 0
    w_F = w_F_1p 
elif(nph==2):
    n = 4
    w_F = w_F_2p

H_2L_diag = np.array([[vals[n], 0],[0, vals[n+1] - w_F]])
H_2L_offdiag = np.array([[0, pos_mat[n,n+1]/2j], [-pos_mat[n+1,n]/2j, 0]])

H_3L_diag = np.array([[vals[n], 0, 0],[0, vals[n+1] - w_F, 0],[0, 0, vals[n+2] - 2*w_F]])
H_3L_offdiag = np.array([[0, pos_mat[n,n+1]/2j, 0], [-pos_mat[n,n+1]/2j, 0, pos_mat[n+1,n+2]/2j],[0, -pos_mat[n+1,n+2]/2j, 0]])

def find_qS_pm(qS, n, nph=1):
    """
    Given an array of quasistates qS, find the states with top 2 (or 3) 
    maximum projection on n, n+1 (and n+2) and return their indices and projections.
    """
    proj_n = np.abs(qS[n,:])**2
    proj_n1 = np.abs(qS[n+1,:])**2
    proj_n2 = np.abs(qS[n+2,:])**2

    #Find the indices of the top 2/3 maximum projections
    if(nph==1):
        proj_tot = proj_n + proj_n1
        max_inds = np.argsort(proj_tot)[-2:][::-1]
    elif(nph==2):
        proj_tot = proj_n + proj_n1 + proj_n2
        max_inds = np.argsort(proj_tot)[-3:][::-1]
    return max_inds

if(0):
    #Plot max_ipr vs g at w_F = w_F_2p and w_F = w_F_1p
    for g in g_arr:
        U_F, qE, qS = load_U_F(potkey, g, w_F_1p, None, None, key, force_recompute=False)
        ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')
        print('g', g, 'ipr_max', ipr_max, 'E_ind', E_ind, 'max_ind', max_ind)
        plt.scatter(g, ipr_max, color='b', s=10)

    plt.xlabel('g')
    plt.ylabel('Max IPR')
    plt.title('Max IPR vs g for w_F={}'.format(w_F_2p))
    plt.show()

ref = 0.0
for g in g_arr:
    print('key', key, 'w_F', w_F)
    U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)

    print('n',n)
    pm = find_qS_pm(qS, n, nph=nph)

    ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')
    print('g', g, 'ipr_max', ipr_max, 'E_ind', E_ind, 'max_ind', max_ind, 'i', i, 'pm', pm)

    if(nph==1):
        H_2L = H_2L_diag + g*H_2L_offdiag
        vals_tls, vecs_tls = np.linalg.eig(H_2L)
    elif(nph==2):
        H_3L = H_3L_diag + g*H_3L_offdiag
        vals_tls, vecs_tls = np.linalg.eig(-H_3L)

    if(1):
        plt.scatter(np.ones(len(qE))*g, qE, s=1)
        #Highlight the floquet state with maximum ipr
        #plt.scatter(g, qE[max_ind], color='r', label='max ipr={:.2f}'.format(ipr_max), s=10, alpha=0.5)
        if(g==0.0):
            print('refs', qE[pm[0]], qE[pm[1]], vals_tls[0], vals_tls[1])
            ref = qE[pm[0]]-vals_tls[0]
            
        for i,c in zip(range(len(pm)),['r','g', 'b']):
            plt.scatter(g, qE[pm[i]], color=c, s=20, alpha=0.5)
            plt.scatter(g, vals_tls[i]+ref, color=c, marker='x')

        del_Epm = abs(qE[pm[0]]-qE[pm[1]])
        #plt.scatter(g, del_Epm, color='r')
    
    if(0): # Plot projections of the floquet state with maximum ipr on |0> and |1> (and |2> if nph=2)

        plt.plot(np.abs(qS[:,max_ind])**2, label='floquet state with max ipr={:.2f}'.format(ipr_max), lw=3, ls='--')
        for i in range(len(pm)):
            plt.plot(np.abs(qS[:,pm[i]])**2, label='projection on |{}> and |{}>'.format(i,i+1))
        #plt.scatter(e_ind, abs(qs[e_ind,argmax_ipr])**2, color='b', label='top 2 projections')
        #plt.annotate('e_ind={}'.format(e_ind), xy = (0.6,0.6), xycoords='axes fraction', fontsize=10)
        plt.title('floquet state with maximum ipr,g={:.2f}, w_F={}'.format(g,w_F))
        plt.xlabel('state index')
        plt.ylabel('amplitude')
        plt.ylim(0,1)
        plt.xlim(0,10)
        #plt.legend()
        plt.show()

plt.xlabel('g')
plt.ylabel('Floquet quasienergy')
plt.title('Floquet quasienergies vs g for w_F={}'.format(w_F))
plt.show()

psi_init = np.zeros(neigs)
psi_init[i] = 1.0


Pnm1 = np.zeros(N)
Pn = np.zeros(N)
Pn1 = np.zeros(N)
Pn2 = np.zeros(N)
Pn3 = np.zeros(N)
Ptot = np.zeros(N)
psi_n = psi_init.copy()


for n in range(N):
    prob_n = np.abs(psi_n)**2
    Pn[n] = prob_n[i]
    Pn1[n] = prob_n[i+1]
    Pn2[n] = prob_n[i+2]
    Pn3[n] = prob_n[i+3]
    Pnm1[n] = prob_n[i-1]
    Ptot[n] = Pn[n] + Pn1[n] + Pn2[n]
    psi_n = U_F.dot(psi_n)

H_F = np.diag(vals) + g*pos_mat
H_F_3L = np.zeros((3,3),dtype=complex)
U_F_3L = np.eye(3,dtype=complex)
#Integrate over one period to get U_F_3L
dt = 1e-6
t_arr = np.arange(0, 2*np.pi/w_F, dt)
for t in t_arr:
    H_t = np.diag(vals[i:i+3]) + g*np.sin(w_F*t)*pos_mat[i:i+3,i:i+3]
    U_F_3L = scipy.linalg.expm(-1j*H_t*dt).dot(U_F_3L)

#Print U_F_3L and U_F[i:i+3,i:i+3]
print('U_F_3L', U_F_3L)
print('U_F[i:i+3,i:i+3]', U_F[i:i+3,i:i+3])
#Check if U_F_3L is unitary
print('U_F_3L is unitary:', np.allclose(U_F_3L.conj().T.dot(U_F_3L), np.eye(3), atol=1e-5))

Pn_3L = np.zeros(N)
Pn1_3L = np.zeros(N)
Pn2_3L = np.zeros(N)
psi_n_3L = np.zeros(3)
psi_n_3L[0] = 1.0
for n in range(N):
    prob_n_3L = np.abs(psi_n_3L)**2
    Pn_3L[n] = prob_n_3L[0]
    Pn1_3L[n] = prob_n_3L[1]
    Pn2_3L[n] = prob_n_3L[2]
    psi_n_3L = U_F_3L.dot(psi_n_3L)


#print('Pn', Pn_3L, Pn1_3L, Pn2_3L)

fig, ax = plt.subplots()
if(1):
    plt.plot(np.arange(N), Pn, label=r'$|\psi_n({})|^2$'.format(i),c='blue')
    plt.plot(np.arange(N), Pn1, label=r'$|\psi_n({})|^2$'.format(i+1),c='orange')
    plt.plot(np.arange(N), Pn2, label=r'$|\psi_n({})|^2$'.format(i+2),c='green')
    #plt.plot(np.arange(N), Pn3, label=r'$|\psi_n({})|^2$'.format(i+3))
    #plt.plot(np.arange(N), Pnm1, label=r'$|\psi_n({})|^2$'.format(i-1))
    plt.plot(np.arange(N), Ptot, label=r'$|\psi_n({})|^2 + |\psi_n({})|^2 + |\psi_n({})|^2$'.format(i,i+1,i+2))

plt.plot(np.arange(N), Pn_3L, '--', label=r'$|\psi_n({})|^2$ (3L)'.format(i),c='blue')
plt.plot(np.arange(N), Pn1_3L, '--', label=r'$|\psi_n({})|^2$ (3L)'.format(i+1),c='orange')
plt.plot(np.arange(N), Pn2_3L, '--', label=r'$|\psi_n({})|^2$ (3L)'.format(i+2),c='green')


plt.title(r'Time evolution of state |{}> for $w_F={}$'.format(i,w_F))
plt.xlabel('n')
plt.ylabel(r'$|\psi_n(k)|^2 = |\langle{{k}}|\hat{{U}}_F^n|{} \rangle|^2$'.format(i))
plt.xlim(-1,N)
plt.ylim(-0.1,1)
plt.legend()
plt.show()


if(0):
    def frame(n):
        ax.clear()
        prob_n = np.abs(np.linalg.matrix_power(U_F, n).dot(psi_init))**2
        ax.bar(np.arange(neigs), prob_n, width=0.5, alpha=0.9, label=f'$n={n}$')
        ax.set(xlim=(-1,10), ylim=(-0.1,1), xlabel='k', ylabel=r'$|\psi_n(k)|^2 = |\langle{{k}}|\hat{{U}}_F^n|{} \rangle|^2$'.format(i),
               title=f'Time evolution of |{i}> for $w_F={w_F}$')
        ax.legend()

    anim = FuncAnimation(fig, frame, frames=N, interval=100)
    #anim.save(f'evol_{i}_wF{w_F}_tp.gif', fps=10)   # or .gif 
    plt.show()

if(1): 
    for n in range(N):
        plt.clf()
        psi_n = np.linalg.matrix_power(U_F, n).dot(psi_init)
        prob_n = np.abs(psi_n)**2
        #plt.plot(np.arange(neigs), prob_n, label='n={}, w_F={}'.format(n, w_F))
        #Plot it as histogram
        plt.bar(np.arange(neigs), prob_n, width=0.5, alpha=0.9, label=r'$n={}$'.format(n))
        plt.title(r'Time evolution of state |{}> for $w_F={}$'.format(i,w_F))

        plt.legend()
        plt.xlabel('k')
        plt.ylabel(r'$|\psi_n(k)|^2 = |\langle{{k}}|\hat{{U}}_F^n|{} \rangle|^2$'.format(i))
        plt.xlim(-1,10)
        plt.ylim(-0.1,1)
        plt.pause(0.1)
plt.legend()
plt.show()

exit()

