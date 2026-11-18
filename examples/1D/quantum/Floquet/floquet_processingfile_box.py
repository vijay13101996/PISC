from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct, compute_ipr
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
import numpy as np
import matplotlib
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.animation import FuncAnimation

if(1): # 1D Box
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

if(0):
    k = 0
    y_arr = pos_mat[k,k+1::2]
    x_arr = vals[k+1::2]-vals[k]
    
    #Log-Log fit of x_arr vs y_arr
    polyfit = np.polyfit(np.log(x_arr), np.log(np.abs(y_arr)), 1)
    print('polyfit', polyfit)
    
    plt.plot(np.log(x_arr), polyfit[0]*np.log(x_arr)+polyfit[1], label='fit: y = {:.2f} log(x) + {:.2f}'.format(polyfit[0], polyfit[1]))
    plt.plot(np.log(x_arr), np.log(np.abs(y_arr)), label='log(|O_{0,k}|)')
    plt.title('Position operator matrix elements for 1D box potential')
    plt.xlabel('k')
    plt.ylabel('log(|O_{0,k}|)')
    plt.grid()  
    plt.show()

g = 3.0
w_F = 5.0
w_F_arr = np.arange(0.5, 30.01, 0.25)
if(0):
    w_F_arr = np.arange(0.5, 50.01, 0.25)
    Ofi_avg = np.zeros(len(w_F_arr))
    init = 1
    fin = 3
    E_i = vals[init]
    E_f = vals[fin]
    avg = 0.0
    
    O = pos_mat
    if(1):
        O = np.zeros((neigs,neigs))
        for i in range(neigs):
            for j in range(1,neigs):
                if(pos_mat[i,j] != 0):
                    #O[i,j] =  100/(abs(vals[i]-vals[j])**1.5)#1.5)
                    #O[i,j] = 100*np.exp(-0.1*abs(vals[i]-vals[j])**2)
                    O[i,j] = 1/abs(j-i)**3#3
                O[j,i] = O[i,j]
        #Verify that O is Hermitian
        if(np.allclose(O, np.conjugate(np.transpose(O)))):
            print('O is Hermitian')
        #plt.plot(np.arange(neigs), O[2,:], label='O[0,:]')
        #plt.plot(np.arange(neigs), abs(pos_mat[2,:]), label='pos_mat[0,:]')
        #plt.show()
        print('O', O[3,0], O[0,1], O[3,2], O[2,1])
        #exit()

        #plt.imshow(abs(O), cmap='hot', interpolation='nearest')
        #plt.colorbar()
        #plt.show()
        #plt.imshow(abs(pos_mat), cmap='hot', interpolation='nearest')
        #plt.colorbar()
        #plt.show()
     
    for k, w_F in enumerate(w_F_arr):
        for m in range(neigs):
            E_m = vals[m]
            Ofi_avg[k] += (O[fin,m]*O[m,init])*(1/(E_i-E_m+w_F) + 1/(E_f-E_m-w_F))

    plt.plot(w_F_arr, Ofi_avg, label=r'$\sum_{i,f} |\sum_m \frac{O_{im} O_{mf}}{E_i-E_m+w_F}|^2$')
    plt.axvline(x=(vals[3]-vals[0]), color='r', linestyle='--', label=r'$w_F = E_2 - E_1$')
    plt.xlabel(r'$w_F$')
    plt.grid()
    plt.legend()
    plt.show()

if(0):
    IPR_maxlist = [5.0, 7.5, 10.0, 12.25, 14.75,17.25, 19.75, 22.25, 24.75, 27.25, 29.75]
    IPR_maxlist_op = [3.75,6.25, 8.75, 11.25, 13.75, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5] 
    i = 4
    N = 100
    key = 'pos_mat'#,'gaussian_decay','exponential_decay','no_decay']#,'landau']
    #for w_F in w_F_arr:
    w_F =14.75# 11.25 # 14.75
    
    print('key', key, 'w_F', w_F)
    U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)
    psi_init = np.zeros(neigs)
    psi_init[i] = 1.0

    fig, ax = plt.subplots()
    def frame(n):
        ax.clear()
        prob_n = np.abs(np.linalg.matrix_power(U_F, n).dot(psi_init))**2
        ax.bar(np.arange(neigs), prob_n, width=0.5, alpha=0.9, label=f'$n={n}$')
        ax.set(xlim=(-1,10), ylim=(-0.1,1), xlabel='k', ylabel=r'$|\psi_n(k)|^2 = |\langle{{k}}|\hat{{U}}_F^n|{} \rangle|^2$'.format(i),
               title=f'Time evolution of |{i}> for $w_F={w_F}$')
        ax.legend()

    anim = FuncAnimation(fig, frame, frames=N, interval=100)
    anim.save(f'evol_{i}_wF{w_F}_tp.gif', fps=10)   # or .gif 
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


#w_F_arr = np.linspace(5.0, 10.0, 21)
#w_F_arr = np.linspace(20.0, 30.0, 21)
#w_F_arr = np.arange(0.5,5.01,0.1)
#w_F_arr = np.arange(0.5,3.01,0.1)
#w_F_arr = np.linspace(1.0, 5.0, 21)
#w_F_arr = w_F_arr[:10]

#w_F_arr = np.linspace(0.2, 3.0, 29)
#w_F_arr = w_F_arr[:10]

g_arr = np.arange(0.5,10.01,0.25)

key_list = ['pos_mat']#,'gaussian_decay','exponential_decay','no_decay']#,'landau']
#key_list = ['dir_pl_decay_hyp']
key_list = ['linear_growth']
label_list = [r'$\hat{{O}} = \hat{{x}}$', r'$\hat{{O}} = \hat{{O}}_g$', r'$\hat{{O}} = \hat{{O}}_e$', r'$\hat{{O}} = \hat{{u}}$',r'$\hat{{O}} = \hat{{O}}_L$']

neigs= 2**9-1
m = 1.0
vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)

colors = ['r','g','b','k','m']
if(0): #Plot level spacing of quasienergies
    for w_F in w_F_arr:
        for c, key, label in zip(colors, key_list, label_list):
            U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, 'pos_mat', force_recompute=False)
            print('U_F shape', U_F.shape, 'qE shape', qE.shape, 'qS shape', qS.shape)
            plt.hist(np.diff(qE), bins=50, density=True, alpha=0.5, color=c, label='key={}, w_F={}'.format(key, w_F))
            plt.xlabel('Quasienergy level spacing')
            plt.ylabel('Density')
            plt.title('Potential: {}, g={}, operator: {}'.format(potkey, g, key))
            plt.legend()
            plt.show()
if(0): # Section along g for fixed w_F and keys
    for g in g_arr:
        for c, key in zip(colors, key_list):
            U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)
            #print('U_F shape', U_F.shape, 'qE shape', qE.shape, 'qS shape', qS.shape)

            ipr_max = compute_ipr(qS, metric='max')
            print('w_F', w_F, 'g', g, 'key', key, 'ipr_max', ipr_max) 
            if(g == g_arr[0]):
                plt.scatter(g, ipr_max, color=c, label='key={}'.format(key), s=30)
            else:
                plt.scatter(g, ipr_max, color=c, s=30)
    plt.xlabel('g')
    plt.ylabel('Max inverse participation ratio')
    plt.title('Potential: {}, w_F={}'.format(potkey, w_F))
    plt.legend()    
    plt.show()

if(0):
    key = 'pos_mat'
    for w_F in w_F_arr:
        U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)
        ipr_max,E_ind,argmax_ipr = compute_ipr(qS, metric='max')
        print('w_F', w_F,'ipr_max', np.around(ipr_max,3), 'E_ind', E_ind, 'dE', np.round(vals_anal[E_ind[1]]-vals_anal[E_ind[0]],3))
        print('w_F', w_F, 'dE', np.round(vals_anal[E_ind[1]]-vals_anal[E_ind[0]],3), 'dqE', np.round(qE[E_ind[1]]-qE[E_ind[0]],3))
        
        plt.plot(np.abs(qS[:,argmax_ipr])**2, label='Floquet state with max IPR={:.2f}'.format(ipr_max))
        plt.scatter(E_ind, abs(qS[E_ind,argmax_ipr])**2, color='b', label='Top 2 projections')
        plt.annotate('E_ind={}'.format(E_ind), xy = (0.6,0.6), xycoords='axes fraction', fontsize=10)
        plt.title('Floquet state with max IPR for potential: {}, g={}, w_F={}'.format(potkey, g, w_F))
        plt.xlabel('State index')
        plt.ylabel('Amplitude')
        plt.ylim(0,1)
        #plt.legend()
        plt.show()


if(1): # Section along w_F for fixed g and keys
    Diff= []
    Diff_qE = []
    print('key_list', key_list,label_list)
    plt.title('Maximal IPR for potential: {}, g={}'.format(potkey, g))
    IPR_maxlist = [5.0, 7.5, 10.0, 12.25, 14.75,17.25, 19.75, 22.25, 24.75, 27.25, 29.75]
    IPR_maxlist_op = [3.75,6.25, 8.75, 11.25, 13.75, 16.0, 18.5, 21.0, 23.5, 26.0, 28.5] 
    for w_F in w_F_arr:
        for c, key, label in zip(colors, key_list, label_list):
            print('key', key, 'w_F', w_F)
            U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, key, force_recompute=False)
            #print('U_F shape', U_F.shape, 'qE shape', qE.shape, 'qS shape', qS.shape)

            ipr_max,E_ind,max_ind = compute_ipr(qS, metric='max')
            #print('w_F', w_F, 'g', g, 'key', key, 'ipr_max', ipr_max, 'dE', vals_anal[E_ind[1]]-vals_anal[E_ind[0]])
            print('w_F', w_F,'ipr_max', np.around(ipr_max,3), 'E_ind', E_ind, 'dE', np.round(vals_anal[E_ind[1]]-vals_anal[E_ind[0]],3))
            print('w_F', w_F, 'dE', np.round(vals_anal[E_ind[1]]-vals_anal[E_ind[0]],3), 'dqE', np.round(qE[E_ind[1]]-qE[E_ind[0]],3))
            #Diff.append(vals_anal[E_ind[1]]-vals_anal[E_ind[0]])
            Diff.append(vals[E_ind[1]]-vals[E_ind[0]])
            Diff_qE.append(qE[E_ind[1]]-qE[E_ind[0]])
            diff_curr = vals[E_ind[1]]-vals[E_ind[0]]
            if(w_F == w_F_arr[0]):
                plt.scatter(w_F, ipr_max, color=c, label=label, s=15, zorder=2)
            else:
                plt.scatter(w_F, ipr_max, color=c, s=15, zorder=2)
                #plt.annotate('w_F={:.2f},ipr={:.2f},\n dE={:.2f},states={},{}'.format(w_F,ipr_max,diff_curr,E_ind[0],E_ind[1]), \
                #        (0.7, 0.7), textcoords="axes fraction", xycoords='axes fraction', fontsize=8, color=c)
            #plt.show()
            if(w_F in IPR_maxlist):
                plt.scatter(w_F, ipr_max, color='m', s=20, linewidth=1.5, zorder=5)
            if(w_F in IPR_maxlist_op):
                plt.scatter(w_F, ipr_max, color='g', s=20, linewidth=1.5, zorder=5)


    diff10 = vals[1]-vals[0]
    diff20 = vals[2]-vals[0]
    diff21 = vals[2]-vals[1]
    diff32 = vals[3]-vals[2]
    diff43 = vals[4]-vals[3]
    diff31 = vals[3]-vals[1]
    diff42 = vals[4]-vals[2]
    diff53 = vals[5]-vals[3]
    diff30 = vals[3]-vals[0]
    diff54 = vals[5]-vals[4]
    diff65 = vals[6]-vals[5]
    diff76 = vals[7]-vals[6]

    if(0):
        plt.axvline(x=diff10, color='k', linestyle='--', label=r'$w_F = E_1 - E_0$')
        #plt.axvline(x=diff20, color='m', linestyle='--', label=r'$w_F = E_2 - E_0$')
        #plt.axvline(x=diff21, color='y', linestyle='--', label=r'$w_F = E_2 - E_1$')
        #plt.axvline(x=diff32, color='c', linestyle='--', label=r'$w_F = E_3 - E_2$')
        #plt.axvline(x=diff43, color='orange', linestyle='--', label=r'$w_F = E_4 - E_3$')
        plt.axvline(x=diff31, color='g', linestyle='--', label=r'$w_F = E_3 - E_1$')
        plt.axvline(x=diff42, color='b', linestyle='--', label=r'$w_F = E_4 - E_2$')
        plt.axvline(x=diff53, color='r', linestyle='--', label=r'$w_F = E_5 - E_3$')
        #plt.axvline(x=diff30, color='purple', linestyle='--', label=r'$w_F = E_3 - E_0$')
        plt.axvline(x=diff54, color='orange', linestyle='--', label=r'$w_F = E_5 - E_4$')
        plt.axvline(x=diff65, color='brown', linestyle='--', label=r'$w_F = E_6 - E_5$')


    plt.xlabel(r'$w_F$')
    plt.ylabel(r'Max IPR')
    if(potkey == 'L_{}_1D_box'.format(L)):
        plt.title(r'Maximal IPR for 1D box potential')
    elif(potkey == 'L_{}_quartic'.format(L)):
        plt.title(r'Maximal IPR for quartic potential')
    elif(potkey == 'L_{}_harmonic'.format(L)):
        plt.title(r'Maximal IPR for harmonic potential')
    #plt.title('Potential: {}, g={}'.format(potkey, g))
    #plt.ylabel('Number of mixed states')
    #plt.title('Number of mixed states for different w_F and keys, g_opt={}'.format(g_opt))
    plt.legend()
    plt.ylim(0.5,30) #10)
    plt.legend(ncol=2) 
    plt.savefig('max_ipr_maxima_vs_wF_{}_g{}_FLOQUET.png'.format(potkey, g), dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(w_F_arr, abs(np.array(Diff)), color='b')#, label=r'$\Delta E$ of max IPR states')
    #plt.plot(w_F_arr, 500*abs(np.array(Diff_qE)), color='m', label='dqE of max IPR states')
    plt.plot(w_F_arr, w_F_arr, color='r', linestyle='--', label=r'$w_F$')
    plt.plot(w_F_arr, 2*w_F_arr, color='g', linestyle='--', label=r'$2w_F$')
    plt.plot(w_F_arr, 3*w_F_arr, color='k', linestyle='--', label=r'$3 w_F$')
    #plt.plot(w_F_arr, 10*w_F_arr, color='c', linestyle='--', label=r'$10 w_F$')
    plt.ylim(0, 120)
    plt.xlabel(r'$w_F$')
    plt.ylabel(r'$\Delta E$ of max IPR states')
    if(potkey == 'L_{}_1D_box'.format(L)):
        plt.title(r'$\Delta E$ of max IPR states for 1D box potential, $\hat{{O}} = \hat{{x}}$')
    elif(potkey == 'L_{}_quartic'.format(L)):
        plt.title(r'$\Delta E$ of max IPR states for quartic potential, $\hat{{O}} = \hat{{x}}$')
    elif(potkey == 'L_{}_harmonic'.format(L)):
        plt.title(r'$\Delta E$ of max IPR states for harmonic potential, $\hat{{O}} = \hat{{x}}$')
    plt.legend()
    #plt.savefig('deltaE_max_ipr_vs_wF_{}_g{}_FLOQUET.png'.format(potkey, g), dpi=300, bbox_inches='tight')
    plt.show()

def compute_corr(O, U_F, t, beta, vals):
    U_Fd = np.conjugate(np.transpose(U_F))
    U_Ft = np.linalg.matrix_power(U_F, t)
    U_Ftd = np.conjugate(np.transpose(U_Ft))
    O_t = np.dot(U_Ftd, np.dot(O, U_Ft))

    #Compute thermal state
    Z = np.sum(np.exp(-beta*vals))
    rho_beta_half = np.diag(np.exp(-beta*vals/2))

    #Ct = 1/Z Tr[rho^{1/2} O_t rho^{1/2} O]
    corr = np.trace(np.dot(rho_beta_half, np.dot(O_t, np.dot(rho_beta_half, O))))
    return corr

if(1):
    #Compute xx-correlation function wrt with initial thermal state
    beta = 0.1
    w_F = 5.0
    t_arr = np.arange(1024)
    corr_arr = np.zeros(len(t_arr), dtype=np.complex128)
    
    try:
        #raise Exception('Force recompute of correlation function')
        t_arr = np.load('t_arr_{}_g{}_wF{}_beta{}.npy'.format(potkey, g, w_F, beta))
        corr_arr = np.load('corr_arr_{}_g{}_wF{}_beta{}.npy'.format(potkey, g, w_F, beta))
        print('Loaded t_arr and corr_arr from file')
    except:
        print('Could not load t_arr and corr_arr from file, computing...')
        O = pos_mat
        U_F, qE, qS = load_U_F(potkey, g, w_F, None, None, 'pos_mat', force_recompute=False)
        print('U_F shape', U_F.shape, 'qE shape', qE.shape, 'qS shape', qS.shape)
        U_Fd = np.conjugate(np.transpose(U_F))

        for t in t_arr:
            corr_arr[t] = compute_corr(O, U_F, t, beta, vals)
            if(t%100==0):
                print('t', t, 'corr', corr_arr[t])

        plt.plot(t_arr, np.real(corr_arr), label='Re[C(t)]')
        plt.plot(t_arr, np.imag(corr_arr), label='Im[C(t)]')
        plt.title('Correlation function for potential: {}, g={}, w_F={}, beta={}'.format(potkey, g, w_F, beta))
        plt.xlabel('t')
        plt.ylabel('C(t)')
        plt.grid()
        plt.legend()
        plt.show()

        np.save('t_arr_{}_g{}_wF{}_beta{}.npy'.format(potkey, g, w_F, beta), t_arr)
        np.save('corr_arr_{}_g{}_wF{}_beta{}.npy'.format(potkey, g, w_F, beta), corr_arr)

    #Plot correlation function   
    plt.plot(t_arr, np.real(corr_arr), label='Re[C(t)]')
    plt.plot(t_arr, np.imag(corr_arr), label='Im[C(t)]')
    plt.title('Correlation function for potential: {}, g={}, w_F={}, beta={}'.format(potkey, g, w_F, beta))
    plt.xlabel('t')
    plt.ylabel('C(t)')
    plt.grid()
    plt.legend()
    plt.show()

    #Compute Fourier transform of correlation function
    corr_ft = np.fft.fft(corr_arr)
    freq_arr = np.fft.fftfreq(len(t_arr), d=1.0)
    plt.plot(freq_arr, np.abs(corr_ft), label='|C(w)|')
    plt.title('Fourier transform of correlation function for potential: {}, g={}, w_F={}, beta={}'.format(potkey, g, w_F, beta))
    plt.xlabel('w')
    plt.ylabel('|C(w)|')
    plt.grid()
    plt.xlim(-10, 10)
    plt.legend()
    plt.show()

    




    





