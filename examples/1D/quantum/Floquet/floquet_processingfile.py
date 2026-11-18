from matplotlib import pyplot as plt
from floquet_tools import Floquetiser, fidelity, load_U_F, compute_U_F_direct, compute_ipr
from PISC.dvr.dvr import DVR1D
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from PISC.engine import Krylov_complexity
import numpy as np
import matplotlib
plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

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

if(0): # Hexic oscillator
    m = 1.0
    L = 6.0
    ngrid = 2**9

    lb = -L
    ub = L
    def pot(x):
        return x**6

    potkey = 'L_{}_hexic'.format(L)

    DVR = DVR1D(ngrid,lb,ub,m,pot)
    neigs = 200 # :ngrid-1
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    if(0):
        plt.plot(DVR.grid[1:ngrid], vecs[:,-1])
        plt.title('Hexic oscillator, L={}'.format(L))
        plt.xlabel('x')
        plt.ylabel('Wavefunction')
        plt.grid()
        plt.show()
        exit()

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs))
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

if(0): # Quartic oscillator
    m = 1.0
    L = 10.0
    n_anharm = 4
    omega = 0.0
    a = 0.0
    b = 1.0
    ngrid = 2**10 

    lb = -L
    ub = L
    pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

    potkey = 'L_{}_quartic'.format(L) #_large_grid'.format(L)
    #potkey = 'L_{}_quartic_neigs_{}'.format(L, 100)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 200 #ngrid-1
    vals,vecs = DVR.Diagonalize(neig_total=neigs)

    diff0 = vals[1]-vals[0]
    diff1 = vals[2]-vals[1]

    print('vals',vals[0])

    print('diff',vals[1]-vals[0],vals[2]-vals[1],vals[3]-vals[2])

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs)) 
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

    if(0):
        warr = vals - vals[0]
        plt.plot(warr, np.log(abs(pos_mat[0,:])), label='pos_mat')
        plt.title('Quartic oscillator, L={}, n_anharm={}'.format(L,n_anharm))
        plt.xlabel('w')
        plt.ylabel('Matrix elements')
        plt.grid()
        plt.legend()
        plt.show()
        exit()

    if(0):
        plt.plot(np.log(abs(pos_mat[0,:])), label='pos_mat')
        plt.title('Quartic oscillator, L={}, n_anharm={}'.format(L,n_anharm))
        plt.xlabel('n')
        plt.ylabel('Matrix elements')
        plt.grid()
        plt.legend()
        plt.show()

if(0): # Harmonic oscillator
    m = 1.0
    omega = 1.0
    ngrid = 2**9
    L = 20.0
    
    lb = -L
    ub = L
    pes = mildly_anharmonic(m,0.0,0.0,w=omega,n=4)  

    potkey = 'L_{}_harmonic'.format(L)

    DVR = DVR1D(ngrid,lb,ub,m,pes.potential_func)
    neigs = 200 # ngrid-1
    vals,vecs = DVR.Diagonalize(neig_total=neigs)
    
    vals_anal = (np.arange(0,neigs) + 0.5)*omega
    
    vals = vals_anal

    x_arr = DVR.grid[:ngrid]
    dx = x_arr[1]-x_arr[0]

    pos_mat = np.zeros((neigs,neigs))
    pos_mat = Krylov_complexity.krylov_complexity.compute_pos_matrix(vecs, x_arr, dx, dx, pos_mat)

    pos_mat_anal = np.zeros((neigs,neigs))
    for i in range(neigs):
        for j in range(i,neigs):
            if abs(i-j) == 1:
                pos_mat_anal[i,j] = np.sqrt(j/2.0)
            else:
                pos_mat_anal[i,j] = 0.0
            pos_mat_anal[j,i] = pos_mat_anal[i,j]
    
    print('Position matrix (numerical):\n', np.around(pos_mat[100:110,100:110],3))
    print('Position matrix (analytical):\n', np.around(pos_mat_anal[100:110,100:110],3))

    print('vals (numerical):', np.round(vals[180:190],3))
    print('vals (analytical):', np.round(vals_anal[180:190],3))
    #assert np.allclose(abs(pos_mat), abs(pos_mat_anal), atol=1e-1), "Position matrix does not match analytical result"
    #print('Position matrix matches analytical result')

    pos_mat = pos_mat_anal

if(0):
    for i in range(neigs-1):
        plt.scatter(vals[i+1]-vals[i], abs(pos_mat[i,i+1]), color='r', s=30)
    if(potkey == 'L_{}_1D_box'.format(L)):
        plt.title(r'1D box potential, L={}'.format(L))
    elif(potkey == 'L_{}_quartic'.format(L)):
        plt.title(r'Quartic potential')
    elif(potkey == 'L_{}_harmonic'.format(L)):
        plt.title(r'Harmonic potential')
    plt.xlabel(r'$E_{n+1}-E_n$')
    plt.ylabel(r'$|x_{n,n+1}|$')
    plt.grid()
    plt.show()
    exit()

    #Find tuple (m,n) with maximum |x_mn| and corresponding w_mn
    max_xmn = 0.0
    max_mn = (0,0)
    for i in range(neigs):
        for j in range(neigs):
            if(abs(pos_mat[i,j]) > max_xmn):
                max_xmn = abs(pos_mat[i,j])
                max_mn = (i,j)
    print('Max |x_mn| = {} for m={}, n={}'.format(max_xmn, max_mn[0], max_mn[1]))
    print('Corresponding w_mn = {}'.format(vals[max_mn[1]]-vals[max_mn[0]]))
    exit()

    # Plot x_mn against w_mn as histogram
    warr = []
    xmn = []
    for i in range(neigs):
        for j in range(neigs):
            warr.append(vals[j]-vals[i])
            xmn.append(abs(pos_mat[i,j]))
    warr = np.array(warr)
    xmn = np.array(xmn)
    plt.scatter(warr, np.log(xmn), s=1)
    plt.title('{}'.format(potkey))
    plt.xlabel('w_mn')
    plt.ylabel('log(|x_mn|)')
    plt.grid()
    plt.show()
    exit()

g = 3.0
w_F = 10.0
w_F_arr = np.arange(0.5, 30.01, 0.25)
#w_F_arr = np.linspace(5.0, 10.0, 21)
#w_F_arr = np.linspace(20.0, 30.0, 21)
#w_F_arr = np.arange(0.5,5.01,0.1)
#w_F_arr = np.arange(0.5,3.01,0.1)
#w_F_arr = np.linspace(1.0, 5.0, 21)
#w_F_arr = w_F_arr[:10]

#w_F_arr = np.linspace(0.2, 3.0, 29)
#w_F_arr = w_F_arr[:10]

diff_9_10 = vals[10]-vals[9]

#w_F_arr = []
#for i in range(1,11):
#    w_F_arr.append(diff_9_10/i)
g_arr = np.arange(0.5,10.01,0.25)

key_list = ['pos_mat']#,'gaussian_decay','exponential_decay','no_decay']#,'landau']
label_list = [r'$\hat{{O}} = \hat{{x}}$', r'$\hat{{O}} = \hat{{O}}_g$', r'$\hat{{O}} = \hat{{O}}_e$', r'$\hat{{O}} = \hat{{u}}$',r'$\hat{{O}} = \hat{{O}}_L$']
#key_list = ['gaussian_decay','exponential_decay','no_decay']
#label_list = [r'$\hat{{O}} = \hat{{O}}_g$', r'$\hat{{O}} = \hat{{O}}_e$', r'$\hat{{O}} = \hat{{u}}$']
#key_list = ['no_decay_alt']
#label_list = [r'$\hat{{O}} = \hat{{u}}_{{alt}}$']

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
    plt.ylim(0.5,5) #10)
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

    




    





