import numpy as np
from PISC.dvr.dvr import DVR1D
from matplotlib import pyplot as plt
from PISC.engine import Krylov_complexity
import scipy

m = 1.0
L = 10.0
ngrid = 1000
lb = 0.0
ub = L

neigs = 200

def TCF_O(vals, beta, neigs, O, t_arr):
    
    n_arr = np.arange(neigs)
    m_arr = np.arange(neigs)

    C_arr = np.zeros_like(t_arr) + 0j

    for n in n_arr:
        for m in m_arr:
            C_arr += np.exp(-beta*(vals[n]+vals[m])/2) * np.exp(1j*(vals[n]-vals[m])*t_arr) * np.abs(O[n,m])**2

    Z = np.sum(np.exp(-beta*vals))
    C_arr /= Z

    return C_arr

def Krylov_O(vals, beta, neigs, O, ncoeff):
    L = np.zeros((neigs,neigs))
    L = Krylov_complexity.krylov_complexity.compute_liouville_matrix(vals,L)

    LO = np.zeros((neigs,neigs))
    LO = Krylov_complexity.krylov_complexity.compute_hadamard_product(L,O,LO)

    barr = np.zeros(ncoeff) 
    barr = Krylov_complexity.krylov_complexity.compute_lanczos_coeffs(O, L, barr, beta, vals,0.5, 'wgm')
    return barr

def potential(x):
    if(x<lb or x>ub):
        return 1e12
    else:
        #print('x',x)
        return 0.0#x**4

def pos_mat_anal(i,j,neigs):
    if(i==j):
        return L/2
    else:
        return L*(1/(i+j+2)**2 - 1/(i-j)**2)*(1-(-1)**(i+j+2))/np.pi**2

DVR = DVR1D(ngrid, lb, ub,m, potential)

vals_anal = np.arange(1,neigs+1)**2*np.pi**2/(2*m*L**2)
vecs_anal = np.zeros((neigs,ngrid))
for i in range(neigs):
    vecs_anal[i,:] = np.sqrt(2/L)*np.sin((i+1)*np.pi*DVR.grid[1:]/L)

O_anal = np.zeros((neigs,neigs))
for i in range(neigs):
    for j in range(neigs):
        O_anal[i,j] = pos_mat_anal(i,j,neigs)


tarr = np.linspace(-100,100,10000)
beta = 0.1


#Normalise the operator
normO2 = 0.0
normO2 = Krylov_complexity.krylov_complexity.compute_ip(O_anal, O_anal, beta, vals_anal, 0.5, normO2, 'wgm')
#O_anal /= np.sqrt(normO2)


C_tcf = TCF_O(vals_anal, beta, neigs, O_anal, tarr)
barr = Krylov_O(vals_anal, beta, neigs, O_anal, 50)

#Find slope of barr
slope, intercept = np.polyfit(np.arange(len(barr)), barr, 1)
print('Slope of b_n:', slope, np.pi/beta)


C_tcf_orig = C_tcf.copy()

#Multiply C_tcf by a gaussian window to reduce fft artifacts
sigma = 5.0
window = np.exp(-0.5*(tarr)**2/sigma**2)

C_tcf = np.fft.ifftshift(C_tcf)#*window)

#Compute fourier transform of C_tcf
freqs = np.fft.fftfreq(len(tarr), d=(tarr[1]-tarr[0]))*2*np.pi
C_tcf_ft = np.fft.fft(C_tcf)
#Plot real part of C_tcf
C_tcf_ft = np.fft.fftshift(C_tcf_ft)
freqs = np.fft.fftshift(freqs)

if(0):
    #Fit freqs vs log C_tcf_ft to a line to find alpha
    freqs_fit = freqs[(freqs>10.0) & (freqs<40.0)]
    C_tcf_ft_fit = np.log(np.abs(C_tcf_ft[freqs>10.0 & freqs<40.0]))
    slope, intercept = np.polyfit(freqs_fit, C_tcf_ft_fit, 1)
    alpha = -np.pi/(2*slope)
    print('Alpha from TCF FT:', alpha)


diffs = np.diff(vals_anal)[:30]


plt.plot(freqs, np.log((C_tcf_ft)))
#plt.xlim([0,0.5])
#plt.ylim([0,500])

##Plot diffs as vertical lines
for d in diffs:
    plt.axvline(x=d, color='r', linestyle='--', alpha=0.5)

plt.xlabel('Frequency')
plt.ylabel('|C(ω)|')
plt.title('Fourier Transform of TCF')
plt.show()


fig, ax = plt.subplots(2,1,figsize=(8,6))
ax[0].plot(tarr, abs(C_tcf), label='Re C(t) TCF')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('C(t)')
ax[0].legend()  

ax[1].plot(barr, label='Krylov b_n')
ax[1].set_xlabel('n')
ax[1].set_ylabel('b_n')
ax[1].legend()  
plt.tight_layout()
plt.show()











