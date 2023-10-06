###=============================================================###
###  Exact evaluation of KT and DKT correlations for simple	###
###  Hamiltonians using Gauss-Hermite (i.e. HO) basis sets.	###
###  The 2D TCF computed are:									###
###  1a) DKT correlation <A;B(t1);C(t2)>						###
###  1b) DKT correlation <dot(A);dot(B)(t1);C(t2)>						###
###  2) standard TCF <AB(t1)C(t2)>								###
###  3) standard TCF <B(t1)C(t2)A>								###
###  4) standard TCF <C(t2)AB(t1)>								###
###  5) standard TCF <AC(t2)B(t1)>								###
###  6) standard TCF <C(t2)B(t1)A>								###
###  7) standard TCF <B(t1)AC(t2)>								###
###  The 1D TCF computed are:									###
###  1) KT correlation <A;B(t1)>								###
###=============================================================###
hbar=1.0
import sys
import numpy as np
import scipy.linalg as linn
import time as get_time
only_freq =True
only_freq =False
compute_DKT=True
compute_PB=True
compute_std=True
t_init=get_time.time()
###================================================================
### read some parameters from command line

### usage
if(len(sys.argv)<=2):
	print('Usage: python {} potential beta A B C (cubic coef)'.format(sys.argv[0]))
	print('Potential should be "HO" or "WAP" or "QP" or "WQP" or "OHqtip4pf" or "QPper" or "Chaos" ')
	print('A,B,C should be x or x2')
	sys.exit() 
### potential
potential=sys.argv[1]
if(potential!='HO' and potential!='WAP' and potential!='QP' and potential!='WQP' and potential!='OHqtip4pf' and potential!='QPper' and potential!='Chaos'\
	and potential!='Tanimura_SB'):
	sys.exit('potential "{}" not defined. potential should be "HO" or "WAP" or "QP" or "WQP" or "OHqtip4pf" or "QPper" or "Chaos" or "Tanimura_SB" '.format(potential))
print('potential = ',potential)

if(len(sys.argv)!=6 and len(sys.argv)!=7):
	sys.exit('Usage: python {} potential beta A B C (cubic coef)'.format(sys.argv[0]))


### beta: length of the imaginary timestep (au)
beta=float(sys.argv[2])
print('beta = ',beta)

### observables
A_obs = sys.argv[3]
B_obs = sys.argv[4]
C_obs = sys.argv[5]

if(A_obs!='x' and A_obs!='x2' and B_obs!='x' and B_obs!='x2' and C_obs!='x' and C_obs!='x2'):
	sys.exit('observable not defined. observable should be "x" or "x2"')
print('A = ',A_obs)
print('B = ',B_obs)
print('C = ',C_obs)

###================================================================

###================================================================
### define some parameters of the simulation

dt = .5			# time step (au)
nstep = 100	# number of time step
#nstep = 200	# number of time step


#Standard yair
nb = 200		# number of basis functions to use
tnb = 16		# truncated basis set 

#Tighter 1
#nb = 400		# number of basis functions to use
#tnb = 16		# truncated basis set 

#Tighter 2
#nb = 400		# number of basis functions to use
#tnb = 32		# truncated basis set 

print('Simulation parameters')
print('dt = ',dt)
print('nstep = ',nstep)
print('nb/tnb = ',nb,tnb)

###================================================================

###================================================================
### define some parameters of the Hamiltonian
### Note: using m=omega=1
### H = p^2/2 + a*x^2 + b*x^3 + c*x^4

if(potential=='HO'):
	### Harmonic Oscillator
	a = 0.5
	b = 0.
	c = 0.
	mass=1.0
elif(potential=='WAP'):
	### Weakly Anharmonic Potential
	a = 0.5
	b = 0.1
	c = 0.01
	mass=1.0
	potkey = 'MAP'
elif(potential=='QPper'):
	### Weakly Cubic Potential
	a = 0.5
	b = 0.0
	try:
	    c= float(sys.argv[6])
	    print('Cuartic coefficient is : {}\n'.format(c))
	except:
	    raise ValueError('Please specify cuartic coefficient\n\n')
	mass=1.0
elif(potential=='QP'):
	### Quartic Potential
	a = 0.
	b = 0.
	c = 0.25
	mass=1.0
elif(potential=='OHqtip4pf'):
	### OH stretch from qtip4pf
	alpha=1.21
	a = 0.5
	b = -alpha*0.5           #-0.6
	c = +7./24. *(alpha)**2  #+0.42
	mass=1.0
elif(potential=='WQP'):
	### Weakly Cubic Potential
	a = 0.5
	try:
	    b= float(sys.argv[6])
	    print('Cubic coefficient is : {}\n'.format(b))
	except:
	    raise ValueError('Please specify cubic coefficient\n\n')
	c = b**2
	mass=1.0
elif(potential=='Chaos'):
	### DW Potential
	mass = 0.5
	g    = 0.08
	wb   = 2.0

	a = g * ( -2.0*(-mass*wb**2) / (4*g) )
	b = 0.0
	c = g**2
#elif(potential=='Tanimura_SB'):
#	D = 0.0234
#	alpha = 0.00857



print('Potential parameters = ',a,b,c)
print('mass =',mass)

###================================================================

###================================================================
### define time grid 

time = np.zeros(nstep)
for i in range(nstep):
	time[i] = i*dt - nstep*dt/2. #ALBERTO (negative and positive times)
	#time[i] = i*dt  # only positive times 
###================================================================

###================================================================
### construct the Hamiltonians in the Gauss-Hermite basis

print('Start Hamiltonian construction')

### x matrix
x_mtrx = np.zeros((nb,nb))
for i in range(nb):
	for j in range(nb):
		if i == (j+1):
			x_mtrx[i,j] = np.sqrt(j+1)
		if i == (j-1):
			x_mtrx[i,j] = np.sqrt(j)
x_mtrx *= (2.*mass)**(-0.5) 

### x^2 matrix
x2_mtrx = np.zeros((nb,nb))
for i in range(nb):
	for j in range(nb):
		if i == (j+2):
			x2_mtrx[i,j] = np.sqrt((j+1)*(j+2))
		if i == j:
			x2_mtrx[i,j] = 2*j+1
		if i == (j-2):
			x2_mtrx[i,j] = np.sqrt(j*(j-1))
x2_mtrx *= (2.0*mass)**(-1.)

### p^2 matrix
p2_mtrx = np.zeros((nb,nb))
for i in range(nb):
	for j in range(nb):
		if i == (j+2):
			p2_mtrx[i,j] = -np.sqrt((j+1)*(j+2))
		if i == j:
			p2_mtrx[i,j] = 2*j+1
		if i == (j-2):
			p2_mtrx[i,j] = -np.sqrt(j*(j-1))
p2_mtrx *= (mass/2.0)

### x^3 matrix
x3_mtrx = np.zeros((nb,nb))
for i in range(nb):
	for j in range(nb):
		if i == (j+3):
			x3_mtrx[i,j] = np.sqrt((j+1)*(j+2)*(j+3))
		if i == (j+1):
			x3_mtrx[i,j] = j*np.sqrt(j+1) + (j+1)**(1.5) + (j+2)*np.sqrt(j+1)
		if i == (j-1):
			x3_mtrx[i,j] = (j-1)*np.sqrt(j) + j**(1.5) + (j+1)*np.sqrt(j)
		if i == (j-3):
			x3_mtrx[i,j] = np.sqrt(j*(j-1)*(j-2))
x3_mtrx *= (2.0*mass)**(-1.5)

### x^4 matrix
x4_mtrx = np.zeros((nb,nb))
for i in range(nb):
	for j in range(nb):
		if i == (j+4):
			x4_mtrx[i,j] = np.sqrt((j+1)*(j+2)*(j+3)*(j+4))
		if i == (j+2):
			x4_mtrx[i,j] = j*np.sqrt((j+1)*(j+2)) + np.sqrt(j+2)*(j+1)**(1.5) + np.sqrt(j+1)*(j+2)**(1.5) + np.sqrt((j+1)*(j+2))*(j+3)
		if i == j:
			x4_mtrx[i,j] = j*(j-1) + j*j + 2*j*(j+1) + (j+1)*(j+1) + (j+1)*(j+2)
		if i == (j-2):
			x4_mtrx[i,j] = (j-2)*np.sqrt(j*(j-1)) + np.sqrt(j)*(j-1)**(1.5) + np.sqrt(j-1)*(j)**(1.5) + np.sqrt(j*(j-1))*(j+1)
		if i == (j-4):
			x4_mtrx[i,j] = np.sqrt(j*(j-1)*(j-2)*(j-3))
x4_mtrx *= (2.0*mass)**(-2)

### Hamiltonian
Ham = 0.5 * p2_mtrx + a * x2_mtrx + b * x3_mtrx + c * x4_mtrx

print('End Hamiltonian construction')

###================================================================

###================================================================
### Diagonalization of the Hamiltonians

print('Begin diagonalization')

vals,vecs = linn.eigh(Ham)

print('End diagonalization')
###================================================================

###================================================================
### compute partition functions

Z = 0.0
for i in range(tnb):
	Z += np.exp(-beta*vals[i])

print('Z = ',Z)

for i in range(1,2):
    print(vals[i]/vals[i-1])
if only_freq:
     sys.exit()
###================================================================

###================================================================
### Build the observables matrix in the energy eigen-basis
###================================================================
### The matrix elements must be expanded in HO eigenstates and weighted 
### by the coeficients of the HO eigenstates
### |n> = Sum_j_{0}^{nb} c_j*|j>

print('Begin building of observables matrix')

clist = np.zeros((tnb,tnb+2))
for n in range(tnb):
	for j in range(tnb):	# indexes are inverted
		clist[n,j] = vecs[j,n]

### observable X

x_eig = np.zeros((tnb,tnb))    
for f in range(tnb):
	for g in range(tnb):
		for i in range(tnb):
			x_eig[f,g] += clist[f,i]*clist[g,i-1]*np.sqrt(i) + clist[f,i]*clist[g,i+1]*np.sqrt(i+1)
x_eig *= 2.**(-0.5)

### observable X^2

x2_eig = np.zeros((tnb,tnb))    
for f in range(tnb):
	for g in range(tnb):
		for i in range(tnb):
			x2_eig[f,g] += clist[f,i]*clist[g,i-2]*np.sqrt(i*(i-1)) + clist[f,i]*clist[g,i]*(2*i+1)+clist[f,i]*clist[g,i+2]*np.sqrt((i+1)*(i+2))
x2_eig *= 2.**(-1)

print('End building of observables matrix')
###================================================================

###================================================================
### Define observables

if(A_obs=='x'):
	A = x_eig
elif(A_obs=='x2'):
	A = x2_eig

if(B_obs=='x'):
	B = x_eig
elif(B_obs=='x2'):
	B = x2_eig

if(C_obs=='x'):
	C = x_eig
elif(C_obs=='x2'):
	C = x2_eig

###================================================================
### Evaluate the KT correlation <A;B(t1)>
### Note: Evaluation of KT as Re{<A;B(t1)>}

print('Begin KT')

KT = np.zeros(nstep,dtype=complex)

for t1 in range(nstep):
	print('step ',t1,' of ',nstep,'total')

	for n in range(tnb):	# trace
		for m in range(tnb):
				fct_beta = np.exp(-beta*vals[n]) 
				if (n==m):
					fct_beta *= beta
				else:
					freq = vals[n] - vals[m] 
					fct_beta *= (np.exp(beta*freq) - 1.) / freq

				KT[t1] += fct_beta	\
					  * A[n,m]		\
					  * B[m,n] * np.exp(1.j*time[t1]*(vals[m]-vals[n])) 

KT /= (beta*Z)

### taking Re{KT}

KT = KT.real

print('End KT')
###================================================================

###================================================================
### Evaluate the DKT correlation <A;B(t1);C(t2)>
### Note: Analytic integration of DOuble Kubo integral

if compute_DKT:
	print('Begin DKT')
	DKT   = np.zeros([nstep,nstep],dtype=complex)
	DKTp1  = np.zeros([nstep,nstep],dtype=complex)
	DKTp2  = np.zeros([nstep,nstep],dtype=complex)
	DKTpp = np.zeros([nstep,nstep],dtype=complex)
	DKTpp2 = np.zeros([nstep,nstep],dtype=complex)

	for t1 in range(nstep):
		print('step ',t1,' of ',nstep,'total')
		for t2 in range(nstep):

			for q in range(tnb):
				for r in range(tnb):
					for s in range(tnb):

						if s == r == q:
							fct_beta = np.exp(-beta*vals[q])*beta**2/2. 
						elif r == s:
							delta = (vals[q]-vals[r])
							fct_beta = np.exp(-beta*vals[q])/delta**2 * ( np.exp(beta*delta)*(beta*delta-1.)+1) 
						elif q == s:
							B_ijk = (np.exp(beta*(vals[q]-vals[r]))-1.) / ((vals[q]-vals[r])*(vals[r]-vals[s]))
							fct_beta = np.exp(-beta*vals[q])*(beta/(vals[r]-vals[s]) - B_ijk) 
						elif q == r:
							A_ijk = (np.exp(beta*(vals[q]-vals[s]))-1.) / ((vals[q]-vals[s])*(vals[r]-vals[s]))
							fct_beta = np.exp(-beta*vals[q])*(A_ijk - beta/(vals[r]-vals[s]))  
						else:
							A_ijk = (np.exp(beta*(vals[q]-vals[s]))-1.) / ((vals[q]-vals[s])*(vals[r]-vals[s]))
							B_ijk = (np.exp(beta*(vals[q]-vals[r]))-1.) / ((vals[q]-vals[r])*(vals[r]-vals[s]))
							fct_beta = np.exp(-beta*vals[q])*(A_ijk-B_ijk)           

						DKT[t1,t2] += fct_beta	\
						   * A[q,r]	\
						   * B[r,s] * np.exp(1.j*time[t1]*(vals[r] - vals[s]))	\
						   * C[s,q] * np.exp(1.j*time[t2]*(vals[s] - vals[q]))

						DKTp1[t1,t2] += fct_beta	\
						   * A[q,r] * (1.j) * (vals[q] - vals[r])	\
						   * B[r,s] * np.exp(1.j*time[t1]*(vals[r] - vals[s]))	\
						   * C[s,q] * np.exp(1.j*time[t2]*(vals[s] - vals[q]))
						
						DKTp2[t1,t2] += fct_beta	\
						   * A[q,r]  \
						   * B[r,s] * (1.j)*(vals[r] - vals[s]) * np.exp(1.j*time[t1]*(vals[r] - vals[s]))	\
						   * C[s,q] * np.exp(1.j*time[t2]*(vals[s] - vals[q]))

						DKTpp[t1,t2] += fct_beta	\
						   * A[q,r] * (1.j)*(vals[q] - vals[r])	\
						   * B[r,s] * (1.j)*(vals[r] - vals[s]) * np.exp(1.j*time[t1]*(vals[r] - vals[s]))	\
						   * C[s,q] * np.exp(1.j*time[t2]*(vals[s] - vals[q]))
						
						DKTpp2[t1,t2] += fct_beta	\
						   * A[q,r] * (1.j)*(vals[q] - vals[r])	\
						   * B[r,s] * np.exp(1.j*time[t1]*(vals[r] - vals[s]))	\
						   * C[s,q] * (1.j)*(vals[s] - vals[q]) * np.exp(1.j*time[t2]*(vals[s] - vals[q]))
					

	DKT /= (Z*beta**2)
	DKTp1 /= (Z*beta**2)
	DKTp2 /= (Z*beta**2)
	DKTpp /= (Z*beta**2)
	DKTpp2 /= (Z*beta**2)

	print('End DKT ')

###================================================================


if compute_PB:
	print('Begin KT_PB')

	KT_PB_ABC = np.zeros([nstep,nstep],dtype=complex)
	KT_PB_ABCp = np.zeros([nstep,nstep],dtype=complex)
	KT_PB_CBA = np.zeros([nstep,nstep],dtype=complex)
	KT_PB_CBAp = np.zeros([nstep,nstep],dtype=complex)


	for t1 in range(nstep):
		print('step ',t1,' of ',nstep,'total')
		for t2 in range(nstep):
			for q in range(tnb):
				for r in range(tnb):
					for s in range(tnb):
						freq1 = vals[q] - vals[r] 
						freq2 = vals[r] - vals[s] 
						freq3 = vals[s] - vals[q] 
						freq4 = vals[q] - vals[s] 
						fct_beta = np.exp(-beta*vals[q]) 
						#if (q==r):
						if (q==s):
							fct_beta *= beta
						else:
							#fct_beta *= (np.exp(beta*freq1) - 1.) / (freq1)
							fct_beta *= (np.exp(beta*freq4) - 1.) / (freq4)
						#aux_1 = fct_beta * A[q,r] * B[r,s] * C[s,q] * np.exp(1.j*time[t2]*(freq1)) * np.exp(1.j*time[t1]*(freq2)) 
						#aux_2 = fct_beta * A[q,r] * C[r,s] * B[s,q] * np.exp(1.j*time[t2]*(freq1)) * np.exp(1.j*time[t1]*(freq3)) 
						aux_1a = fct_beta * A[q,r] * B[r,s] * C[s,q] * np.exp(1.j*time[t2]*(freq3)) * np.exp(1.j*time[t1]*(freq2)) 
						aux_1b = fct_beta * A[r,s] * B[q,r] * C[s,q] * np.exp(1.j*time[t2]*(freq3)) * np.exp(1.j*time[t1]*(freq1)) 
						aux_2a = fct_beta * C[q,r] * B[r,s] * A[s,q] * np.exp(1.j*time[t2]*(freq1)) * np.exp(1.j*time[t1]*(freq2)) 
						aux_2b = fct_beta * C[r,s] * B[q,r] * A[s,q] * np.exp(1.j*time[t2]*(freq2)) * np.exp(1.j*time[t1]*(freq1)) 
	
						KT_PB_ABC[t1,t2] += (1.j/hbar)*(aux_1a - aux_1b)
						KT_PB_ABCp[t1,t2] += (1.j/hbar)* (1.j/hbar)*    ( (aux_1a * (vals[q]-vals[r])) - (aux_1b * (vals[r]-vals[s])) )
	
						KT_PB_CBA[t1,t2] += (1.j/hbar)*(aux_2a - aux_2b)
						KT_PB_CBAp[t1,t2] += (1.j/hbar)* (1.j/hbar)*    ( (aux_2a * (vals[s]-vals[q])) - (aux_2b * (vals[s]-vals[q])) )



	KT_PB_ABC /= (beta*Z)
	KT_PB_ABCp /= (beta*Z)
	KT_PB_CBA /= (beta*Z)
	KT_PB_CBAp /= (beta*Z)

	print('End KT_PB_ABC')

###=====================================================================
### Evaluate standard functions (i.e. C_{ABC}) 
if compute_std:
	print('Begin standard TCF')

	C_ABC = np.zeros([nstep,nstep],dtype=complex)
	C_BCA = np.zeros([nstep,nstep],dtype=complex)
	C_CAB = np.zeros([nstep,nstep],dtype=complex)
	C_ACB = np.zeros([nstep,nstep],dtype=complex)
	C_CBA = np.zeros([nstep,nstep],dtype=complex)
	C_BAC = np.zeros([nstep,nstep],dtype=complex)

	for t1 in range(nstep):
		print('step ',t1,' of ',nstep,'total')
		for t2 in range(nstep):
			for q in range(tnb):
				for r in range(tnb):
					for s in range(tnb):
	
						fct_beta = np.exp(-beta*vals[q])
	
						C_ABC[t1,t2] += fct_beta	\
						     * A[q,r]			\
						     * B[r,s] * np.exp(1.j*time[t1]*(vals[r] - vals[s]))	\
						     * C[s,q] * np.exp(1.j*time[t2]*(vals[s] - vals[q]))           

						C_BCA[t1,t2] += fct_beta	\
						     * B[q,r] * np.exp(1.j*time[t1]*(vals[q] - vals[r]))	\
						     * C[r,s] * np.exp(1.j*time[t2]*(vals[r] - vals[s]))	\
						     * A[s,q]	

						C_CAB[t1,t2] += fct_beta	\
						     * C[q,r] * np.exp(1.j*time[t2]*(vals[q] - vals[r]))	\
						     * A[r,s]	\
						     * B[s,q] * np.exp(1.j*time[t1]*(vals[s] - vals[q]))

						C_BAC[t1,t2] += fct_beta	\
						     * B[q,r] * np.exp(1.j*time[t1]*(vals[q] - vals[r]))	\
						     * A[r,s]		\
						     * C[s,q] * np.exp(1.j*time[t2]*(vals[s] - vals[q]))           

						C_ACB[t1,t2] += fct_beta	\
						     * A[q,r]		\
						     * C[r,s] * np.exp(1.j*time[t2]*(vals[r] - vals[s]))    \
						     * B[s,q] * np.exp(1.j*time[t1]*(vals[s] - vals[q]))	

						C_CBA[t1,t2] += fct_beta	\
						     * C[q,r] * np.exp(1.j*time[t2]*(vals[q] - vals[r]))    \
						     * B[r,s] * np.exp(1.j*time[t1]*(vals[r] - vals[s]))	\
						     * A[s,q]		


	C_ABC /= Z
	C_BCA /= Z
	C_CAB /= Z
	C_BAC /= Z
	C_ACB /= Z
	C_CBA /= Z

	print('End standard TCF')
#=====================================================================

#=====================================================================
### save data to file

if compute_PB:
	### KT_PT: order is [time, time, KT_PB.real, KT_PB.imag]
	output= open('KT_PB_ABC.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],KT_PB_ABC.real[i,j],KT_PB_ABC.imag[i,j]))
		output.write('\n')
	output.close()

	output= open('KT_PB_ABCp.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],KT_PB_ABCp.real[i,j],KT_PB_ABCp.imag[i,j]))
		output.write('\n')
	output.close()

	output= open('KT_PB_CBA.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],KT_PB_CBA.real[i,j],KT_PB_CBA.imag[i,j]))
		output.write('\n')
	output.close()

	output= open('KT_PB_CBAp.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],KT_PB_CBAp.real[i,j],KT_PB_CBAp.imag[i,j]))
		output.write('\n')
	output.close()


if compute_DKT:
	### KT: order is [time, KT]
	output= open('KT.dat', 'w')
	for i in range(nstep):
		output.write('{} {} \n'.format(time[i],KT[i]))
	output.close()

	### DKT: order is [time, time, DKT.real, DKT.imag]
	output= open('DKT.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],DKT.real[i,j],DKT.imag[i,j]))
		output.write('\n')
	output.close()

	output= open('DKTp1.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],DKTp1.real[i,j],DKTp1.imag[i,j]))
		output.write('\n')
	output.close()
	output= open('DKTp2.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],DKTp2.real[i,j],DKTp2.imag[i,j]))
		output.write('\n')
	output.close()

	output= open('DKTpp.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],DKTpp.real[i,j],DKTpp.imag[i,j]))
		output.write('\n')
	output.close()

	output= open('DKTpp2.dat', 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j],DKTpp2.real[i,j],DKTpp2.imag[i,j]))
		output.write('\n')
	output.close()

if compute_std:

	### C_ABC: order is [time, time, C.real, C.imag]
	output= open('C_ABC_{}_beta_{}.dat'.format(potkey,beta), 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j], C_ABC.real[i,j],C_ABC.imag[i,j]))
		output.write('\n')
	output.close()

	### C_BCA: order is [time, time, C.real, C.imag]
	output= open('C_BCA_{}_beta_{}.dat'.format(potkey,beta), 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j], C_BCA.real[i,j],C_BCA.imag[i,j]))
	output.close()

	### C_CAB: order is [time, time, C.real, C.imag]
	output= open('C_CAB_{}_beta_{}.dat'.format(potkey,beta), 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j], C_CAB.real[i,j],C_CAB.imag[i,j]))
		output.write('\n')
	output.close()

	### C_ACB: order is [time, time, C.real, C.imag]
	output= open('C_ACB_{}_beta_{}.dat'.format(potkey,beta), 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j], C_ACB.real[i,j],C_ACB.imag[i,j]))
		output.write('\n')
	output.close()

	### C_CBA: order is [time, time, C.real, C.imag]
	output= open('C_CBA_{}_beta_{}.dat'.format(potkey,beta), 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j], C_CBA.real[i,j],C_CBA.imag[i,j]))
		output.write('\n')
	output.close()

	### C_BAC: order is [time, time, C.real, C.imag]
	output= open('C_BAC_{}_beta_{}.dat'.format(potkey,beta), 'w')
	for i in range(nstep):
		for j in range(nstep):
			output.write('{} {} {} {} \n'.format(time[i],time[j], C_BAC.real[i,j],C_BAC.imag[i,j]))
		output.write('\n')
	output.close()

	#=====================================================================
	### End program
t_final=get_time.time()
print('DONE in {}s!!'.format(t_final-t_init))

