import numpy as np


nmat = 100
rng = np.random.default_rng(0)
A = rng.standard_normal((nmat, nmat)) + 1j * rng.standard_normal((nmat, nmat))
if(0):
    A = A + A.conjugate().T  # Make A Hermitian
    #Set diagonal of A to zero
    np.fill_diagonal(A, 0.0)
Aadj = A.conjugate().T   # Adjoint of A

p1 = np.zeros(nmat, dtype=np.complex128)
p1[0] = 1.0 + 0.0j

q1 = np.zeros(nmat, dtype=np.complex128)
q1[0] = 1.0 + 0.0j

def bilanczos(A, p0, q0, ncoeffs):
    """ Perform the bi-Lanczos algorithm to compute the tridiagonalization of A. """
    
    P = np.zeros((nmat, ncoeffs), dtype=np.complex128)  
    Q = np.zeros((nmat, ncoeffs), dtype=np.complex128)

    alphas = np.zeros(ncoeffs, dtype=np.complex128)
    betas = np.zeros(ncoeffs-1, dtype=np.complex128)
    gammas = np.zeros(ncoeffs-1, dtype=np.complex128)

    r = A @ q1    # For Liouvillian, this becomes L*O (Hadamard product)
    s = Aadj @ p1
    
    P[:, 0] = p1    # For operators, we can output specific basis vectors to verify biorthogonality
    Q[:, 0] = q1
    
    pj = p1
    qj = q1

    for j in range(ncoeffs-1):
        alphaj = np.vdot(pj, r)   # Inner product replaces this step for operators
        alphas[j] = alphaj

        r = r - alphaj * qj
        s = s - np.conjugate(alphaj) * pj

        if (np.linalg.norm(r) < 1e-10 or np.linalg.norm(s) < 1e-10):
            break
        wj = np.vdot(r, s)
        if abs(wj) < 1e-10:
            break
        betaj1 = np.sqrt(abs(wj))
        gammaj1 = np.conjugate(wj) / betaj1
        betas[j] = betaj1
        gammas[j] = gammaj1

        qj1 = r / betaj1
        pj1 = s / np.conj(gammaj1)
        
        P[:, j+1] = pj1
        Q[:, j+1] = qj1

        r = A @ qj1 - gammaj1 * qj
        s = Aadj @ pj1 - np.conj(betaj1) * pj

        pj = pj1 
        qj = qj1

    alphaj = np.vdot(pj, r)
    alphas[ncoeffs-1] = alphaj

    return alphas, betas, gammas, P, Q

ncoeffs = 20
alphas, betas, gammas, P, Q = bilanczos(A, p1, q1, ncoeffs)

print("Alphas:", alphas)
print("Betas:", betas)
print("Gammas:", gammas)

# Reconstruct tridiagonal matrix T
T = np.zeros((ncoeffs, ncoeffs), dtype=np.complex128)
for i in range(ncoeffs):
    T[i, i] = alphas[i]
    if i < ncoeffs - 1:
        T[i+1, i] = betas[i]
        T[i, i+1] = gammas[i]
PQT = np.conjugate(P.T) @ A @ Q

#Check that P^H A Q = T upto tol=1e-10
tol = 1e-10
if np.allclose(PQT, T, atol=tol):
    print("P^H A Q matches T within tolerance.")
else:
    print("P^H A Q does not match T within tolerance.")

#Check Biorthogonality of P and Q
PtQ = np.conjugate(P.T) @ Q
QtP = np.conjugate(Q.T) @ P
if np.allclose(PtQ, np.eye(ncoeffs), atol=tol):
    print("P and Q are biorthogonal: P^H Q = I within tolerance.")


#print("Reconstructed T from P^H A Q:\n", np.around(PQT[10:20,10:20], 2))
#print("Directly constructed T:\n", np.around(T[10:20,10:20], 2))
