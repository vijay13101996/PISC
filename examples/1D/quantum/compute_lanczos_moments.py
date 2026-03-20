import numpy as np


#Give the moments of the correlation function, compute the Lanczos coefficients

def compute_Dn(mu_arr,n):
    # Assumes that all the moments are given
    n_mom = len(mu_arr)

    Dn = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
                Dn[i,j] = mu_arr[i+j]

    return np.linalg.det(Dn)

def compute_Lanczos_det(mu_arr):
    # Assumes that all the moments are given

    n_mom = len(mu_arr)
    D_nm2 = 1
    D_nm1 = mu_arr[0]

    ncoeff = 0

    if (n_mom%2 == 0):
        ncoeff = n_mom//2
    else:
        ncoeff = (n_mom-1)//2
    

    bn_arr = np.zeros(ncoeff)
    bn_arr[0] = mu_arr[0]**0.5

    for n in range(1,ncoeff):
        Dn = compute_Dn(mu_arr,n)
        #print('n,Dn',n,Dn)
        bn_arr[n] = (Dn*D_nm2/D_nm1**2)**0.5
        D_nm2 = D_nm1
        D_nm1 = Dn
        #print('n,bn',n,bn_arr[n])
    return bn_arr

def compute_Lanczos_iter(mu_arr):
    # Assumes that the even moments are given

    neven_mom = len(mu_arr)
    ncoeff = neven_mom

    bn_arr = np.zeros(ncoeff)

    bn_arr[0] = mu_arr[0]**0.5

    print('n',0,bn_arr[0])
    for n in range(1,ncoeff):
        j=1    
        mu_jm2 = 0.0
        mu_jm1 = mu_arr[n]

        while(j<=n):
            if(j>1):
                b_jm2 = bn_arr[j-2]
                b_jm1 = bn_arr[j-1]
            elif(j==1):
                b_jm2 = 1.0
                b_jm1 = bn_arr[0]
            
            mu_j = mu_jm1/b_jm1**2 - mu_jm2/b_jm2**2
            mu_jm2 = mu_jm1
            mu_jm1 = mu_j
            #print('j',j, mu_jm1/b_jm1**2,mu_jm2/b_jm2**2)
            j+=1


        if(mu_j>1e-12):
            bn_arr[n] = mu_j**0.5
        else:
            print('returning',n,mu_j)
            return bn_arr

        print('n',n,bn_arr[n])

    return bn_arr

        




    
