import numpy as np

def proj(U,V,axg):
    return (np.sum(U*V,axis=axg)/np.sum(U*U,axis=axg))

def norm(U,axg):
    return np.sum(U*U,axis=axg)**0.5

def Gram_Schmidt(T):
	i=0
	j=0
	N = len(T.T)
	T_gram_prim = T.copy()
	T_gram = np.zeros_like(T)
	while(i<N):
		j=0
		while(j<i):
			T_gram_prim[...,i]-=  proj(T_gram_prim[...,j],T[...,i],1)[:,None]*T_gram_prim[...,j]
			j+=1
		T_gram[...,i]=T_gram_prim[...,i]/norm(T_gram_prim[...,i],1)[:,None]  
		i+=1

	return T_gram


