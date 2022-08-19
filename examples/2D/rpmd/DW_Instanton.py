import numpy as np
from PISC.engine.instools import find_instanton, find_extrema, inst_init, inst_double
from PISC.utils.readwrite import store_arr

def find_instanton_DW(nbeads,m,pes,beta,ax=None,plt=None,plot=False,step=1e-4,tol=1e-2,nb_start=4,qinit=None,path=None):
	print('Starting')
	sp=np.array([0.0,0.0])
	eigvec = np.array([1.0,0.0])

	nb = nb_start
	if(qinit is None):
		qinit = inst_init(sp,0.1,eigvec,nb)	
	while nb<=nbeads:
		instanton = find_instanton(m,pes,qinit,beta,nb,dim=2,scale=1.0,stepsize=step,tol=tol)
		if(path is not None):
			store_arr(instanton,'Instanton_beta_{}_nbeads_{}_z_{}'.format(beta,nb,pes.z),'{}/Datafiles/'.format(path))
		print('Instanton config. with nbeads=', nb, 'computed')	
		if(ax is not None):
			ax.scatter(instanton[0,0],instanton[0,1],label='nbeads = {}'.format(nb))	
			plt.pause(0.01)
		qinit=inst_double(instanton)
		nb*=2
	print('Exiting')
	if(plot):
		plt.show()	
	return instanton