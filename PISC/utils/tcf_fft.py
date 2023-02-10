import numpy as np
from PISC.utils import tcf_fort_tools, tcf_fort_tools_omp

def gen_tcf(Aarr,Barr,tarr,corraxis=None):
	q1arr=np.array(Aarr)
	q1arr[(len(tarr)//2):] = 0.0	
	q1_tilde = np.fft.rfft(q1arr,axis=0)
	q2_tilde = np.fft.rfft(Barr,axis=0)
	
	tcf = np.fft.irfft(np.conj(q1_tilde)*q2_tilde,axis=0)
	tcf = tcf[:len(tcf)//2,:,:]  #Truncating the padded part
	if(corraxis is None):
		tcf = np.sum(tcf,axis=2)  #Summing over the dimension (Dot product)
	else:
		tcf = tcf[:,:,corraxis]
	tcf = np.mean(tcf,axis=1) #Averaging over the particles
	   
	tcf/=(len(tcf))
	tarr = tarr[:len(tcf)]
	return tarr,tcf

def gen_2pt_tcf(dt,tarr, Carr, Barr, Aarr=None,dt_tcf = 0.1,trans_sym=False):
	# The ordering is C[t2]B[t1]A[t0]. If A is not provided
	# then it is C[t2]B[t1]. If trans_sym is false, t0 is set to 0
	# by default, as it a 2-point TCF.
	
	# Rewrite this in FORTRAN, this is extremely slow. 
	
	if(trans_sym):
			#Use symmetry w.r.t translation about t0	
			halflen = len(tarr)//2
			tcf = np.zeros((halflen,halflen))	
			for k in range(halflen):
				for i in range(halflen): #t1 axis
					for j in range(halflen): #t2 axis
						#Dot product and Ensemble average in the same order.
						tcf[i,j] += np.mean(np.sum(Carr[j+k]*Barr[i+k]*Aarr[k],axis=1),axis=0) 	
			tcf/=halflen # To normalise contributions from all t1 translated tcfs.						
			tcf = tcf[:halflen]
			tarr = tarr[:halflen]
			return tarr, tcf  		

	else:
		stride=1
		if(dt<dt_tcf):
			stride = int(dt_tcf//dt)	
		tar = tarr[::stride]
		Car = Carr[::stride]
		Bar = Barr[::stride]
		if(Aarr is not None):
			Aar = Aarr[::stride]
		tlen = len(tar)	
		tcf = np.zeros((tlen,tlen))

		if(1): #FORTRAN	
			tcf_fort = np.ascontiguousarray(tcf)
			Bar = np.asfortranarray(Bar)
			Car = np.asfortranarray(Car)
			if(Aarr is not None):
				Aar = np.asfortranarray(Aar)
				tcf = tcf_fort_tools_omp.tcf_tools.two_pt_3op_tcf(Aar,Bar,Car,tcf_fort)
			else:
				tcf = tcf_fort_tools_omp.tcf_tools.two_pt_2op_tcf(Bar,Car,tcf_fort)
		if(0): #PYTHON
			tcf[:] = 0.0
			for i in range(tlen): #t1 axis
				for j in range(tlen): #t2 axis
					#Dot product and Ensemble average in the same order.
					if Aarr is None:
						tcf[i,j] = np.mean(np.sum(Car[j]*Bar[i],axis=1),axis=0) 	
					else:
						tcf[i,j] = np.mean(np.sum(Car[j]*Bar[i]*Aar[0],axis=1),axis=0)
					
		return tar, tcf	

