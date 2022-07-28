import numpy as np

def gen_tcf(Aarr,Barr,tarr):
	q1arr=np.array(Aarr)
	q1arr[(len(tarr)//2):] = 0.0	
	q1_tilde = np.fft.rfft(q1arr,axis=0)
	q2_tilde = np.fft.rfft(Barr,axis=0)
	
	tcf = np.fft.irfft(np.conj(q1_tilde)*q2_tilde,axis=0)
	tcf = tcf[:len(tcf)//2,:,:]  #Truncating the padded part
	tcf = np.sum(tcf,axis=2)  #Summing over the dimension (Dot product)
	tcf = np.mean(tcf,axis=1) #Averaging over the particles
	   
	tcf/=(len(tcf))
	tarr = tarr[:len(tcf)]
	return tarr,tcf 
