import numpy as np
import pickle

def store_1D_plotdata(x,y,fname,fpath=None,ebar=None):
	if(ebar is None):
		data = np.column_stack([x,y])
	else:
		data = np.column_stack([x,y,ebar])

	if(fpath is None):
		datafile_path = "/home/vgs23/Pickle_files/{}.txt".format(fname)
	else:
		datafile_path = "{}/{}.txt".format(fpath,fname)
	np.savetxt(datafile_path , data)#,fmt = ['%f','%f'])

def read_1D_plotdata(fname):
	data = np.loadtxt("{}".format(fname),dtype=complex)
	return data

def chunks(L, n): 
	return [L[x: x+n] for x in range(0, len(L), n)]

def store_arr(arr,fname,fpath=None):
	if(fpath is None):
		f = open('/home/vgs23/Pickle_files/{}.dat'.format(fname),'wb')
	else:
		f = open('{}/{}.dat'.format(fpath,fname),'wb')
	pickle.dump(arr,f)	
                        
def read_arr(fname,fpath=None):
	if fpath is None:
		f = open('/home/vgs23/Pickle_files/{}.dat'.format(fname),'rb')
		arr = pickle.load(f)
	else:
		f = open('{}/{}.dat'.format(fpath,fname),'rb')
		arr = pickle.load(f)
	return arr	
     
