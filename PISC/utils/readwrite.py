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

def store_2D_imagedata(X,Y,F,fname,fpath=None):
	if(len(np.array(X).shape) == 1 and len(np.array(Y).shape) ==1):
		x,y = np.meshgrid(X,Y)
	else:
		x,y = X,Y

	if(fpath is None):
		datafile_path = "/home/vgs23/Pickle_files/{}.txt".format(fname)
	else:
		datafile_path = "{}/{}.txt".format(fpath,fname)
	
	with open(datafile_path, 'w') as outfile:
		for data,qual in zip([x,y,F],['x','y','f']):
			outfile.write('#{}\n'.format(qual))
			np.savetxt(outfile, data)
			
def read_2D_imagedata(fname):
	data = np.loadtxt(fname)
	X = data[:len(data)//3]
	Y = data[len(data)//3:2*len(data)//3]
	F = data[2*len(data)//3:]
	return X,Y,F
			
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
     
