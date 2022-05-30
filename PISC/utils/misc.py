import numpy as np
from PISC.utils.readwrite import read_1D_plotdata
import os
#import execnet

def pairwise_swap(a,l):
	tempodd = a[...,1:l:2].copy()
	tempeven = a[...,2:l:2].copy()
	temp = a.copy()
	temp[...,1:l:2] = tempeven
	temp[...,2:l:2] = tempodd
	return temp[...,:l]

def find_OTOC_slope(fname,tst,tend):
	data = read_1D_plotdata('{}.txt'.format(fname))
	t_arr = data[:,0]
	OTOC_arr = np.log(abs(data[:,1]))

	ist = (np.abs(t_arr - tst)).argmin()
	iend = (np.abs(t_arr - tend)).argmin()

	x_trunc = t_arr[ist:iend]
	y_trunc = OTOC_arr[ist:iend]
	slope,ic = np.polyfit(x_trunc,y_trunc,1)
	print('slope',slope)

	return slope,ic,x_trunc,y_trunc

def seed_collector(kwlist,datapath,tarr,Carr,allseeds=True,seedcount=None):
	flist = []
	print('kwlist',kwlist)
	for fname in os.listdir(datapath):
		if all(kw in fname for kw in kwlist):
			flist.append(fname)

	count=0
	#flist = flist[:500]
	for f in flist:
		data = read_1D_plotdata('{}/{}'.format(datapath,f))
		tarr = data[:,0]
		Carr += data[:,1]
		count+=1

	print('count',count)
	Carr/=count
	return tarr,Carr
	
def call_python_version(Version, Path, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python{}//chdir={}".format(Version,Path))
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    ret = channel.receive()  
