import numpy as np
import execnet

def pairwise_swap(a,l):
	tempodd = a[...,1:l:2].copy()
	tempeven = a[...,2:l:2].copy()
	temp = a.copy()
	temp[...,1:l:2] = tempeven
	temp[...,2:l:2] = tempodd
	return temp[...,:l]

def find_slope(x,y,ist,iend):
	x_trunc = x[ist:iend]
	y_trunc = y[ist:iend]
	slope,ic = np.polyfit(x_trunc,y_trunc,1)
	print('slope',slope)

	return slope,ic,x_trunc,y_trunc

def call_python_version(Version, Path, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python{}//chdir={}".format(Version,Path))
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    ret = channel.receive()  
