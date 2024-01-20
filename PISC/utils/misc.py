import numpy as np
from PISC.utils.readwrite import read_1D_plotdata, read_2D_imagedata
import os
from pathlib import Path
from scipy.odr import *
from scipy.optimize import *
from PISC.utils.misc_f import misc 

def pairwise_swap(a,l):
    tempodd = a[...,1:l:2].copy()
    tempeven = a[...,2:l:2].copy()
    temp = a.copy()
    temp[...,1:l:2] = tempeven
    temp[...,2:l:2] = tempodd
    return temp[...,:l]

def hess_compress(arr,rp):
    return arr.reshape(-1,rp.ndim*rp.nbeads,\
        rp.ndim*rp.nbeads)

def hess_expand(arr,rp):
    return arr.reshape(-1,rp.ndim,rp.nbeads,rp.ndim,rp.nbeads)

def hess_mul(ddpot,arr_i,arr_o,rp,dt,fort=False):
    hess = hess_compress(ddpot,rp)
    arr_in = hess_compress(arr_i,rp)
    arr_out = hess_compress(arr_o,rp)
    if(fort):
        misc.hess_mul(hess.T,arr_in.T,arr_out.T,dt)
    else:
        arr_out-=np.matmul(hess,arr_in)*dt
    arr_o[:] = hess_expand(arr_out,rp) 

def linear(x,m,c):
    return m*x + c
    
def find_OTOC_slope(fname,tst,tend,witherror=False,return_cov=False):
    data = read_1D_plotdata('{}.txt'.format(fname))
    t_arr = np.real(data[:,0])
    OTOC_arr = np.log(abs(data[:,1]))
            
    ist = (np.abs(t_arr - tst)).argmin()
    iend = (np.abs(t_arr - tend)).argmin()

    x_trunc = t_arr[ist:iend]
    y_trunc = OTOC_arr[ist:iend]
    p,V = np.polyfit(x_trunc,y_trunc,1,cov=True)
    slope, ic = p
    print('slope, ic, cov',slope,ic,V[0,0]**0.5,V[1,1]**0.5)
    if(witherror):
        stdarr = np.real(data[:,2])
        yerr_trunc = stdarr[ist:iend]
        if(0):
            popt,pcov =curve_fit(linear, x_trunc, y_trunc, sigma=yerr_trunc, absolute_sigma=True)       
            slope,ic = popt
            print('slope with scipy', slope)
            print('covariance', pcov[0,0]**0.5, pcov[1,1]**0.5)
        if(1):
            p,V = np.polyfit(x_trunc,y_trunc,1,w=1/yerr_trunc,cov='unscaled')
            slope, ic = p
            print('slope with polyfit',slope)
            print('covariance',V[0,0]**0.5,V[1,1]**0.5)

    if return_cov:
        return slope,ic,x_trunc,y_trunc,V
    else:
        return slope,ic,x_trunc,y_trunc

def estimate_OTOC_slope(kwlist,datapath,tarr,Carr,tst,tend,allseeds=True,seedcount=None,logerr=True):
    flist = []
    print('kwlist',kwlist)
    for fname in os.listdir(datapath):
        if all(kw in fname for kw in kwlist):
            #print('f',fname)
            flist.append(fname)

    count=0
    if(allseeds is False):  
        flist=flist[:seedcount]
    slope_arr = []

    for f in flist:
        data = read_1D_plotdata('{}/{}'.format(datapath,f))
        t_arr = np.real(data[:,0])
        OTOC_arr = np.log(abs(data[:,1]))
        ist = (np.abs(t_arr - tst)).argmin()
        iend = (np.abs(t_arr - tend)).argmin()

        x_trunc = t_arr[ist:iend]
        y_trunc = OTOC_arr[ist:iend]
        slope,ic = np.polyfit(x_trunc,y_trunc,1)    
        slope_arr.append(slope)
        count+=1

    mean = np.mean(slope_arr)
    stderr = np.std(slope_arr)/np.sqrt(len(slope_arr))
    print('mean, std error', mean, stderr)

def seed_collector(kwlist,datapath,tarr,Carr,allseeds=True,seedcount=None,logerr=True,exclude=[]):
    flist = []
    print('kwlist',kwlist)
    for fname in os.listdir(datapath):
        if all(kw in fname for kw in kwlist) and not any(ex in fname for ex in exclude):
            #print('f',fname)
            flist.append(fname)

    count=0
    if(allseeds is False):  
        flist=flist[:seedcount]
    Carr_stack = [] 
    for f in flist:
        data = read_1D_plotdata('{}/{}'.format(datapath,f))
        tarr = data[:,0]
        Carr += data[:,1]
        Carr_stack.append(data[:,1])
        count+=1

    Carr_stack = np.array(Carr_stack)
    mean = np.mean(Carr_stack,axis=0)
    if(logerr):
        stderr = np.std(np.log(Carr_stack),axis=0)#/np.sqrt(len(Carr_stack))
    else:
        stderr = np.std(Carr_stack,axis=0)#/np.sqrt(len(Carr_stack))
    print('count',count)
    Carr[:] = mean
    return tarr,Carr,stderr

def seed_collector_imagedata(kwlist,datapath,allseeds=True,seedcount=None):
    flist = []
    print('kwlist',kwlist)
    for fname in os.listdir(datapath):
        if all(kw in fname for kw in kwlist):
            #print('f',fname)
            flist.append(fname)

    count=0
    if(allseeds is False):  
        flist=flist[:seedcount]
    
    Carr_stack = []
    X,Y,F = read_2D_imagedata('{}/{}'.format(datapath,flist[0]))
    Carr = F.copy()
    Carr_stack.append(F)
    count+=1

    for f in flist[1:]:
        X,Y,F = read_2D_imagedata('{}/{}'.format(datapath,f))
        Carr += F
        Carr_stack.append(F)
        count+=1

    Carr_stack = np.array(Carr_stack)
    mean = np.mean(Carr_stack,axis=0)
    print('count',count)
    F[:] = mean
    return X,Y,F


def seed_finder(kwlist,datapath,allseeds=True,sort=True,dropext=False):
    flist = []
    print('kwlist',kwlist)
    count=0
    for fname in os.listdir(datapath):
        if all(kw in fname for kw in kwlist):
            if(dropext is True):
                fname= os.path.splitext(fname)[0]
                flist.append(fname)
            else:
                flist.append(fname)
            count+=1
    print('count',count)    
    if(sort is True):
        flist.sort()
        return flist
    else:
        return flist
    
def call_python_version(Version, Path, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python{}//chdir={}".format(Version,Path))
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    ret = channel.receive()  
