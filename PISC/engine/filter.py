import numpy as np
from matplotlib import pyplot as plt
def cross_filt(rp,rp0):#works because it is true for all time if it is true for one time, not applicable to radius of gyration thoug
	q0 = rp0.q[:,0,0]
	qt = rp.q[:,0,0]
	ind = np.where(q0*qt < 0.0)
	return ind	
class Rg_Filter():
    def __init__(self,rg_lower,rg_upper,N,nthermsteps,Hist=False):
        self.max_gyr=np.zeros(N)
        self.N=N
        self.rg_lower=rg_lower
        self.rg_upper=rg_upper
        self.nthermsteps=nthermsteps
        self.hist=Hist
        self.i=0#thermsteps counter for histogram
        if(Hist==True):
            self.all_gyr=np.zeros((N,self.nthermsteps))#only_for hist of all
            self.all_max_gyr=[]

    def calculate_Rg(self,rp):
        cent=rp.q[:,:,0]/rp.nbeads**0.5
        tmp_gyr=0
        for j in range(rp.nbeads):
            tmp_gyr = np.sum(((cent[:,:]-rp.qcart[:,:,j])**2),axis=1)
        tmp_gyr /=rp.nbeads
        return tmp_gyr

    def filter(self,rp):
        Rg=self.calculate_Rg(rp)
        if(self.hist==True):
            self.all_gyr[:,self.i]=Rg[:]#only for hist
            self.i+=1
        for j in range(len(Rg)):
            if(Rg[j]>self.max_gyr[j]):
                self.max_gyr[j]=Rg[j]
    def index_array(self):
        logic1 = self.max_gyr<self.rg_upper
        logic2 = self.max_gyr>self.rg_lower
        logic_filter = np.logical_and(logic1, logic2)
        print(np.sum(logic_filter),self.N)
        if(self.hist==True):
            self.all_max_gyr.append(self.max_gyr)
            self.plot_all_max_hist()
            #self.plot_hist()
            #self.plot_max_hist()
        return logic_filter

    def plot_hist(self,bins=400, range=(0,0.2)):
        plt.hist(self.all_gyr.flat,bins=bins, range=range)
        plt.show()

    def plot_max_hist(self,bins=50,range=(0,0.5)):
        print('(min_max, mean_max, max_max)(rg): ',np.min(self.max_gyr),np.mean(self.max_gyr),np.max(self.max_gyr))
        plt.hist(self.max_gyr,bins=bins,range=range)
        plt.show()
    def plot_all_max_hist(self,bins=50,range=()):
        print('(min_max, mean_max, max_max)(rg): ',np.min(self.all_max_gyr),np.mean(self.all_max_gyr),np.max(self.all_max_gyr))
        plt.hist(self.all_max_gyr,bins=bins)#,range=range)
        print(len(self.all_max_gyr))
        plt.show()