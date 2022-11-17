import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.readwrite import read_1D_plotdata

def plot_1D(ax,fname,label=None,color='k',style ='-',linewidth=1.5,log=False,magnify=1.0,plot_error=False):
	if label is not None:
		label=label
	else:
		label=fname
	
	data = read_1D_plotdata('{}.txt'.format(fname))
	x = data[:,0]
	y = magnify*data[:,1]
	if(plot_error):
		stdarr = data[:,2]
	if(log):
		ax.plot(x,np.log(y),style,color=color,linewidth=linewidth,label=label)
		if(plot_error):
			ax.errorbar(x,np.log(y),yerr=stdarr/2,ecolor='m',errorevery=100,capsize=2.0)
	else:
		ax.plot(x,y,style,color=color,linewidth=linewidth,label=label)
		if(plot_error):	
			ax.errorbar(x,y,yerr=stdarr/2,ecolor='m',errorevery=100,capsize=2.0)
		


	
