import numpy as np
from matplotlib import pyplot as plt
from PISC.utils.readwrite import read_1D_plotdata

def plot_1D(ax,fname,label=None,color='k',style ='-',linewidth=1.5,log=False):
	if label is not None:
		label=label
	else:
		label=fname
	
	data = read_1D_plotdata('{}.txt'.format(fname))
	x = data[:,0]
	y = data[:,1]
	if(log is True):
		ax.plot(x,np.log(y),style,color=color,linewidth=linewidth,label=label)
	else:
		ax.plot(x,y,style,color=color,linewidth=linewidth,label=label)
	

	
