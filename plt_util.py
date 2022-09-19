from matplotlib import pyplot as plt 
import matplotlib


font = {'family':'serif','size':18}

std_width = 20
std_height = 12
std_dpi = 150
file_dpi = 1200

in_in_cm = 0.393701
def prepare_fig(width=std_width, height=std_height,tex=False):
    fig_width = width*in_in_cm
    fig_height = height*in_in_cm
    matplotlib.rc('font', **font)
    if tex:
        matplotlib.rc('text', usetex=True)
    fig = plt.figure(num=None, figsize=(fig_width, fig_height),dpi=std_dpi, facecolor='w', edgecolor='k')
    return fig
def prepare_fig_ax(dim_tuple=None,dim=1,width=std_width, height=std_height,tex=False,share_x=False,share_y=False):
	fig_width = width*in_in_cm
	fig_height = height*in_in_cm
	matplotlib.rc('font', **font)
	if tex:
		matplotlib.rc('text', usetex=True)
	if(dim_tuple is not None):
		fig,ax = plt.subplots(dim_tuple[0],dim_tuple[1],sharex=share_x,sharey=share_y, figsize=(fig_width, fig_height),dpi=std_dpi, facecolor='w', edgecolor='k')	
	else:    
		fig,ax = plt.subplots(1,dim,sharex=share_x,sharey=share_y, figsize=(fig_width, fig_height),dpi=std_dpi, facecolor='w', edgecolor='k')

	return fig,ax
