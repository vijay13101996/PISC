fig.savefig('/home/vgs23/Images/bn_vs_n_1DB_1panel.pdf', dpi=400, bbox_inches='tight',pad_inches=0.00
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

#We want to plot a matrix as a 2D grid of colored squares

N = 300
matrix = np.zeros((N, N))


def mat_comp(N, func):
    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[i, j] = 1
            else:
                if func=='pl':
                    matrix[i, j] = 1e3/abs(i-j)**2 #
                elif func=='exp':
                    matrix[i, j] = 1e3*np.exp(-abs(i-j))
                elif func=='gauss':
                    matrix[i, j]  = 1e3*np.exp(-0.5*abs(i-j)**2)
    return matrix

def funct(funckey):
    xarr = np.linspace(1, N, 100)
    yarr = np.zeros(100)
    if funckey=='pl':
        return xarr, 1e3/xarr**2
    elif funckey=='exp':
        return xarr, 1e3*np.exp(-xarr)
    elif funckey=='gauss':
        return xarr, 1e3*np.exp(-1e-4*xarr**2)

def Lanczos(funckey):
    xarr = np.linspace(1, N, 100)
    yarr = np.zeros(100)
    if funckey=='pl':
        return xarr, xarr*np.pi*0.99
    elif funckey=='exp':
        return xarr, xarr*np.pi/2
    elif funckey=='gauss':
        return xarr, 10*xarr**0.5

fig,ax = plt.subplots(2,3, figsize=(10,7), constrained_layout=True)
fig.subplots_adjust(wspace=0.1, hspace=0.1)

axes = ax[0]

for a,func in zip(axes, ['pl', 'exp', 'gauss']):
    
    matrix = mat_comp(N, func)
    a.imshow((matrix), vmax=1, cmap='YlGn')  # 'magma')
    #a.set_title(r'$O_{ij} \sim$' + func, fontsize=tp_fs)
    a.set_xlabel(r'$j$', fontsize=xl_fs)
    
    #Set y label only for first plot
    if func == 'pl':
        a.set_ylabel(r'$i$', fontsize=yl_fs)
    a.xaxis.set_major_locator(MultipleLocator(20))  # every 2 units on x-axis
    a.yaxis.set_major_locator(MultipleLocator(20))  # every 0.5 units on y-axis
    a.grid(True, color='k', linewidth=0.5)

    #plot dashed line along diagonal
    a.plot([0, N-1], [0, N-1], color='white', linestyle='--', linewidth=1)

    a.set_xticklabels([])
    a.set_yticklabels([])

    a.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

axes = ax[0]

for axe, funckey in zip(axes, ['pl', 'exp', 'gauss']):
    
    # Create an inset axis within the current axis
    ax_inset = axe.inset_axes([0.65, 0.65, 0.3, 0.3])  # [x0, y0, width, height]

    a = ax_inset

    xarr, yarr = funct(funckey)
    a.plot(xarr, np.log(yarr)/np.log(yarr)[0], color='b', linewidth=2)
    a.set_xlabel(r'$|E_i-E_j|$', fontsize=xl_fs-5, labelpad=5)
    a.set_ylabel(r'$\log O_{ij}$', fontsize=yl_fs-5, labelpad=5)
    # Make fonts bolder
    a.xaxis.label.set_fontweight('bold')
    a.yaxis.label.set_fontweight('bold')


    #bring xlabel and ylable closer to axis
    a.xaxis.set_label_coords(0.5,-0.08)
    a.yaxis.set_label_coords(-0.05,0.5)
    
    white_bg = Rectangle(
        (-0.35, -0.3), 1.4, 1.4, transform=a.transAxes, 
                     facecolor='white', edgecolor='black', 
                     linewidth=1, zorder=-1, clip_on=False)
    a.add_patch(white_bg)
    
    # Keep only bottom and left spines (x and y axes)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    #set x limit to 0-300
    a.set_xlim(0, 300)
    a.set_yticks([1])
    a.set_xticks([])
    a.set_xticklabels([])
    a.set_yticklabels(['1'])

    #Arrows on both axes
    a.annotate('', xy=(1, 0), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    a.annotate('', xy=(0, 1), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


axes = ax[1]

for a,func in zip(axes, ['pl', 'exp', 'gauss']):
    
    a.set_xlabel(r'$n$', fontsize=xl_fs)
    
    #Set y label only for first plot
    if func == 'pl':
        a.set_ylabel(r'$b_n$', fontsize=yl_fs)
    a.xaxis.set_major_locator(MultipleLocator(50))  # every 2 units on x-axis
    a.yaxis.set_major_locator(MultipleLocator(500))  # every 0.5 units on y-axis

    xarr, yarr = Lanczos(func)
    a.scatter(xarr, yarr, color='k', s=10)
    a.set_xlim(0, 300)
    #a.set_ylim(0, 3000)

    # Make fonts bolder
    a.xaxis.label.set_fontweight('bold')
    a.yaxis.label.set_fontweight('bold')

    #plot dashed line along y= x*np.pi
    a.plot(xarr[:N//5], xarr[:N//5]*np.pi, color='red', linestyle='--', linewidth=2)

    a.set_xticklabels([])
    a.set_yticklabels([])

    a.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    # Annotate with \alpha = \pi/\beta along the dashed line
    a.text(60, 250, r'$\alpha = \pi/\beta$', color='k', fontsize=tp_fs, rotation=55)


fig.savefig('Matrix_illustration.pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()




