import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import Krylov_complexity_harm_draft1 as fig2a_f
import Krylov_complexity_Landau as fig2b_f
import Krylov_complexity_1DB_manu2 as fig2c_f

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 13 
yl_fs = 13
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

scat_size = 10

fig,ax = plt.subplots(1,3,figsize=(8,2.5))
fig.subplots_adjust(wspace=0.25)

fig2a_f.plot_ax_2a(ax[0])
print('plotted fig2a')
fig2b_f.plot_ax_2b(ax[1])
print('plotted fig2b')
fig2c_f.plot_ax_2c(ax[2])
print('plotted fig2c')

ax[0].set_ylim(0,60)

ax[0].annotate('(a)',xy=(0.02,0.9),xycoords='axes fraction',fontsize=tp_fs)
ax[1].annotate('(b)',xy=(0.02,0.9),xycoords='axes fraction',fontsize=tp_fs)
ax[2].annotate('(c)',xy=(0.02,0.9),xycoords='axes fraction',fontsize=tp_fs)

#plt.savefig('FIG2.pdf',dpi=300,bbox_inches='tight')
plt.show()
