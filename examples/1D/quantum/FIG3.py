import numpy as np
from matplotlib import pyplot as plt
import Krylov_complexity_billiards_1DB as fig_3a
import Krylov_complexity_XXZ_draft1 as fig_3b
import matplotlib

plt.rcParams.update({'font.size': 10, 'font.family': 'serif','font.style':'italic','font.serif':'Garamond'})
matplotlib.rcParams['axes.unicode_minus'] = False

xl_fs = 16 
yl_fs = 16
tp_fs = 12

le_fs = 13#9.5
ti_fs = 12

fig, ax = plt.subplots(2,1, figsize=(4,8))
plt.subplots_adjust(wspace=0.1,hspace=0.1)

axin3a = ax[0].inset_axes([0.55, 0.02, 0.4, 0.3]) # x0, y0, width, height
fig_3a.plot_ax_3a_in(axin3a)
print('plotted fig3a inset')
fig_3a.plot_ax_3a(ax[0])
print('plotted fig3a')
fig_3b.plot_ax_3b(ax[1])
print('plotted fig3b')

axin3b = ax[1].inset_axes([0.5, 0.65, 0.45, 0.33]) # x0, y0, width, height
fig_3b.plot_ax_3b_in(axin3b)


#plt.savefig('FIG3.pdf',dpi=300,bbox_inches='tight')
plt.show()


