###=============================================================###
###  Compute FFT of a second-order response function		###
###=============================================================###

import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt

__version__ = "2"
__date__ = "Jan2023"
__author__ = "Y. Litman"

only_time=False
positive_and_negative_times=True
# Parameters for plots
cm_color = "jet"
cm_color = "bwr"
titlesz = 10
plt.rcParams["axes.linewidth"] = 1.5
color = ["black", "white"]
lw = 1.0
labelsz = 12.0
ticksz = 11
figsize = [8, 8]
nlevels = 100
# Time plots
time_xlabel = "t1 (a.u.)"
time_ylabel = "t2 (a.u.)"
time_axis_x = [-20, 20]
time_axis_y = [-20, 20]
# Freq-plots
freq_xlabel = r'$\omega_1$'
freq_ylabel = r'$\omega_2$'
freq_axis_x = [-2.5, 2.5]
freq_axis_y = [-2.5, 2.5]


def main(
    filename="test_sym.dat",
    tau=13.0,
    ndim1=300,
    ndim2=300,
    dt=0.1,
    beta=1,
    n_order=2,
    lmax=None,
    tcf_type='Quantum'
):
    """Compute and plot FFT CF-sym and CF-asym
    filename_sym: COMPLETE
    ndim1: Dimension along time 1
    ndim2: Dimension along time 2
    dt: time step
    tau: time scalce of damping function:
         CF[i,j]   *= np.exp(-delta)
          with	delta = np.power(np.abs(time[i])/tau,n_order) + np.power(np.abs(time[j])/tau,n_order)
    n_order = COMPLETE
    """

    ### Read data: Expected  order is [time1,time2,CF.real,CF.imag]
    time_t,temp_r1, temp_i1 = np.loadtxt(filename, unpack=True, usecols=[0,2, 3])
    CF_1 = temp_r1   + 1.0j * temp_i1
    #CF_1 = temp_i1   #+ 1.0j * temp_i1
    if tcf_type=='Quantum':
        CF_1 =(-1.0j)**3 *(CF_1 - np.conj(CF_1))*0.5
        #CF_1 =  2.0 * CF_1
    elif tcf_type=='classical':
         pass
    elif tcf_type=='rpmd':
         pass
    elif tcf_type=='Kubo':
         pass
    else:
      raise NotImplementedError

    assert ndim1 == ndim2
    ndim = ndim1

    if not positive_and_negative_times:
       tmax = ndim * dt
       time = np.linspace(0, ndim * dt, ndim)
    else:
       tmax = (ndim-1)//2 * dt
       time = np.linspace(-tmax, +tmax, ndim)


    #time = np.zeros(ndim)
    #for i in range(ndim):
    #    time[i] = temp_t[i]



    CF_1 = np.reshape(CF_1, [ndim, ndim])

    ### Apply damping
    for i in range(ndim):
        for j in range(ndim):
            delta = np.power(np.abs(time[i]) / tau, n_order) + np.power(np.abs(time[j]) / tau, n_order   )
            CF_1[i, j] *= np.exp(-delta)
    ###================================================================
    ###====================== Computing FFTs =========================#
    ###================================================================

    freq = np.fft.fftfreq(ndim, dt)
    freq *= 2.0 * np.pi
    freq = np.fft.fftshift(freq)


    CF_1_fft  = np.fft.fft2(np.fft.fftshift(CF_1))*dt*dt
    CF_1_fft  = np.fft.fftshift(CF_1_fft)

    print("\n >>> Performing 2D FFT: done \n")

    ###================================================================
    ########## Computing terms of the response function ###############
    ###================================================================

    Resp_real = np.zeros([ndim, ndim])
    Resp_imag = np.zeros([ndim, ndim])
    for i in range(ndim):
        for j in range(ndim):
            Resp_real[i,j] =  CF_1_fft[i,j].real
            Resp_imag[i,j] =  CF_1_fft[i,j].imag

    ###================================================================
    ###===================== Plot Time-domain =========================
    ###================================================================

    title1 = "CF.real"
    title2 = "CF.imag"
    legend = [title1, title2]

    # 2D-plot
    z1 = CF_1.real
    z2 = CF_1.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=time_xlabel,
        ylabel1=time_ylabel,
        ylabel2=time_ylabel,
        axis_x=time_axis_x,
        axis_y=time_axis_y,
        figsize=figsize,
        lmax=lmax
    )

    if only_time:
        plt.show()
        print('\n\nplot only time\n\n')
        sys.exit()
    ###================================================================
    ###==================== Plot Freq-domain ==========================
    ###================================================================

    if True:
        title1 = "Resp Real"
        title2 = "Resp imag"
        temp_data_sym = Resp_real
        temp_data_asym = Resp_imag

        legend = [title1, title2]

        fig6 = plot_2d_2(
        freq,
        freq,
        temp_data_sym,
        temp_data_asym,
        title1=title1,
        title2=title2,
        xlabel2=freq_xlabel,
        ylabel1=freq_ylabel,
        ylabel2=freq_ylabel,
        axis_x=freq_axis_x,
        axis_y=freq_axis_y,
        lmax=lmax
    )
        if False:
            idx = int(ndim / 2)
            data = [temp_data_sym[idx, :], temp_data_asym[idx, :]]
            title = "cut at w1={}".format(freq[idx])
            fig6a = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend,x_lim=freq_axis_x)

    ###================================================================
    plt.show()
    #plt.savefig('my_plot.pdf')

    ###================================================================
    ###========================= Save-data ============================
    ###================================================================

    print("Have a nice day!!\n")


def plot_1d_list(x, y, figtitle="", title="", xlabel="", ylabel="", legend="",x_lim=None,y_lim=None):
    """1d plot where y is a list of curves"""

    ### define figure
    fig, ax = plt.subplots()

    ### define legend
    if not legend:
        legend = np.zeros(len(y))

    ### plot data
    for i in range(len(y)):
        ax.plot(x, y[i], label=legend[i])

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax.set_title(title, size=titlesz)
    ax.set_xlabel(xlabel, size=labelsz)
    ax.set_ylabel(ylabel, size=labelsz)
    if x_lim is not None:
       ax.set_xlim(x_lim)
    if y_lim is not None:
       ax.set_ylim(y_lim)

    ax.legend()

    return fig


def plot_2d(
    x,
    y,
    z,
    figtitle="",
    title="",
    xlabel="",
    ylabel="",
    axis_x="",
    axis_y="",
    fisize=[8, 8],
    nlevels=100,
    lmax=None
):
    """Contour plot"""

    ### define figure
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)

    ### define levels
    if lmax is None:
       lmax = np.max(np.abs(z))
    lmin = -lmax
    ldel = (lmax - lmin) / nlevels
    levels = np.arange(lmin, lmax + ldel, ldel)
    idx = np.argmin(np.abs(levels))
    levels = np.delete(levels, idx)  # remove 0.0 contour line

    ### plot data
    CS0 = ax.contourf(x, y, np.transpose(z1), cmap=cm_color, levels=levels)
    CL0 = ax.contour(CS0, colors="k", linewidths=lw)

    ### plot colorbars
    CB0 = fig.colorbar(CS0)

    ### define axis limits
    if axis_x:
        ax.set_xlim(axis_x)
    if axis_y:
        ax.set_ylim(axis_y)

        ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax.set_title(title1, size=titlesz)

    ax.set_xlabel(xlabel1, size=labelsz)

    ax.set_ylabel(ylabel1, size=labelsz)

    return fig


def plot_2d_2(
    x,
    y,
    z1,
    z2,
    figtitle="",
    title1="",
    xlabel1="",
    ylabel1="",
    title2="",
    xlabel2="",
    ylabel2="",
    axis_x="",
    axis_y="",
    figsize=[8, 8],
    nlevels=100,
    reference=None,
    lmax=None):
    """Contour plot [x2]"""

    ### define figure
    fig, ax = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)

    ### define levels
    if reference is None and lmax is None:
        lmax = max(np.max(np.abs(z1)), np.max(np.abs(z2)))
    elif reference ==1 :
            lmax = np.max(np.abs(z1))
    elif reference ==2 :
            lmax = np.max(np.abs(z2))
    elif reference is None and lmax is not None:
         pass
    else:
            raise NotImplementedError
    lmin = -lmax
    ldel = (lmax - lmin) / nlevels
    levels = np.arange(lmin, lmax + ldel, ldel)
    idx = np.argmin(np.abs(levels))
    levels = np.delete(levels, idx)  # remove 0.0 contour line

    ### plot data
    CS0 = ax[0].contourf(x, y, np.transpose(z1), cmap=cm_color, levels=levels)

    CS1 = ax[1].contourf(x, y, np.transpose(z2), cmap=cm_color, levels=levels)
    ### plot colorbars
    CB0 = fig.colorbar(CS0, ax=ax[0])
    CB1 = fig.colorbar(CS1, ax=ax[1])

    ### define axis limits
    if axis_x:
        ax[0].set_xlim(axis_x)
        ax[1].set_xlim(axis_x)
    if axis_y:
        ax[0].set_ylim(axis_y)
        ax[1].set_ylim(axis_y)

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax[0].set_title(title1, size=titlesz)
    ax[1].set_title(title2, size=titlesz)

    ax[0].set_xlabel(xlabel1, size=labelsz)
    ax[1].set_xlabel(xlabel2, size=labelsz)

    ax[0].set_ylabel(ylabel1, size=labelsz)
    ax[1].set_ylabel(ylabel2, size=labelsz)

    return fig


def plot_2d_3(
    x,
    y,
    z1,
    z2,
    z3,
    figtitle="",
    title1="",
    xlabel1="",
    ylabel1="",
    title2="",
    xlabel2="",
    ylabel2="",
    title3="",
    xlabel3="",
    ylabel3="",
    axis_x="",
    axis_y="",
):
    """Contour plot [x3]"""

    ### define figure
    fig, ax = plt.subplots(3, figsize=figsize, sharex=True, sharey=True)

    ### define levels
    lmax = max(np.max(np.abs(z1)), np.max(np.abs(z2)), np.max(np.abs(z3)))
    lmin = -lmax
    ldel = (lmax - lmin) / nlevels
    levels = np.arange(lmin, lmax + ldel, ldel)
    idx = np.argmin(np.abs(levels))
    levels = np.delete(levels, idx)  # remove 0.0 contour line

    ### plot data
    CS0 = ax[0].contourf(x, y, np.transpose(z1), cmap=cm_color, levels=levels)
    CL0 = ax[0].contour(CS0, colors="k", linewidths=lw)

    CS1 = ax[1].contourf(x, y, np.transpose(z2), cmap=cm_color, levels=levels)
    CL1 = ax[1].contour(CS1, colors="k", linewidths=lw)

    CS2 = ax[2].contourf(x, y, np.transpose(z3), cmap=cm_color, levels=levels)
    CL2 = ax[2].contour(CS2, colors="k", linewidths=lw)

    ### plot colorbars
    CB0 = fig.colorbar(CS0, ax=ax[0])
    CB1 = fig.colorbar(CS1, ax=ax[1])
    CB2 = fig.colorbar(CS2, ax=ax[2])

    ### define axis limits
    if axis_x:
        ax[0].set_xlim(axis_x)
        ax[1].set_xlim(axis_x)
        ax[2].set_xlim(axis_x)
    if axis_y:
        ax[0].set_ylim(axis_y)
        ax[1].set_ylim(axis_y)
        ax[2].set_ylim(axis_y)

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax[0].set_title(title1, size=titlesz)
    ax[1].set_title(title2, size=titlesz)
    ax[2].set_title(title3, size=titlesz)

    ax[0].set_xlabel(xlabel1, size=labelsz)
    ax[1].set_xlabel(xlabel2, size=labelsz)
    ax[2].set_xlabel(xlabel3, size=labelsz)

    ax[0].set_ylabel(ylabel1, size=labelsz)
    ax[1].set_ylabel(ylabel2, size=labelsz)
    ax[2].set_ylabel(ylabel3, size=labelsz)

    return fig


def plot_1d(x, y, figtitle="", title="", xlabel="", ylabel=""):
    """1d plot"""

    ### define figure
    fig, ax = plt.subplots()

    ### plot data
    ax.plot(x, y)

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax.set_title(title, size=titlesz)

    ax.set_xlabel(xlabel, size=labelsz)

    ax.set_ylabel(ylabel, size=labelsz)

    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Script to plot two time CF in time and frequency domain.\n
        example of usage \n
        python ../../FFT_2D.py -file1 DKTpp.dat --file2  KT_PB_ABCp.dat -n1 100 -n2 100 -beta 1 -dt 0.5 """ )

    parser.add_argument(
        "-file1", "--file1", type=str, default="test_sym.dat", help="Symmetric CF"
    )
    parser.add_argument(
        "-tau", "--tau", type=float, default=13.0, help="Time scale of damping function"
    )
    parser.add_argument(
        "-beta", "--beta", type=float, required=True, help="Beta in atomic units"
    )
    parser.add_argument("-dt", "--dtime", type=float, default=0.1, help="Time step ")
    parser.add_argument(
        "-n1", "--ndim_1", type=int, default=300, help="Dimension along time 1"
    )
    parser.add_argument(
        "-n2", "--ndim_2", type=int, default=300, help="Dimension along time 2"
    )
    parser.add_argument(
        "-lmax", "--lmax", type=float, default=None, help="Intensity Maximum"
    )
    parser.add_argument(
        "-t", "--tcf_type", type=str, choices=['Kubo','Quantum','classical','rpmd'],help='Type of TCF',required=True,
    )

    args = parser.parse_args()
    filename = args.file1
    tau = args.tau
    ndim1 = args.ndim_1
    ndim2 = args.ndim_2
    dt = args.dtime
    beta = args.beta

    main(filename, tau, ndim1, ndim2, dt, beta,n_order=2,lmax=args.lmax,tcf_type=args.tcf_type)
