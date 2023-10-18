###=============================================================###
###  Compute FFT of a second-order response function		###
###=============================================================###

import sys
import numpy as np
import argparse
import time as clock
from matplotlib import pyplot as plt
from toolkit.tools.non_linear.kubo_factors import h_factor

__version__ = "1"
__date__ = "Apr2023"
__author__ = "Y. Litman"

only_time=False
plot_time=False
positive_and_negative_times=True
plot_1d_cuts=False
plot_1d_cuts=True
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
nlevels = 400
# Time plots
time_xlabel = "t1 (a.u.)"
time_ylabel = "t2 (a.u.)"
time_axis_x = [-20, 20]
time_axis_y = [-20, 20]
# Freq-plots
freq_xlabel = r'$\omega_1$'
freq_ylabel = r'$\omega_3$'
freq_axis_x = [-4.0, 4.0]
freq_axis_y = [-4.0, 4.0]
freq_axis_x = [-2.5, 2.5]
freq_axis_y = [-2.5, 2.5]

beta=8.0

def main(
    filename="test_sym.dat",
    tau=13.0,
    ndim1=300,
    dt=0.1,
    factor=1,
    n_order=2,
    lmax=None,
    quantum=False,
    omega2=2.0,
    n_zeros=0.0,
    tcf_type='Quantum',
    beta=8.0
):
    """Compute and plot FFT CF-sym and CF-asym
    filename_sym: COMPLETE
    ndim1: Dimension along time 1
    dt: time step
    tau: time scalce of damping function:
         CF[i,j]   *= np.exp(-delta)
          with	delta = np.power(np.abs(time[i])/tau,n_order) + np.power(np.abs(time[j])/tau,n_order)
    n_order = COMPLETE
    """

    ### Read data: Expected  order is [time1,time2,CF.real,CF.imag]
    ti=clock.time()
    time_t,temp_r1, temp_i1 = np.loadtxt(filename, unpack=True, usecols=[0,3, 4])
    CF_1 = temp_r1   + 1.0j * temp_i1
    if tcf_type =='Quantum':
       CF_1 = ( (-1.0j)**3 ) * (CF_1 - np.conj(CF_1))*0.5
       CF_1 = -1.0 * CF_1 # ALBERTO
    elif tcf_type=='Kubo' or tcf_type=='Ksym':
       pass
    elif tcf_type=='classical':
       pass
    elif tcf_type=='rpmd':
       pass
    else:
      raise NotImplementedError

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



    CF_1 = np.reshape(CF_1, [ndim, ndim,ndim])
    tf=clock.time()
    print("\n >>> Read data: done  ({} s)\n".format(tf-ti))

    ########## Apply damping #################################
    ti=clock.time()
    if tau>0:
     for i in range(ndim):
        for j in range(ndim):
            for k in range(ndim):
               delta = np.power(np.abs(time[i]) / tau, n_order) + np.power(np.abs(time[j]) / tau, n_order   ) + np.power(np.abs(time[k]) / tau, n_order   )
               CF_1[i, j, k] *= np.exp(-delta)
     tf=clock.time()
     print("\n >>> Apply damping : done  ({} s)\n".format(tf-ti))
    else:
        print('Skiping damping ...')

    if quantum:
        CF_1 =  2.0 * CF_1


    ################# Add zeros  ############################
    ti=clock.time()
    if n_zeros>0:
        n_aux = 2*n_zeros + ndim
        CF_aux = np.zeros([n_aux,n_aux,n_aux],dtype=np.complex128)
        CF_aux[n_zeros:-n_zeros,n_zeros:-n_zeros,n_zeros:-n_zeros]=CF_1
        CF_1=CF_aux
        ndim = n_aux

    tf=clock.time()
    print("\n >>> Add zeros: done  ({} s)\n".format(tf-ti))

    #CF_1*=factor
    ###================================================================
    ###====================== Computing FFTs =========================#
    ###================================================================
    ti=clock.time()
    freq = np.fft.fftfreq(ndim, dt)
    freq *= 2.0 * np.pi
    freq = np.fft.fftshift(freq)
    CF_1_fft  = np.fft.fftn(np.fft.fftshift(CF_1))*dt*dt
    CF_1_fft  = np.fft.fftshift(CF_1_fft)
    tf=clock.time()

    print("\n >>> Performing 2D FFT: done  ({} s)\n".format(tf-ti))

    if tcf_type=='Kubo':
       ti=clock.time()
       kubo_factor = np.zeros((ndim,ndim))
       idx = (np.abs(freq - omega2)).argmin()
       for i in range(ndim):
           for j in range(ndim):
                 kubo_factor[i,j] =  h_factor(freq[i],freq[idx],freq[j],beta)

       tf=clock.time()
       print("\n >>> Computing Kubo Factor: done ({} s) \n".format(tf-ti))

    ###================================================================
    ########## Computing terms of the response function ###############
    ###================================================================

    Resp_real = np.zeros([ndim, ndim])
    Resp_imag = np.zeros([ndim, ndim])
    idx = (np.abs(freq - omega2)).argmin()
    print('Lookint at w2= {}'.format(freq[idx]))
    for i in range(ndim):
        for j in range(ndim):
            Resp_real[i,j] =  CF_1_fft[i,idx,j].real
            Resp_imag[i,j] =  CF_1_fft[i,idx,j].imag
    if tcf_type =='Quantum' or tcf_type=='Ksym':
           pass
    elif tcf_type=='Kubo':
               Resp_real = - np.multiply(Resp_real, kubo_factor)
               Resp_imag = - np.multiply(Resp_imag, kubo_factor)


    ###================================================================
    ###===================== Plot Time-domain =========================
    ###================================================================
    if plot_time:
        title1 = "CF.real " + filename
        title2 = "CF.imag " + filename
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

    if tcf_type=='Kubo':
        title1 = "Kubo factor, beta {} ".format(beta)
        temp_data_real = kubo_factor

        fig6 = plot_2d(
        freq,
        freq,
        temp_data_real,
        title=title1,
        xlabel=freq_xlabel,
        ylabel=freq_ylabel,
        axis_x=freq_axis_x,
        axis_y=freq_axis_y,
        lmax=1.0
    )
    if True:
        title1 = "Resp Real " + filename + r' ($\omega_2={}$)'.format(omega2)
        title2 = "Resp Imag " + filename + r' ($\omega_2={}$)'.format(omega2)
        idx = (np.abs(freq - omega2)).argmin()
        #data_1 = CF_1_fft[:,idx,:].real *factor
        data_1 = Resp_real
        data_2 = Resp_imag

        legend = [title1, title2]

        fig6 = plot_2d_2(
        freq,
        freq,
        data_1,
        data_2,
        title1=title1,
        title2=title2,
        xlabel2=freq_xlabel,
        ylabel1=freq_ylabel,
        ylabel2=freq_ylabel,
        axis_x=freq_axis_x,
        axis_y=freq_axis_y,
        lmax=lmax
    )
        if plot_1d_cuts:
            #omega_target1 = 0.0
            #idx1 = (np.abs(freq - omega_target1)).argmin()
            #title1 = "w3={} ".format(freq[idx1])
            #dataA = data_2[:, idx1]
            #
            #omega_target2 = 1.0
            #idx2 = (np.abs(freq - omega_target2)).argmin()
            #title2 = "w3={} ".format(freq[idx2])
            #legend = [title1, title2]
            #dataB = data_2[:, idx2]

            #omega_target3 = 2.0
            #idx3 = (np.abs(freq - omega_target3)).argmin()
            #title3 = "w3={} ".format(freq[idx3])
            #dataC = data_2[:, idx3]

            #omega_target4 = -1.0
            #idx3 = (np.abs(freq - omega_target4)).argmin()
            #title4 = "w3={} ".format(freq[idx3])
            #dataD = data_2[:, idx3]

            #legend = [title1, title2,title3,title4]
            #title = "1d cuts (fixed  w3)"
            #data = [dataA,dataB,dataC,dataD]
            #fig6a = plot_1d_list(freq, data, title=title, xlabel="w1", legend=legend,x_lim=freq_axis_x,normalize=False)

            omega_target1 = 0.0
            idx1 = (np.abs(freq - omega_target1)).argmin()
            title1b = "w1={} ".format(freq[idx1])
            dataAb = data_2[ idx1,:]

            omega_target2 = 1.0
            idx2 = (np.abs(freq - omega_target2)).argmin()
            title2b = "w1={} ".format(freq[idx2])
            legend = [title1, title2]
            dataBb = data_2[ idx2,:]

            omega_target3 = 2.0
            idx3 = (np.abs(freq - omega_target3)).argmin()
            title3b = "w1={} ".format(freq[idx3])
            dataCb = data_2[ idx3,:]

            omega_target4 = -1.0
            idx3 = (np.abs(freq - omega_target4)).argmin()
            title4b = "w1={} ".format(freq[idx3])
            dataDb = data_2[ idx3,:]

            legend = [title1b, title2b,title3b,title4b]
            title = "1d cuts (fixed  w1)"
            data = [dataAb,dataBb,dataCb,dataDb]
            fig6a = plot_1d_list(freq, data, title=title, xlabel="w3", legend=legend,x_lim=freq_axis_x,normalize=False)
    if False:
        title1 = "Resp Real "+filename
        title2 = "Resp imag "+filename
        temp_data_real = Resp_real
        temp_data_imag = Resp_imag

        legend = [title1, title2]

        fig6 = plot_2d_2(
        freq,
        freq,
        temp_data_real,
        temp_data_imag,
        title1=title1,
        title2=title2,
        xlabel2=freq_xlabel,
        ylabel1=freq_ylabel,
        ylabel2=freq_ylabel,
        axis_x=freq_axis_x,
        axis_y=freq_axis_y,
        lmax=lmax
    )
        if plot_1d_cuts:
            omega_target1 = 0.0
            idx1 = (np.abs(freq - omega_target1)).argmin()
            title1 = "w1={} ".format(freq[idx1])
            data1 = temp_data_real[idx1, :]

            omega_target2 = 1.0
            idx2 = (np.abs(freq - omega_target2)).argmin()
            title2 = "w1={} ".format(freq[idx2])
            data2 = temp_data_real[idx2, :]

            omega_target3 = 2.0
            idx3 = (np.abs(freq - omega_target3)).argmin()
            title3 = "w1={} ".format(freq[idx3])
            data3 = temp_data_real[idx3, :]

            legend = [title1, title2,title3]
            title = "1d cuts (fixed  w1)"
            data = [data1,data2,data3]
            fig6a = plot_1d_list(freq, data, title=title, xlabel="w2", legend=legend,x_lim=freq_axis_x,normalize=False)


    ###================================================================
    plt.show()

    ###================================================================
    ###========================= Save-data ============================
    ###================================================================

    print("Have a nice day!!\n")


def plot_1d_list(x, y, figtitle="", title="", xlabel="", ylabel="", legend="",x_lim=None,y_lim=None,normalize=False):
    """1d plot where y is a list of curves"""

    ### define figure
    fig, ax = plt.subplots()

    ### define legend
    if not legend:
        legend = np.zeros(len(y))

    if normalize:
       for i,yy in enumerate(y):
           y[i]=yy/np.max(np.abs(yy))
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
    CS0 = ax.contourf(x, y, np.transpose(z), cmap=cm_color, levels=levels)
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

    ax.set_title(title, size=titlesz)

    ax.set_xlabel(xlabel, size=labelsz)

    ax.set_ylabel(ylabel, size=labelsz)

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
    assert lmax > lmin, 'Everything is zero?'
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
        python ../../FFT_2D.py -file1 DKTpp.dat --file2  KT_PB_ABCp.dat -n1 100 -n2 100 -factor 1 -dt 0.5 """ )

    parser.add_argument(
        "-file1", "--file1", type=str, default="test_sym.dat", help="Symmetric CF"
    )
    parser.add_argument(
        "-tau", "--tau", type=float, default=13.0, help="Time scale of damping function"
    )
    parser.add_argument(
        "-factor", "--factor", type=float, required=False,default=1.0, help="Scaling factor"
    )
    parser.add_argument("-dt", "--dtime", type=float, default=0.1, help="Time step ")
    parser.add_argument(
        "-n1", "--ndim_1", type=int, default=300, help="Dimension along time 1"
    )
    parser.add_argument(
        "-lmax", "--lmax", type=float, default=None, help="Intensity Maximum"
    )
    parser.add_argument(
        "-q", "--quantum", action='store_true', help="COMPLETE"
    )
    parser.add_argument(
        "-w2", "--omega2", type=float, default=1.0,help='Value of second frequency (atomic units)'
    )
    parser.add_argument(
        "-n0", "--nzeros", type=int, default=200,help='Number of 0 to add before applying the FFT (default 200)'
    )
    parser.add_argument(
        "-t", "--tcf_type", type=str, choices=['Kubo','Quantum','Ksym','classical','rpmd'],help='Type of TCF',required=True,
    )
    parser.add_argument(
        "-beta", "--beta", type=float, default=8.0,help='Inverse temperature (atomic units)'
    )

    args = parser.parse_args()
    filename = args.file1
    tau = args.tau
    ndim1 = args.ndim_1
    dt = args.dtime
    factor = args.factor
    main(filename, tau, ndim1,  dt, factor,n_order=2,lmax=args.lmax,quantum=args.quantum,omega2=args.omega2,n_zeros=args.nzeros,tcf_type=args.tcf_type,beta=args.beta)
