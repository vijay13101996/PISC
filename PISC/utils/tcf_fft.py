import numpy as np
import sys

try:
    from PISC.utils import tcf_fort_tools as tcf_fort_tools_omp
except ImportError:
    print("\nCannot import tcf_fort_tools @ PISC/utils/tcf_fft.py")
    print('Please compile these tools by running the "PISC/utils/compile.sh" script\n')
    sys.exit(1)


def gen_tcf(Aarr, Barr, tarr, corraxis=None):
    q1arr = np.array(Aarr)
    q1arr[(len(tarr) // 2) :] = 0.0
    q1_tilde = np.fft.rfft(q1arr, axis=0)
    q2_tilde = np.fft.rfft(Barr, axis=0)

    tcf = np.fft.irfft(np.conj(q1_tilde) * q2_tilde, axis=0)
    tcf = tcf[: len(tcf) // 2, :, :]  # Truncating the padded part
    if corraxis is None:
        tcf = np.sum(tcf, axis=2)  # Summing over the dimension (Dot product)
    else:
        tcf = tcf[:, :, corraxis]
    tcf = np.mean(tcf, axis=1)  # Averaging over the particles

    tcf /= len(tcf)
    tarr = tarr[: len(tcf)]
    return tarr, tcf


def gen_2pt_tcf(dt, tarr, Carr, Barr, Aarr=None, dt_tcf=0.1, trans_sym=False):
    """Function to compute 2 time correlation functions related to R2
    The ordering is C[t2]B[t1]A[t0].
    If A is not provided then we consider  C[t2]B[t1;t0].
    If trans_sym is false, t0 is set to 0 by default, as it a 2-point TCF.
    """

    # Rewrite this in FORTRAN, this is extremely slow.

    if trans_sym:
        # Use symmetry w.r.t translation about t0
        halflen = len(tarr) // 2
        tcf = np.zeros((halflen, halflen))
        for k in range(halflen):
            for i in range(halflen):  # t1 axis
                for j in range(halflen):  # t2 axis
                    # Dot product and Ensemble average in the same order.
                    tcf[i, j] += np.mean(
                        np.sum(Carr[j + k] * Barr[i + k] * Aarr[k], axis=1), axis=0
                    )
        tcf /= halflen  # To normalise contributions from all t1 translated tcfs.
        tcf = tcf[:halflen]
        tarr = tarr[:halflen]
        return tarr, tcf

    else:
        stride = 1
        if dt < dt_tcf:
            stride = int(dt_tcf // dt)
        tar = tarr[::stride]
        Car = Carr[::stride]
        Bar = Barr[::stride]
        if Aarr is not None:
            Aar = Aarr[::stride]
        tlen = len(tar)
        tcf = np.zeros((tlen, tlen))

        if 1:  # FORTRAN
            tcf_fort = np.ascontiguousarray(tcf)
            Bar = np.asfortranarray(Bar)
            Car = np.asfortranarray(Car)
            if Aarr is not None:
                Aar = np.asfortranarray(Aar)
                tcf = tcf_fort_tools_omp.tcf_tools.two_pt_3op_tcf(
                    Aar, Bar, Car, tcf_fort
                )
            else:
                tcf = tcf_fort_tools_omp.tcf_tools.two_pt_2op_tcf(Bar, Car, tcf_fort)
        if 0:  # PYTHON
            tcf[:] = 0.0
            for i in range(tlen):  # t1 axis
                for j in range(tlen):  # t2 axis
                    # Dot product and Ensemble average in the same order.
                    if Aarr is None:
                        tcf[i, j] = np.mean(np.sum(Car[j] * Bar[i], axis=1), axis=0)
                    else:
                        tcf[i, j] = np.mean(
                            np.sum(Car[j] * Bar[i] * Aar[0], axis=1), axis=0
                        )

        return tar, tcf


def gen_R3_tcf(dt, tarr, Aarr, Barr, Marr, beta, dt_tcf=0.1, verbose=0):
    """Function to compute third-order response
    R3 = beta < (Mqq(t3,t0)Mqp(t2,t0) - Mqp(t3,t0)Mqq(t2,t0)) - (Mpp(-t1,t0)-beta p(0)p(-t1)) >
    """
    verbose = 5

    stride = 1
    if dt < dt_tcf:
        stride = int(dt_tcf // dt)
    tar = tarr[::stride]
    Aar = Aarr[::stride]
    Bar = Barr[::stride]
    Mqq = Marr["qq"][::stride]
    Mqp = Marr["qp"][::stride]
    Mpq = Marr["pq"][::stride]
    Mpp = Marr["pp"][::stride]
    tlen = len(tar) // 2
    ndim = 2 * tlen + 1
    tcf = np.zeros((tlen + 1, tlen + 1, tlen + 1))
    if verbose > 1:
        print("tar", tar.shape)
        print("Aar", Aar.shape)
        print("Bar", Bar.shape)
        print("Mqq", Mqq.shape)
        print("Mqp", Mqp.shape)
        print("Mpq", Mpq.shape)
        print("Mpp", Mqq.shape)
        print(tlen, ndim, tcf.shape)
    if 0:  # FORTRAN
        raise NotImplementedError
        # tcf_fort = np.ascontiguousarray(tcf)
        # Aar = np.asfortranarray(Aar)
        # Bar = np.asfortranarray(Bar)
        # if Aarr is not None:
        #    Aar = np.asfortranarray(Aar)
        #    tcf = tcf_fort_tools_omp.tcf_tools.two_pt_3op_tcf(Aar, Bar, Car, tcf_fort)
        # else:
        #    tcf = tcf_fort_tools_omp.tcf_tools.two_pt_2op_tcf(Bar, Car, tcf_fort)
    if 1:  # PYTHON
        # tarr [0,1,...,tlen,-1,...,-tlen]*dt_tcf
        for it1 in range(tlen + 1):
            if it1 > 0:
                index1 = tlen + it1 - 1
            else:
                index1 = it1
            print(it1)
            for it2 in range(tlen + 1):
                for it3 in range(tlen + 1):
                    # Dot product and ensemble average in the same line.
                    # tcf[it3,it2,it1] = np.mean(np.sum(Car[j]*Bar[i]*Aar[0],axis=1),axis=0)
                    # R3 = beta < ( Mqq(t3,t0) Mqp(t2,t0) - Mqp(t3,t0)Mqq(t2,t0)  ) - (Mpp(-t1,t0)-beta p(0)p(-t1)) >
                    tcf[it3, it2, it1] = np.mean(
                        beta * (Mqq[it3] * Mqp[it2] - Mqp[it3] * Mqq[it2])
                        - (Mpp[index1] - beta * Bar[0] * Aar[index1])
                    )

        return tar, tcf
