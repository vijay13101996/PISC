import numpy as np
import argparse
from PISC.utils.readwrite import *
from os import walk
from pathlib import Path

def main(prefix, name_output,folder):
    aux = []
    if folder is None:
        folder ='.'
    for dirpath, dirnames, filenames in walk(folder):
        aux.extend(filenames)
        break
    f = [item for item in aux if prefix in item]
    ndata = len(f)
    assert  ndata>0,'We havent found any file. Folder {}'.format(folder)
    print("#We have found {} files ".format(ndata))

    t1 = None
    t2 = None
    CF = None
    for filename in f:
        print(filename)
        X, Y, F = read_2D_imagedata(Path(folder)/filename)

        if t1 is None:
            X[:, len(X) // 2 + 1 :] = X[:, : -len(X) // 2 : -1]
            X = np.roll(X, len(X) // 2, axis=1)
            t1 = X
        if t2 is None:
            Y[len(Y) // 2 + 1 :, :] = Y[: -len(Y) // 2 : -1, :]
            Y = np.roll(Y, len(Y) // 2, axis=0)
            t2 = Y
        if CF is None:
            CF = np.zeros_like(F)

        F[:, len(X) // 2 + 1 :] = F[:, : -len(X) // 2 : -1]
        F[len(Y) // 2 + 1 :, :] = F[: -len(Y) // 2 : -1, :]
        F = np.roll(np.roll(F, len(X) // 2, axis=1), len(Y) // 2, axis=0)

        CF += F

    CF/=ndata
    store_2D_imagedata_column(t1, t2, CF, name_output,fpath='.',extcol=np.zeros_like(X))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Script average seeds for 2 time correlation function \n""")
    parser.add_argument(
        "-pre", "--prefix", type=str, default="", help="Prefix for all files"
    )
    parser.add_argument(
        "-folder", "--folder", type=str, default=None, help="Folder containing files"
    )
    parser.add_argument(
        "-out", "--out", type=str, default="average.out", help="Output name"
    )

    args = parser.parse_args()
    main(args.prefix, args.out,args.folder)
