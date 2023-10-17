import numpy as np
import argparse
from PISC.utils.readwrite import *
from os import walk
from pathlib import Path


def main(prefix, name_output, folder):
    """Average time correlation function"""
    aux = []
    if folder is None:
        folder = "."
    for dirpath, dirnames, filenames in walk(folder):
        aux.extend(filenames)
        break
    f = [item for item in aux if prefix in item]
    ndata = len(f)
    assert ndata > 0, "We havent found any file. Folder {}".format(folder)
    print("# We have found {} files ".format(ndata))

    CF = None
    for filename in f:
        data = np.loadtxt(Path(folder) / filename)

        if CF is None:
            CF = np.zeros_like(data)

        CF += data

    CF /= ndata
    np.savetxt(name_output, CF)
    print("# Please check {}\n".format(name_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Script average seeds for time correlation function \n"""
    )
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
    main(args.prefix, args.out, args.folder)
