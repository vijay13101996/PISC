import numpy as np
import argparse
import sys


def rotate_time(file_name, ndimensions, dt, output_name):
    """Rotate time from (t1,t2) to (t1,t1+t2)"""
    data = np.loadtxt(file_name)
    if ndimensions != 2:
        assert data.shape[1] == 4, "Data must have four columns"
    else:
        raise NotImplementedError("Only 2D data is supported")

    # Get the time
    nlen = int(data.shape[0] ** 0.5)
    time = data[:, 0][0:nlen]
    total_time_orig = nlen * dt
    tcf_data = data[:, 2].reshape((nlen, nlen))
    # Rotate the data
    new_len = nlen
    # new_data = np.array(list(zip(*tcf_data[::-1]))) # rotates 90

    new_data = np.zeros((new_len, new_len))
    for i in range(-nlen // 2, nlen // 2):
        for j in range(-nlen // 2, nlen // 2):
            aux1_i = i + nlen // 2
            aux1_j = j + nlen // 2
            aux2_j = (i + j) + nlen // 2
            if np.absolute(aux2_j) < new_len:
                new_data[aux1_i, aux2_j] = tcf_data[aux1_i, aux1_j]
    with open(output_name, "w") as outfile:
        for i in range(new_len):
            for j in range(new_len):
                outfile.write("{} {} {} {}\n".format(i, j, new_data[i, j], 0.0))
    print("\nRotating frame completed\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("file_name", help="Name of the file to rotate")
    args.add_argument("ndimension", help="Number of dimensions of the data")
    args.add_argument("dt", help="Time step of the data")
    args.add_argument("output_name", help="Name of the output file")

    args = args.parse_args()
    rotate_time(args.file_name, args.ndimension, args.dt, args.output_name)
