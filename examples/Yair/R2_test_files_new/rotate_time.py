import numpy as np
import argparse
import sys


def rotate_time(file_name, ndimensions, dt, output_name):
    """Rotate the data in the file_name"""
    data = np.loadtxt(file_name)
    if ndimensions != 2:
        assert data.shape[1] == 4, "Data must have four columns"
    else:
        raise NotImplementedError("Only 2D data is supported")

    # Get the time
    nlen = int(data.shape[0] ** 0.5)
    time = data[:, 0][0:nlen]
    total_time_orig = nlen * dt
    tcf_data = data[:,2].reshape((nlen, nlen))
    print(np.max(tcf_data))
    print(np.min(tcf_data))
    # Rotate the data
    new_len=nlen
    # new_data = np.array(list(zip(*tcf_data[::-1]))) # rotates 90

    from scipy.ndimage import rotate
    #new_data = rotate(tcf_data,angle=-45)
    new_data =tcf_data
    #new_data = np.zeros((new_len, new_len))
    #for i in range(nlen//2):
    #    for j in range(nlen//2):
    #        new_data[(i + j) , (i - j)] = tcf_data[i,j]
    with open(output_name, "w") as outfile:
        for i in range(new_len):
            for j in range(new_len):
                outfile.write("{} {} {} {}\n".format(i, j, new_data[i, j],0.0))


if __name__ == "__main__":
    file_name = sys.argv[1]
    ndimension = sys.argv[2]
    dt = sys.argv[3]
    output_name = sys.argv[4]

    rotate_time(file_name, ndimension, dt,output_name)
