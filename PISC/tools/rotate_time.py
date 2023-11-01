import numpy as np
import argparse

debug = False


def rotate_time(file_name, ndimensions, dt, output_name):
    """Interface to specific functions"""
    if ndimensions == 2:
        rotate_time_R2(file_name, dt, output_name)
    elif ndimensions == 3:
        rotate_time_R3(file_name, dt, output_name)
    else:
        raise NotImplementedError(
            "rotate time for {} dimensions is not implemented".format(ndimensions)
        )


def rotate_time_R3(file_name, dt, output_name):
    """Rotate time from (t1,t2,t3) to (t1,t1+t2,t1+t3)"""
    data = np.loadtxt(file_name)
    assert data.shape[1] == 5, "Data must have four columns"

    # Get the time
    nlen = round(data.shape[0] ** (1.0 / 3.0))
    # Rotate the data
    tcf_data = data[:, 3].reshape((nlen, nlen, nlen))  # Assumes only real data
    # print("\ntcf_data.shape", tcf_data.shape)
    new_len = nlen

    new_data = np.zeros((new_len, new_len, new_len))
    for i in range(-nlen // 2, nlen // 2):
        for j in range(-nlen // 2, nlen // 2):
            for k in range(-nlen // 2, nlen // 2):
                aux1_i = i + nlen // 2
                aux1_j = j + nlen // 2
                aux1_k = k + nlen // 2
                aux2_j = (i + j) + nlen // 2  # ALBERTO
                aux2_k = (i + k) + nlen // 2  # ALBERTO
                if np.absolute(aux2_j) < new_len and np.absolute(aux2_k) < new_len:
                    aux = tcf_data[aux1_i, aux1_j, aux1_k]
                    new_data[aux1_i, aux2_j, aux2_k] = aux
    imaginary_part = 0.0
    with open(output_name, "w") as outfile:
        for i in range(new_len):
            for j in range(new_len):
                for k in range(new_len):
                    outfile.write(
                        "{:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}\n".format(
                            (i - nlen // 2) * dt,
                            (j - nlen // 2) * dt,
                            (k - nlen // 2) * dt,
                            new_data[i, j, k],
                            imaginary_part,
                        )
                    )

    print("\nRotating frame completed\n")


def rotate_time_R2(file_name, dt, output_name):
    """Rotate time from (t1,t2) to (t1,t1+t2)"""
    data = np.loadtxt(file_name)
    assert data.shape[1] == 4, "Data must have four columns"

    # Get the time
    nlen = int(data.shape[0] ** 0.5)
    # time = data[:, 0][0:nlen]
    # total_time_orig = nlen * dt
    tcf_data = data[:, 2].reshape((nlen, nlen))
    # Rotate the data
    new_len = nlen
    # new_data = np.array(list(zip(*tcf_data[::-1]))) # rotates 90

    new_data = np.zeros((new_len, new_len))
    imaginary_part = 0.0
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
                outfile.write(
                    "{:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(
                        (i - nlen // 2) * dt,
                        (j - nlen // 2) * dt,
                        new_data[i, j],
                        imaginary_part,
                    )
                )
    print("\nRotating frame completed\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("file_name", type=str, help="Name of the file to rotate")
    args.add_argument("ndimension", type=int, help="Number of dimensions of the data")
    args.add_argument("dt", type=float, help="Time step of the data")
    args.add_argument("output_name", help="Name of the output file")

    args = args.parse_args()
    print("\nRotating frame ...\n")
    rotate_time(args.file_name, args.ndimension, args.dt, args.output_name)
