import numpy as np
import argparse

debug = False


def get_slice(filename, axis, time, output_filename, manual=False):
    """Get a slice of the data at a given time and axis"""

    data = np.loadtxt(filename)
    assert data.shape[1] == 5
    nlen = round(data.shape[0] ** (1 / 3))

    if manual:
        pass
    else:
        aux_data = np.reshape(data, (nlen, nlen, nlen, 5))
        if debug:
            print("nlen", nlen)
            print("shape", aux_data.shape)
        time_array = aux_data[:, 0, 0, 0]
        index_time = np.argmin(np.abs(time_array - time))
        if axis == "t1":
            new_data = aux_data[index_time:, :, :]
        elif axis == "t2":
            new_data = aux_data[:, index_time, :, :]
        elif axis == "t3":
            new_data = aux_data[:, :, index_time, :]
        else:
            raise ValueError("axis must be t1, t2, or t3")

    with open(output_filename, "w") as outfile:
        for i in range(nlen):
            for j in range(nlen):
                if debug:
                    outfile.write(
                        "{:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}\n".format(
                            new_data[i, j, 0],
                            new_data[i, j, 1],
                            new_data[i, j, 2],
                            new_data[i, j, 3],
                            new_data[i, j, 4],
                        )
                    )
                else:
                    if axis == "t1":
                        outfile.write(
                            "{:6.3f} {:6.3f} {:6.3f} {:6.3f}\n".format(
                                new_data[i, j, 1],
                                new_data[i, j, 2],
                                new_data[i, j, 3],
                                new_data[i, j, 4],
                            )
                        )
                    elif axis == "t2":
                        outfile.write(
                            "{:6.3f} {:6.3f} {:6.3f} {:6.3f}\n".format(
                                new_data[i, j, 0],
                                new_data[i, j, 2],
                                new_data[i, j, 3],
                                new_data[i, j, 4],
                            )
                        )
                    elif axis == "t3":
                        outfile.write(
                            "{:6.3f} {:6.3f} {:6.3f} {:6.3f}\n".format(
                                new_data[i, j, 0],
                                new_data[i, j, 1],
                                new_data[i, j, 3],
                                new_data[i, j, 4],
                            )
                        )
                    else:
                        raise ValueError("axis must be t1, t2, or t3")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        """ Get a slice of the data at a given time and axis"""
    )
    argparser.add_argument("filename", help="filename of the data")
    argparser.add_argument(
        "axis", choices=["t1", "t2", "t3"], help="axis to slice along"
    )
    argparser.add_argument("time", type=float, help="time to slice at")
    argparser.add_argument("output_filename", type=str, help="filename of the output")
    argparser.add_argument(
        "--manual", action="store_true", help="manually select the slice"
    )

    args = argparser.parse_args()

    get_slice(args.filename, args.axis, args.time, args.output_filename, args.manual)
