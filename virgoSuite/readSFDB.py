# Copyright (C) 2023  Riccardo Felicetti
#  under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import numpy as np

location = "/storage/users/felicetti/CartaBianca/"
file_name = location + "H1:GDS-CALIB_STRAIN_20190331_232950.SFDB09"


def fread(fid, nelements, dtype):
    """
    Reads SFDB files

    Function to read SFDB files as in matlab.

    Parameters
    **********
        fid : str
            file path.
        nelements : int
            number of elements to be read.
        dtype : type
            type of the element to select.

    Examples
    ********
        Extracting an integer from example.SFDB09

        >>>fread("example.SFDB09", 1, np.int32)
        [[0]]

    """
    if dtype is str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype
    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array


with open(file_name) as fid:
    print(fread(fid, 1, np.int32))
    print(fread(fid, 1, np.int32))
    print(fread(fid, 1, np.int32))
    print(fread(fid, 1, np.double))
