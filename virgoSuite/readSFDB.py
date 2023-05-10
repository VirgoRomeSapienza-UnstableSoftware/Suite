# Copyright (C) 2023  Riccardo Felicetti
#  under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

import numpy as np

location = "C:/Coding/CartaBianca/"
file_name = location + "H1_DCS-CALIB_STRAIN_GATED_SUB60HZ_C01_20190401_000000.SFDB09"


def fread(fid, nelements, dtype):
    """
    Reads SFDB like matlab

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

        >>> fread("example.SFDB09", 1, np.int32)
        [[0]]

    """
    if dtype is str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype
    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array


def foo(foo1, foo2):
    """foo _summary_

    _extended_summary_

    Arguments:
        foo1 -- _description_
        foo2 -- _description_
    """

    return 0
