# Copyright (C) 2023  Riccardo Felicetti
#  under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
# %%
import numpy as np
import pandas
import zarr

location = "C:/Coding/Suite/"
file_name = location + "H1_DCS-CALIB_STRAIN_GATED_SUB60HZ_C01_20190401_000000.SFDB09"


def fread(fid, n_elements: int, dtype: str) -> np.ndarray:
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
    data_array = np.fromfile(fid, dt, n_elements)

    if n_elements == 1:
        return data_array[0]
    else:
        return data_array


"""
def load_sfdb(path_to_sfdb: str) -> pandas.DataFrame:
    metadata_list = np.array(
        [
            "count",
            "detector",
            "gps_sec",
            "gps_nsec",
            "tbase",
            "firstfring",
            "nsamples",
            "red",
            "typ",
            "n_flag",
            "einstein",
            "mjdtime",
            "nfft",
            "wink",
            "normd",
            "normw",
            "frinit",
            "tsamplu",
            "deltanu",
        ]
    )
    with open(path_to_sfdb) as fid:
        sfdb_header_dict = {
            "value": np.array(
                [
                    fread(fid, 1, "double"),  # count
                    fread(fid, 1, "int32"),  # detector
                    fread(fid, 1, "int32"),  # gps_sec
                    fread(fid, 1, "int32"),  # gps_nsec
                    fread(fid, 1, "double"),  # tbase
                    fread(fid, 1, "int32"),  # firstfrind
                    fread(fid, 1, "int32"),  # nsamples
                    fread(fid, 1, "int32"),  # red
                    fread(fid, 1, "int32"),  # typ
                    fread(fid, 1, "float32"),  # n_flag
                    fread(fid, 1, "float32"),  # einstein
                    fread(fid, 1, "double"),  # mjdtime
                    fread(fid, 1, "int32"),  # nfft
                    fread(fid, 1, "int32"),  # wink
                    fread(fid, 1, "float32"),  # normd
                    fread(fid, 1, "float32"),  # normw
                    fread(fid, 1, "double"),  # frinit
                    fread(fid, 1, "double"),  # tsamplu
                    fread(fid, 1, "double"),  # deltanu
                ]
            )
        }
        _ = sfdb_header_dict["value"]
        sfdb_header_dict["value"] = _.reshape(_.shape[0])

        sfdb_header_df_1 = pandas.DataFrame(sfdb_header_dict, index=metadata_list)

        if sfdb_header_df_1.loc["detector"][0] == 0:
            metadata_list = np.array(
                [
                    "frcal",
                    "freqm",
                    "freqp",
                    "taum",
                    "taup",
                ]
            )
            sfdb_header_dict = {
                np.array(
                    [
                        fread(fid, 1, "double"),  # frcal
                        fread(fid, 1, "double"),  # freqm
                        fread(fid, 1, "double"),  # freqp
                        fread(fid, 1, "double"),  # taum
                        fread(fid, 1, "double"),  # taup
                    ]
                )
            }
        else:
            metadata_list = np.array(
                [
                    "vx_eq",
                    "vy_eq",
                    "vz_eq",
                    "px_eq",
                    "py_eq",
                    "pz_eq",
                    "n_zeros",
                    "sat_howmany",
                    "spare1",
                    "spare2",
                    "spare3",
                    "spare4",
                    "spare5",
                    "spare6",
                    "spare7",
                    "spare8",
                    "spare9",
                ]
            )
            sfdb_header_dict = {
                "value": np.array(
                    [
                        fread(fid, 1, "double"),  # vx_eq
                        fread(fid, 1, "double"),  # vy_eq
                        fread(fid, 1, "double"),  # vz_eq
                        fread(fid, 1, "double"),  # px_eq
                        fread(fid, 1, "double"),  # py_eq
                        fread(fid, 1, "double"),  # pz_eq
                        fread(fid, 1, "int32"),  # n_zeros
                        fread(fid, 1, "double"),  # sat_howmany
                        fread(fid, 1, "double"),  # spare1
                        fread(fid, 1, "double"),  # spare2
                        fread(fid, 1, "double"),  # spare3
                        fread(fid, 1, "float32"),  # spare4
                        fread(fid, 1, "float32"),  # spare5
                        fread(fid, 1, "float32"),  # spare6
                        fread(fid, 1, "int32"),  # spare7
                        fread(fid, 1, "int32"),  # spare8
                        fread(fid, 1, "int32"),  # spare9
                    ]
                )
            }

        _ = sfdb_header_dict["value"]
        sfdb_header_dict["value"] = _.reshape(_.shape[0])
        sfdb_header_df_2 = pandas.DataFrame(sfdb_header_dict, index=metadata_list)

        return pandas.concat([sfdb_header_df_1, sfdb_header_df_2])


print(load_sfdb(file_name))
"""
