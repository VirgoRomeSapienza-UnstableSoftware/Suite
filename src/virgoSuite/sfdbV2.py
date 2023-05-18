# Copyright (C) 2023  Riccardo Felicetti (riccardo.felicetti@infn.it)
#  under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# The SFDB09 file format (Short FFT DataBase, 2009 specification) is developed by Sergio Frasca and Ornella Piccinni
#
# The following Code is a derived and improved version of the software from the
# master's thesis work of Federico Muciaccia (federicomuciaccia@gmail.com),
# https://github.com/FedericoMuciaccia/RGBgw/tree/master/code.
#
# The function that reads the sfdbs is a porting from the code made by
# Pia Astone, defined inside the Snag Matlab package (written by Sergio Frasca).
#
# Snag is a Matlab data analysis toolbox oriented to gravitational-wave antenna data
# Snag webpage: http://grwavsf.roma1.infn.it/snag/
# version 2, released 12 May 2017
# installation instructions:
# http://grwavsf.roma1.infn.it/snag/Snag2_UG.pdf

import numpy as np
import pandas
import astropy.time

import os

from fnmatch import fnmatch
from pathlib import Path
from typing import TextIO

import random
import string

import h5py


def get_random_string(length: int) -> str:
    # choose from all lowercase letter
    letters = string.ascii_letters
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


# =============================================================================
# =============================================================================


def fread(fid: TextIO, n_elements: int, dtype: str) -> np.ndarray:
    """
    MatLab-like file reading

    A simple function that reproduces the behaviour of fread() function in matlab.
    It is used inside the VirgoSuite package to read SFDB files.

    Parameters
    ----------
    fid : TextIO
        The file to be read.
    nelements : int
        Number of elements to be read in sequence.
    dtype : type
        Type of the element to select. It is very important to explicitly
        pass the data type, this avoids incorrect readings.

    Returns
    -------
    data_array : numpy.ndarray
        A numpy.ndarray containing the values extracted.

    Examples
    --------
    Extracting an integer from example.SFDB09

    >>> fread("example.SFDB09", 2, np.int32)
    [[3], [15]]

    """
    if dtype is str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype
    data_array = np.fromfile(fid, dt, n_elements)

    if len(data_array) < n_elements:
        return None
    elif n_elements == 1:
        out_variable = data_array[0]
    else:
        out_variable = data_array

    if (dt == "int") or (dt == "int32"):
        return out_variable.astype("int64")
    elif dt in ["float32", "double", "float", "float64", "single"]:
        return out_variable.astype("double")


# =============================================================================
# =============================================================================


def read_block(fid: TextIO) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a block of FFTs

    `Snag <http://grwavsf.roma1.infn.it/snag/>`_ inspired function to read a
    block of FFTs from a SFDB file. The function will look for a list of values
    inside the file and will return them in an handy packed version.

    Parameters
    ----------
    fid : TextIO
        SFDB File

    Returns
    -------
    header : dict
        Header element of SFDB files, see for specifications.
    periodogram: np.ndarray of float
        The periodogram contained in SFDBs.
    autoregressive_spectrum : np.ndarray of float
        The autoregressive spectrum contained in SFDBs.
    fft_data : np.ndarray of complex
        The complex data used to compute the spectrum.

    Notes
    ----------------------

    A typical SFDB files contains a special set of entries, they are called header
    and contain a set of useful metadata.
    Due to the way SFDB files are generated, we are bound to read those values
    sequentially and to specify what those numbers are "Hard Coded", however this
    will not be a thing anymore in the :doc:`Suite</index>`.

    The content of the header is the following:

    * count : double
        A control variable # TODO To be understood!!
    * detector: int32
        The antenna used for the measurement
    * gps_seconds: int32
        Seconds of the gps time
    * gps_nanoseconds: int32
        Nanoseconds of the gps time
    * fft_lenght: double
        Old tbase
    * starting_fft_sample_index: int32
        Old firstfrind
    * unilateral_number_of_samples: int32
        Old nsamples
    * reduction_factor: int32
        Old red
    * fft_interlaced: int32
        Old type
    * number_of_flags: float32
        Old n_flag
    * scaling_factor: float32
        Old einstein
    * mjd_time: double
        Old mjdtime
    * fft_index: int32
        Old nfft
    * window_type: int32
        Old wink
    * normalization_factor: float32
        Old normd
    * window_normalization: float32
        Old normw
    * starting_fft_frequency: double
        Old frinit
    * subsampling_time: double
        Old tsamplu
    * frequency_resolution: double
        Old deltanu
    * v_x: double
        X coordinate of the velocity of the detector !TODO what frame
    * v_y: double
        Y coordinate of the velocity of the detector !TODO what frame
    * v_z: double
        Z coordinate of the velocity of the detector !TODO what frame
    * p_x: double
        X coordinate of the position of the detector !TODO what frame
    * p_y: double
        Y coordinate of the position of the detector !TODO what frame
    * p_z: double
        Z coordinate of the position of the detector !TODO what frame
    * number_of_zeroes: int32
        Old n_zeroes
    * sat_howmany: double
        Old sat_howmany !TODO not used anymore
    * spare1: double
        !TODO Not used
    * spare2: double
        !TODO Not used
    * spare3: double
        !TODO Not used
    * percentage_of_zeroes: float32
        Old spare4
    * spare5: float32
        !TODO Not used
    * spare6: float32
        !TODO Not used
    * lenght_of_averaged_time_spectrum: int32
        Old spare7
    * scientific_segment: int32
        Old spare8
    * spare9: int32
        !TODO Not used

    """
    count = fread(fid, 1, "double")  # count

    if count is None:
        return None, None, None, None

    det = fread(fid, 1, "int32")  # detector
    if det == 0:
        detector = "Nautilus"
    elif det == 1:
        detector = "Virgo"
    elif det == 2:
        detector = "Ligo Hanford"
    elif det == 3:
        detector = "Ligo Livingston"

    gps_seconds = fread(fid, 1, "int32")  # gps_sec
    gps_nanoseconds = fread(fid, 1, "int32")  # gps_nsec
    gps_time = gps_seconds + gps_nanoseconds * 1e-9

    fft_lenght = fread(fid, 1, "double")  # tbase
    starting_fft_sample_index = fread(fid, 1, "int32")  # firstfrind

    unilateral_number_of_samples = fread(fid, 1, "int32")  # nsamples
    reduction_factor = fread(fid, 1, "int32")  # red

    typ = fread(fid, 1, "int32")  # typ
    if typ == 1:
        fft_interlaced = True
    elif typ == 2:
        fft_interlaced = False

    # Number of data labeled with some kind of warning flag
    # (eg: non-science flag)
    number_of_flags = fread(fid, 1, "float32")  # n_flag

    # Scaling factor : 1e-20
    scaling_factor = fread(fid, 1, "float32")  # einstein

    # FFT starting time (using Modified Julian Date)
    # (computed using seconds and nanoseconds)
    mjd_time = fread(fid, 1, "double")  # mjdtime

    # Index in python start from 0
    fft_index = fread(fid, 1, "int32") - 1  # nfft

    # Window type used in FFT
    wink = fread(fid, 1, "int32")  # wink
    if wink == 0:
        window_type = "none"
    if wink == 1:
        window_type = "Hanning"
    if wink == 2:
        window_type = "Hamming"
    if wink == 3:
        window_type = "MAP"  # "Maria Alessandra Papa" time window, used at Ligo
    if wink == 4:
        window_type = "Blackmann flatcos"
    if wink == 5:
        window_type = "Flat top cosine edge"

    # normalization factor for the power spectrum extimated from
    # the square modulus of the FFT due to the data quantity
    # (sqrt(dt/nfft))
    normalization_factor = fread(fid, 1, "float32")  # normd
    # corrective factor due to power loss caused by the FFT window
    window_normalization = fread(fid, 1, "float32")  # normw

    starting_fft_frequency = fread(fid, 1, "double")  # frinit

    # sampling time used to obtain a given frequency band, subsampling the data
    subsampling_time = fread(fid, 1, "double")  # tsamplu

    frequency_resolution = fread(fid, 1, "double")  # deltanu

    if detector == "Nautilus":
        raise Exception("UNSUPPORTED DETECTOR")
    else:
        v_x = fread(fid, 1, "double")  # vx_eq
        v_y = fread(fid, 1, "double")  # vy_eq
        v_z = fread(fid, 1, "double")  # vz_eq
        x = fread(fid, 1, "double")  # px_eq
        y = fread(fid, 1, "double")  # py_eq
        z = fread(fid, 1, "double")  # pz_eq
        # number of artificial zeros, used to fill every time hole in the FFT (eg: non-science data)
        number_of_zeroes = fread(fid, 1, "int32")  # n_zeros

        # sat_howmany nowadays isn't used anymore: it was a saturation flag used in the early Virgo
        sat_howmany = fread(fid, 1, "double")  # sat_howmany

        spare1 = fread(fid, 1, "double")  # spare1
        spare2 = fread(fid, 1, "double")  # spare2
        spare3 = fread(fid, 1, "double")  # spare3
        percentage_of_zeroes = fread(fid, 1, "float32")  # spare4
        spare5 = fread(fid, 1, "float32")  # spare5
        spare6 = fread(fid, 1, "float32")  # spare6
        # lenght of the FFT divided in pieces by the reduction factor (128)
        lenght_of_averaged_time_spectrum = fread(fid, 1, "int32")  # lavesp

        # not used anymore
        scientific_segment = fread(fid, 1, "int32")  # spare8
        spare9 = fread(fid, 1, "int32")  # spare9

    header = {
        "_count": count,
        "detector": detector,
        "gps_seconds": gps_seconds,
        "gps_nanosecons": gps_nanoseconds,
        "gps_time": gps_time,
        "fft_lenght": fft_lenght,
        "starting_fft_sample_index": starting_fft_sample_index,
        "unilateral_number_of_samples": unilateral_number_of_samples,
        "reduction_factor": reduction_factor,
        "fft_interlaced": fft_interlaced,
        "number_of_flags": number_of_flags,
        "scaling_factor": scaling_factor,
        "mjd_time": mjd_time,
        "fft_index": fft_index,
        "window_type": window_type,
        "normalization_factor": normalization_factor,
        "window_normalization": window_normalization,
        "starting_fft_frequency": starting_fft_frequency,
        "subsampling_time": subsampling_time,
        "frequency_resolution": frequency_resolution,
        "position": np.array([x, y, z]),
        "velocity": np.array([v_x, v_y, v_z]),
        "number_of_zeroes": number_of_zeroes,
        "sat_howmany": sat_howmany,
        "spare1": spare1,
        "spare2": spare2,
        "spare3": spare3,
        "percentage_of_zeroes": percentage_of_zeroes,
        "spare5": spare5,
        "spare6": spare6,
        "lenght_of_averaged_time_spectrum": lenght_of_averaged_time_spectrum,
        "scientific_segment": scientific_segment,
        "spare9": spare9,
    }

    if lenght_of_averaged_time_spectrum > 0:
        lsps = lenght_of_averaged_time_spectrum
        # This was tps
        periodogram = fread(
            fid,
            lsps,
            "float32",
        )
    else:
        periodogram = fread(
            fid,
            reduction_factor,
            "float32",
        )
        lsps = unilateral_number_of_samples / reduction_factor

    # This was sps
    autoregressive_spectrum = fread(fid, lsps, "float32")

    # Inside sfdb complex number are saved as follows:
    # even -> Real part
    # odds -> Imaginary part
    # Since python can hadle complex numbers there is no reasons for that
    # This was calle sft
    _ = fread(
        fid,
        2 * unilateral_number_of_samples,
        "float32",
    )
    fft_data = _[0::2] + 1j * _[1::2]

    # Here is important to specify to export data as float, otherwise the
    # precision will not be enough
    return (
        header,
        periodogram,
        autoregressive_spectrum,
        fft_data.astype("complex64"),
    )


# =============================================================================
# =============================================================================


def sfdb_to_h5(
    path_to_sfdb_database: str,
    save_path: str,
) -> np.ndarray:
    """
    SFDB to netCDF4

    Contert SFDB files into netCDF4.
    netCDF4 is like hdf5 but hadles multidimensional data with more ease.

    Parameters
    ----------
    path_to_sfdb_database : str
        path to the file.
    save_path : str
        path to save folder

    """
    list_of_timestaps_in_block = []
    with open(path_to_sfdb_database) as fid:
        for i in range(200):
            # for i in range(10):  # For test only
            # Any sfdb file contains multiple ffts, check all ffts until EOF
            (
                head,
                periodogram,
                autoregressive_spectrum,
                fft_data,
            ) = read_block(fid)

            if head is None:
                # Checking EOF
                # This could happen in the case of an interruption in data
                break

            fft_frequencies = (
                np.arange(
                    start=0,
                    stop=len(fft_data),
                    step=1,
                    dtype=int,
                )
                * head["frequency_resolution"]
            )

            spectrum_frequencies = (
                np.arange(
                    start=0,
                    stop=len(periodogram),
                    step=1,
                    dtype=int,
                )
                * head["frequency_resolution"]
                * head["reduction_factor"]
            )

            total_normalization = (
                np.sqrt(2)
                * head["normalization_factor"]
                * head["window_normalization"]
                / np.sqrt(1 - head["percentage_of_zeroes"])
            )
            power_spectrum = np.square(np.abs(fft_data * total_normalization))
            power_spectrum = power_spectrum * head["scaling_factor"] ** 2

            # float64 slows down computation and cannot be handled by GPU
            # so we are forced to take into account the possibility of overflow
            # and truncation errors (RuntimeWarning: overflow)
            # replace the eventual infinities with the maximum float32 number
            power_spectrum[np.isinf(power_spectrum)] = np.finfo(
                np.float32
            ).max  # float32_max = 3.4028235e+38

            # autoregressive_spectrum and periodogram are stored in sfdbs
            # as square roots, so we need to make the square of them
            autoregressive_spectrum = np.square(
                autoregressive_spectrum * head["scaling_factor"]
            )
            periodogram = np.square(periodogram * head["scaling_factor"])

            # given the fact that out current data are really dirty, we place
            # a condition on the median of the autoregressive spectrum, to be sure
            # that it lies in the correct range.
            # the periodogram can be higher than the autoregressive spectrum, because
            # it suffers when there are bumps and unwanted impulses in the time domain
            # the median is more robust than the average
            #
            # autoregressive_spectrum_median = np.median(autoregressive_spectrum, axis=1)

            # autoregressive_spectrum and periodogram must be more or less the
            # same in this flat area they are different in the peaks, because by
            # construction the autoregressive mean ignores them
            # the autoregressive_spectrum can follow the noise nonstationarities
            #
            # periodogram_median = np.median(periodogram, axis=1)

            # HANDLING TIME
            gps_time = astropy.time.Time(
                val=head["gps_time"],
                format="gps",
                scale="utc",
            )
            # ISO 8601 compliant date-time format: YYYY-MM-DD HH:MM:SS.sss
            iso_time_value = gps_time.isot
            # time of the first FFT of this file
            human_readable_start_time = iso_time_value.replace(":", ".")
            datetimes = pandas.to_datetime(iso_time_value)

            list_of_timestaps_in_block.append(iso_time_value.encode("utf8"))

            # Setting compression level for FFT data files
            compression_level = 9

            # Creating the subfolder for the raw data and spectrum data
            data_save_path = save_path + "/data"
            spectrum_save_path = save_path + "/spectrum"

            Path(data_save_path).mkdir(mode=777, exist_ok=True, parents=True)
            Path(spectrum_save_path).mkdir(mode=777, exist_ok=True, parents=True)

            # Saving FFT Data in hdf files
            # =================================================================
            with h5py.File(
                f"{data_save_path}/{human_readable_start_time}.hdf5", mode="w"
            ) as data_file_obj:
                # Saving FFTs in hdf format with compression
                fft_dataset = data_file_obj.create_dataset(
                    "fft_data",
                    data=fft_data,
                    compression="gzip",
                    compression_opts=compression_level,
                )

                # Adding time dependent metadata
                for attribute_key, attribute in head.items():
                    fft_dataset.attrs[attribute_key] = attribute

            # Saving spectrum data
            with h5py.File(
                f"{spectrum_save_path}/{human_readable_start_time}.hdf5", mode="w"
            ) as spectrum_file_obj:
                # Spectrum data are stored inside "spectrum" group
                spectrum = spectrum_file_obj.create_group("spectrum")

                periodogram_dset = spectrum.create_dataset(
                    "periodogram", data=periodogram
                )
                ar_spectrum_dset = spectrum.create_dataset(
                    "autoregressive_spectrum", data=autoregressive_spectrum
                )

                # Adding time dependent metadata
                for attribute_key, attribute in head.items():
                    spectrum.attrs[attribute_key] = attribute

    # Adding a file with the frequencies of data
    with h5py.File(
        f"{data_save_path}/frequencies.hdf5", mode="w"
    ) as fft_frequencies_file:
        fft_frequencies_file.create_dataset(
            "frequencies",
            data=fft_frequencies,
            compression="gzip",
            compression_opts=compression_level,
        )
    # Adding a file with the frequencies of spectrum
    with h5py.File(
        f"{spectrum_save_path}/frequencies.hdf5", mode="w"
    ) as spectrum_frequencies_file:
        spectrum_frequencies_file.create_dataset(
            "frequencies",
            data=spectrum_frequencies,
        )

    return list_of_timestaps_in_block


# =============================================================================
# =============================================================================


def list_sfdb_in_directory(path: str) -> list:
    file_names = []
    # Check if a directory was given
    is_a_directory = os.path.isdir(path)
    if is_a_directory:
        for path, subdirs, files in os.walk(path):
            for name in files:
                if fnmatch(name, "*.SFDB09"):
                    file_names.append(os.path.join(path, name))

    return file_names


# =============================================================================
# =============================================================================


def convert_sfdb_database(
    path_to_sfdb_database: str,
    output_path: str,
) -> None:
    # First we check whether a directory or a file are provided
    is_a_File = os.path.isfile(path_to_sfdb_database)
    is_a_directory = os.path.isdir(path_to_sfdb_database)
    if (not is_a_File) and (not is_a_directory):
        raise TypeError("Please check the path to the sfdb database.")

    # Opening the file (files)
    if is_a_File:
        file_name_list = [path_to_sfdb_database]
    elif is_a_directory:
        file_name_list = list_sfdb_in_directory(path_to_sfdb_database)

    if len(file_name_list) == 1:
        print(f"{len(file_name_list)} file was found.")
    elif len(file_name_list) > 1:
        print(f"{len(file_name_list)} files were found.")

    print(f"Starting conversion...")

    timestamp_list_database = [0] * len(file_name_list)
    for j, file_name in enumerate(file_name_list):
        print(f"Processing : {file_name}")

        print("Extracting data from path...")
        print("Please make sure that the path to the files is:")
        print("/*/*/[detector]/sfdb/[run]/[calibration]/[cleaning]/*/[sfdb-files]")
        print("or")
        print("/*/*/[detector]/sfdb/[run]/[calibration]/*/[sfdb-files]")

        path_splitted = file_name.strip("\n").split("/")
        detector = str(path_splitted[path_splitted.index("sfdb") - 1])
        run = str(path_splitted[path_splitted.index("sfdb") + 1])
        calibration = str(path_splitted[path_splitted.index("sfdb") + 2])

        save_path = output_path + f"/{detector}/hdf5/{run}/{calibration}"

        if len(path_splitted) > 8:
            cleaning = str(path_splitted[path_splitted.index("sfdb") + 3])
            save_path = save_path + f"{cleaning}"

        Path(save_path).mkdir(mode=777, exist_ok=True, parents=True)
        timestap_list_block = sfdb_to_h5(file_name, save_path=save_path)
        timestamp_list_database[j] = timestap_list_block

    timestamp_list_database = np.concatenate(timestamp_list_database)
    with h5py.File(save_path + "/timeseries.hdf", "w") as timeseries_file:
        timeseries_file.create_dataset("times", data=timestamp_list_database)
