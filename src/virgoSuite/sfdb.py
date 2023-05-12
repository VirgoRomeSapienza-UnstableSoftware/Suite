# Copyright (C) 2023  Riccardo Felicetti
#  under GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

# %%
import numpy as np
import pandas
import astropy.time
import xarray

import matplotlib.pyplot as plt


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

    Returns
    *******
        data_array : numpy.ndarray
            A numpy.ndarray containing the values extracted

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


def read_block(fid) -> list:
    count = fread(fid, 1, "double")  # count

    det = fread(fid, 1, "int32")  # detector
    if det == 0:
        detector = "Nautilus"
    elif det == 1:
        detector = "Virgo"
    elif det == 2:
        detector = "LIGO Hanford"
    elif det == 3:
        detector = "LIGO Livingston"

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
        "x": x,
        "y": y,
        "z": z,
        "v_x": v_x,
        "v_y": v_y,
        "v_z": v_z,
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
        "spare1": spare1,
    }

    header = pandas.DataFrame.from_dict([header])

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

    return header, periodogram, autoregressive_spectrum, fft_data


def load_file_sfdb(path_to_sfdb: str, save_path: str) -> pandas.DataFrame:
    """
    SFDB to netCDF4

    Contert SFDB files into netCDF4.
    netCDF4 is like hdf5 but hadles multidimensional data with more ease.

    Parameters
    **********
        path_to_sfdb : str
            path to the file.
        save_path : str
            path to save folder

    """

    # SFDB files come with several attributes.
    # We want to separate attributes, metadata and data to ensure readability
    # and ease of use.

    with open(path_to_sfdb) as fid:
        for i in range(100):
            head, tps, sps, sft = read_block(fid)
            if i == 0:
                header = head
                periodogram = tps
                autoregressive_spectrum = sps
                fft_data = sft
            else:
                header = pandas.concat([header, head], ignore_index=True)
                periodogram = np.vstack((periodogram, tps))
                autoregressive_spectrum = np.vstack((autoregressive_spectrum, sps))
                fft_data = np.vstack((fft_data, sft))

    # ============================================
    # Preparing data to be saved in the new format
    # ============================================
    """

    This part needs some study with Federico

    """

    minimum_frequency = header.starting_fft_frequency[0]  # 0 Hz
    # This here is hard coded.
    # We can discuss if we can obtain it from the time of subsampling
    # Questo non mi sembra giusto!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    maximum_frequency = 128
    frequency_interval = maximum_frequency - minimum_frequency
    frequencies = np.linspace(
        start=minimum_frequency,
        stop=maximum_frequency,
        num=int(frequency_interval / header.frequency_resolution[0]),
    )
    frequencies = (
        np.arange(
            start=0,
            stop=periodogram.shape[0],
            step=1,
            dtype=int,
        )
        * header.frequency_resolution[0]
        * header.reduction_factor[0]
    )
    subsampled_frequencies = np.linspace(
        start=minimum_frequency,
        stop=maximum_frequency,
        num=int(
            frequency_interval
            / (header.frequency_resolution[0] * header.reduction_factor[0]),
        ),
    )

    total_normalization = (
        np.sqrt(2)
        * header.normalization_factor
        * header.window_normalization
        * np.sqrt(1 - header.percentage_of_zeroes)
    )
    power_spectrum = np.square(
        np.abs(np.einsum("ij, i -> ij", fft_data, total_normalization))
    )
    power_spectrum = np.einsum("ij, i -> ij", power_spectrum, header.scaling_factor)

    # float64 slows down computation and cannot be handled by GPU
    # so we are forced to take into account the possibility of overflow
    # and truncation errors (RuntimeWarning: overflow)
    # replace the eventual infinities with the maximum float32 number
    power_spectrum[np.isinf(power_spectrum)] = np.finfo(
        np.float32
    ).max  # float32_max = 3.4028235e+38

    # autoregressive_spectrum and periodogram are stored in sfdbs
    # as square roots, so we need to make the square of them
    autoregressive_spectrum = np.einsum(
        "ij, i -> ij", np.square(autoregressive_spectrum), header.scaling_factor
    )
    periodogram = np.einsum(
        "ij, i -> ij", np.square(periodogram), header.scaling_factor
    )

    # untill now, we have filtered and selected frequencies. so it was
    # useful to have the main axis of the matrices on the dimension
    # "frequency" from here on, we will need to iterate over "time".
    # so it's useful to transpose everything
    power_spectrum = np.transpose(power_spectrum)
    autoregressive_spectrum = np.transpose(autoregressive_spectrum)
    periodogram = np.transpose(periodogram)

    # given the fact that out current data are really dirty, we place
    # a condition on the median of the autoregressive spectrum, to be sure
    # that it lies in the correct range.
    # the periodogram can be higher than the autoregressive spectrum, because
    # it suffers when there are bumps and unwanted impulses in the time domain
    # the median is more robust than the average
    autoregressive_spectrum_median = np.median(autoregressive_spectrum, axis=1)

    # autoregressive_spectrum and periodogram must be more or less the
    # same in this flat area they are different in the peaks, because by
    # construction the autoregressive mean ignores them
    # the autoregressive_spectrum can follow the noise nonstationarities
    periodogram_median = np.median(periodogram, axis=1)

    # HANDLING TIME
    gps_time = astropy.time.Time(
        val=header.gps_time,
        format="gps",
        scale="utc",
    )
    gps_time_values = gps_time.value.astype(np.float64)
    # ISO 8601 compliant date-time format: YYYY-MM-DD HH:MM:SS.sss
    iso_time_values = gps_time.iso
    # time of the first FFT of this file
    human_readable_start_time = iso_time_values[0]
    datetimes = pandas.to_datetime(iso_time_values)

    # ================
    # Saving to xarray
    # ================
    print(frequencies.shape)
    print(datetimes.shape)
    print(header.detector.shape)
    print(power_spectrum.shape)
    coordinate_names = ["frequency", "time", "detector"]
    coordinate_values = [frequencies, datetimes, header.detector]
    attributes = {
        "FFT_lenght": header.fft_lenght,
        "observing_run": "O3",  # TODO Remove hard coding
        "calibration": "C01",  # TODO Remove hard coding
        "maximum_frequency": maximum_frequency,  # TODO hardcoded
        "start_ISO_time": human_readable_start_time,
        # TODO Take back all the attributes in the sfdb!!!!
    }

    spectrogram = xarray.DataArray(
        data=np.expand_dims(np.transpose(power_spectrum), axis=1),
        dims=coordinate_names,
        coords=coordinate_values,
    )
    dataset = xarray.Dataset(
        data_vars={
            "spectrogram": spectrogram,
        },
        attrs=attributes,
    )

    return dataset
