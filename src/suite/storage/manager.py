# TODO: choose a license
#
# TODO: funzioni per leggere tutti i tipi di file supportati
#           // load_sfdb09(**args, **kwargs) -> xarra.array.Dataset
#           -/ load_zarr(**args, **kwargs) -> xarra.array.Dataset
#           -/ load_hdf5(**args, **kwargs) -> xarra.array.Dataset
#
# TODO: funtion to load in memory from any file
#           -/ load_data(**args, **kwargs) -> xarra.array.Dataset
#
# TODO: funtion to convert entire database from any format to any other
#           -/ convert_database(**args, **kwargs) -> None
#
# TODO: funtion to extract data from any format to any format
#           -/ extract_data(**args, **kwargs) -> None
#
# TODO: function to save loaded data to file
# TODO: ATTENZIONE! SALVARE PER POI CARICARE FILE IN ZARR ALTERA I FLOAT!!!

"""
    The choice was to use Dask and Xarray as a backend for managing data. 
    The user will be free to load, save, operate and convert from/to any supported
    format.
    Supported formats will be SFDB09, ZARR, HDF5, ZIP(?)
"""

# NUMPY
import numpy
from numpy.typing import NDArray

# DASK
import dask.array
from dask.delayed import delayed

# ASTROPY
from astropy import time

# PANDAS
import pandas

# XARRAY
import xarray

# ZARR
import zarr

# STANDARD MODULES
from typing import TextIO, NamedTuple
from os import walk
from os.path import isdir, isfile, join
from fnmatch import fnmatch
from dataclasses import dataclass


# =============================================================================
# *****************************************************************************
# =============================================================================
# TODO: SCRIVERE QUESTE
@dataclass
class Position:
    ...


@dataclass
class Velocity:
    ...


@dataclass
class Header:
    count: numpy.float64
    detector: numpy.int32
    fft_lenght: numpy.float64
    starting_fft_sample_index: numpy.int32
    unilatera_number_of_samples: numpy.int32
    reduction_factor: numpy.int32
    fft_interlaced: numpy.int32
    number_of_flags: numpy.float32
    scaling_factor: numpy.float32
    fft_index: numpy.int32
    window_type: numpy.int32
    normalization_factor: numpy.float32
    window_normalization: numpy.float32
    starting_fft_frequency: numpy.float64
    subsampling_time: numpy.float64
    frequency_resolution: numpy.float64
    number_of_zeros: numpy.int32
    sat_howmany: numpy.float64
    percentage_of_zeros: numpy.float32
    lenght_of_averaged_time_spectrum: numpy.int32
    scientific_segment: numpy.int32

    # TODO: AGGIUNGERE LE PROPERTIES
    # sampling rates, nyquist, coherence time, half coherence time (time delta between interlaced coherence time)
    # TODO: ASSERT, CONTROLLARE CHE LE PROPERTIES COINCIDANO CON QUELLO CHE SI INSERISCE
    # TODO: COMPUTE CHUNKSIZE


# NOTE: Using named tuple for "read-only" provileges
class Header_Pia(NamedTuple):
    # DOCUMENT THIS
    ...


# =============================================================================
# *****************************************************************************
# =============================================================================


def list_files_in_directory(path: str, file_type: str) -> list:
    # DOCUMENT THIS
    file_names = []
    # Check if a directory was given
    for path, subdirs, files in walk(path):
        for name in files:
            if fnmatch(name, "*." + file_type):
                file_names.append(join(path, name))

    return file_names


# =============================================================================


def header_to_human(value: any, attribute_name: str) -> str:
    # TODO: SEPARATE TO 3 FUNCTIONS
    # DOCUMENT THIS
    supported = [
        "detector",
        "fft_interlaced",
        "window_type",
    ]
    if attribute_name not in supported:
        raise ValueError(f"{attribute_name} is not a valid option")
    if attribute_name == "detector":
        if value == 0:
            return "Nautilus"
        elif value == 1:
            return "Virgo"
        elif value == 2:
            return "Ligo Hanford"
        elif value == 3:
            return "Ligo Livingston"

    elif attribute_name == "fft_interlaced":
        if value == 1:
            return "Half interlaced"
        elif value == 2:
            return "Not interlaced"

    elif attribute_name == "window_type":  # wink
        if value == 0:
            return "None"
        elif value == 1:
            return "Hanning"
        elif value == 2:
            return "Hamming"
        elif value == 3:
            return "Maria Aless Papa"
        elif value == 4:
            return "Blackmann flatcos"
        elif value == 5:
            return "Flat top cosine edge"


# =============================================================================


def load_sfdb09(file_name: str | TextIO, verbose: int = 0) -> list:
    # DOCUMENT THIS
    # Check if given path is a valid file or folder
    file_list = []
    if isinstance(file_name, str) and isdir(file_name):
        if verbose > 0:
            print(f"\nLooking for .SFDB09 files inside {file_name}")

        file_list = list_files_in_directory(file_name, "SFDB09")

    elif isinstance(file_name, str) and isfile(file_name):
        if verbose > 0:
            print(f"\nLooking for {file_name}")
        file_list = [file_name]

    if verbose > 0:
        print(f"{len(file_list)} file(s) found")

    if verbose > 0:
        print("Opening files...")

    # -------------------------------------------------------------------------

    _header_database = []
    _periodogram_database = []
    _ar_spectrum_database = []
    _fft_spectrum_database = []

    # All SFDB files have the same structure, so here we set a global variable for
    # the reading
    header_dtype_arr = [
        ("count", "float64"),
        ("detector", "int32"),
        ("gps_seconds", "int32"),
        ("gps_nanoseconds", "int32"),
        ("fft_lenght", "float64"),
        ("starting_fft_sample_index", "int32"),
        ("unilateral_number_of_samples", "int32"),
        ("reduction_factor", "int32"),
        ("fft_interlaced", "int32"),  # typ
        ("number_of_flags", "float32"),
        ("scaling_factor", "float32"),
        ("mjd_time", "float64"),
        ("fft_index", "int32"),
        ("window_type", "int32"),  # wink
        ("normalization_factor", "float32"),
        ("window_normalization", "float32"),
        ("starting_fft_frequency", "float64"),
        ("subsampling_time", "float64"),
        ("frequency_resolution", "float64"),
        ("position_x", "float64"),
        ("position_y", "float64"),
        ("position_z", "float64"),
        ("velocity_x", "float64"),
        ("velocity_y", "float64"),
        ("velocity_z", "float64"),
        ("number_of_zeros", "int32"),
        ("sat_howmany", "float64"),
        ("spare1", "float64"),
        ("spare2", "float64"),
        ("spare3", "float64"),
        ("percentage_of_zeros", "float32"),
        ("spare5", "float32"),
        ("spare6", "float32"),
        ("lenght_of_averaged_time_spectrum", "int32"),
        ("scientific_segment", "int32"),
        ("spare9", "int32"),
        # fft_data, periodogram and ARSpectrum have variable lengths
    ]

    header_dtype = numpy.dtype(header_dtype_arr)

    # Checking if datasets of different shapes were loaded
    # In case loading is aborted, this is not very efficient
    list_of_first_header = []
    if verbose > 2:
        print(
            f"Opening first header of each file to detect if there are databases with different attributes."
        )
    for sfdb_file_name in file_list:
        first_header = numpy.fromfile(sfdb_file_name, dtype=header_dtype, count=1)
        list_of_first_header.append(first_header)

    is_unique = []
    unique_list_str = numpy.array(
        [
            "detector",
            "fft_lenght",
            "window_type",
            "lenght_of_averaged_time_spectrum",
            "fft_interlaced",
            "scientific_segment",
        ]
    )
    list_of_first_header = numpy.array(list_of_first_header)
    for item in unique_list_str:
        is_unique.append(len(numpy.unique(list_of_first_header[:][item])) == 1)

    if numpy.any(numpy.logical_not(is_unique)):
        raise ValueError(
            f"\
                \nGiven path contains multiple databases with different T_fft.\
                \nPlease select a path with SFDB of fixed lenght.\
                \n\nIn particular look for unique {unique_list_str[numpy.logical_not(is_unique)]}"
        )

    # -------------------------------------------------------------------------

    # Extracting human-readable common attributes
    detector = header_to_human(first_header["detector"], "detector")
    window_type = header_to_human(first_header["window_type"], "window_type")
    fft_interlaced = header_to_human(first_header["fft_interlaced"], "fft_interlaced")

    # If no problem was found on the files to load, the process starts.
    # Begin cycling over all found files

    # TODO: APRIRE TUTTI GLI HEADER E ALLOCARE LA MEMORIA, FARE INOLTRE TUTTI I
    # TODO: SANITY CHECK SUL CONTENUTO DEGLI HEADER

    for sfdb_file_name in file_list:
        # Opening the first header to check for shape of spectrum, periodogram
        # and AR spectrum
        if verbose > 1:
            print(f"Checking header of {sfdb_file_name}")
        header = numpy.fromfile(sfdb_file_name, dtype=header_dtype, count=1)

        lenght_of_averaged_time_spectrum = header["lenght_of_averaged_time_spectrum"]
        reduction_factor = header["reduction_factor"]
        unilateral_number_of_samples = header["unilateral_number_of_samples"]

        if lenght_of_averaged_time_spectrum > 0:
            periodogram_shape = lenght_of_averaged_time_spectrum
            ar_spectrum_shape = periodogram_shape
        else:
            periodogram_shape = reduction_factor
            ar_spectrum_shape = int(unilateral_number_of_samples / reduction_factor)

        spectrum_shape = unilateral_number_of_samples
        # Creating a custom dtype to read sfdb files
        if verbose > 1:
            print(f"Opening {sfdb_file_name}")
        sfdb_dtype = numpy.dtype(
            [
                ("header", header_dtype_arr),
                ("periodogram", "float32", periodogram_shape),
                ("autoregressive_spectrum", "float32", ar_spectrum_shape),
                ("fft_spectrum", "complex64", spectrum_shape),
            ]
        )

        # Here the magic happens: opening sfdb file
        sfdb = dask.array.from_array(
            numpy.fromfile(
                file=sfdb_file_name,
                dtype=sfdb_dtype,
            ),
            chunks=1,
        )

        _header_database.append(sfdb["header"])
        _periodogram_database.append(sfdb["periodogram"])
        _ar_spectrum_database.append(sfdb["autoregressive_spectrum"])
        _fft_spectrum_database.append(sfdb["fft_spectrum"])

    # No more needed, this is to avoid errors in programming
    del header

    # We want the header to be immediately computed, so that the resulting dataset
    # has all the useful informations.
    header_database = (
        dask.array.concatenate(_header_database, axis=0).rechunk("auto").compute()
    )

    periodogram_database = dask.array.concatenate(_periodogram_database, axis=0)
    ar_spectrum_database = dask.array.concatenate(_ar_spectrum_database, axis=0)

    # -------------------------------------------------------------------------
    # TODO: LA CLASSE HEADER DEVE FARE QUESTO CONTO
    sampling_rate = 1 / header_database[0]["subsampling_time"]
    nyquist = sampling_rate / 2
    coherence_time = 1 / header_database[0]["frequency_resolution"]
    samples_per_hertz = ((coherence_time * sampling_rate) / 2) / nyquist

    frequency_chunk_size = 64 * samples_per_hertz
    # TODO: CI SONO 200/100 FFT PER FILE, OGNUNO DISTA 512 DA QUELLO DOPO
    # every chunk is 2 ** 8 * coherence_time / 2 seconds
    time_chunk_size = 64
    # -------------------------------------------------------------------------

    fft_spectrum_database = dask.array.concatenate(
        _fft_spectrum_database, axis=0
    ).rechunk(
        chunks=(time_chunk_size, frequency_chunk_size),
        # chunks=(1, -1)
    )

    # Extracting frequency information from sfdb
    spectrum_frequency_index = dask.array.arange(
        0, fft_spectrum_database.shape[1], 1, dtype="int32"
    )
    spectrum_frequencies = (
        header_database["frequency_resolution"][0] * spectrum_frequency_index
    )

    periodogram_frequency_index = dask.array.arange(
        0, periodogram_database.shape[1], 1, dtype="int32"
    )
    periodogram_frequencies = (
        periodogram_frequency_index
        * header_database["frequency_resolution"][0]
        * header_database["reduction_factor"][0]
    )

    # Extracting times
    _gps_time = (
        header_database["gps_seconds"] + header_database["gps_nanoseconds"] * 1e-9
    )
    gps_time = delayed(
        time.Time(
            _gps_time,
            format="gps",
            scale="utc",
        )
    )
    iso_time_values = gps_time.iso
    datetimes = pandas.to_datetime(iso_time_values.compute())

    # Saving to Xarray and Datasets
    # TODO: INVERTIRE ASSI
    # coordinates_names = ["time", "frequency"]
    coordinates_names = ["frequecy", "time"]

    # TODO: SI PUO' METTERLI COME DATAARRAY E FARE LO SCLICING SUL DATASET
    coordinate_values = dict(
        time=datetimes,
        x=(("time"), header_database["position_x"]),
        y=(("time"), header_database["position_y"]),
        z=(("time"), header_database["position_z"]),
        v_x=(("time"), header_database["velocity_x"]),
        v_y=(("time"), header_database["velocity_y"]),
        v_z=(("time"), header_database["velocity_z"]),
    )

    spectrum_coordinate_values = {
        **coordinate_values,
        "frequency": spectrum_frequencies,
    }
    periodogram_coordinate_values = {
        **coordinate_values,
        "frequency": periodogram_frequencies,
    }

    # the attributes will be shared between alla datasets, they contain the time
    # independent values of the header
    # TODO: to_dict() DA DTYPE CUSTOM (usare le named tuples)
    attributes = dict(
        count=header_database["count"][0],
        detector=detector,
        fft_lenght=header_database["fft_lenght"][0],
        starting_fft_sample_index=header_database["starting_fft_sample_index"][0],
        unilateral_number_of_samples=header_database["unilateral_number_of_samples"][0],
        reduction_factor=header_database["reduction_factor"][0],
        fft_interlaced=fft_interlaced,
        scaling_factor=header_database["scaling_factor"][0],
        window_type=window_type,
        normalization_factor=header_database["normalization_factor"][0],
        window_normalization=header_database["window_normalization"][0],
        starting_fft_frequency=header_database["starting_fft_frequency"][0],
        subsampling_time=header_database["subsampling_time"][0],
        frequency_resolution=header_database["frequency_resolution"][0],
        sat_howmany=header_database["sat_howmany"][0],
        lenght_of_averaged_time_spectrum=header_database[
            "lenght_of_averaged_time_spectrum"
        ][0],
        scientific_segment=header_database["scientific_segment"][0],
    )

    spectrum = xarray.DataArray(
        data=fft_spectrum_database.transpose(),
        dims=coordinates_names,
        coords=[spectrum_frequencies, datetimes],
        attrs=attributes,
    )
    periodogram = xarray.DataArray(
        data=periodogram_database.transpose(),
        dims=coordinates_names,
        coords=[periodogram_frequencies, datetimes],
        attrs=attributes,
    )
    ar_spectrum = xarray.DataArray(
        data=ar_spectrum_database.transpose(),
        dims=coordinates_names,
        coords=[periodogram_frequencies, datetimes],
        attrs=attributes,
    )

    # Building the dataset
    # TODO: DOVE LI PIJO I BUCHI?
    fft_data = xarray.Dataset(
        data_vars={
            "spectrum": spectrum.where(spectrum != 0).astype("complex64"),
        },
        attrs=attributes,
    )
    regressive_data = xarray.Dataset(
        data_vars={
            "periodogram": periodogram.where(periodogram != 0),
            "autoregressive_spectrum": ar_spectrum.where(ar_spectrum != 0),
        },
        attrs=attributes,
    )

    return (fft_data, regressive_data)


def scan_database() -> dask.array:
    ...


def load_database():
    return scan_database.compute()


def convert_database():
    ...


def slice_database():
    ...
