# NUMPY
import numpy

# DASK
import dask.array
from dask import delayed

# ASTROPY
import astropy

# PANDAS
import pandas

# XARRAY
import xarray

# STANDARD MODULES
from typing import TextIO
from os import walk
from os.path import isdir, isfile, join
from fnmatch import fnmatch

# All SFDB files have the same structure, so here we set a global variable for
# the reading

HEADER = [
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
    ("number_of_zeroes", "int32"),
    ("sat_howmany", "float64"),
    ("spare1", "float64"),
    ("spare2", "float64"),
    ("spare3", "float64"),
    ("percentage_of_zeroes", "float32"),
    ("spare5", "float32"),
    ("spare6", "float32"),
    ("lenght_of_averaged_time_spectrum", "int32"),
    ("scientific_segment", "int32"),
    ("spare9", "int32"),
]

HEADER_dtype = numpy.dtype(HEADER)


def list_sfdb_in_directory(path: str) -> list:
    file_names = []
    # Check if a directory was given
    for path, subdirs, files in walk(path):
        for name in files:
            if fnmatch(name, "*.SFDB09"):
                file_names.append(join(path, name))

    return file_names


def convert_sfdb(
    file_name: str | TextIO,
    fft_per_file: int = 200,
    verbose: int = 0,
) -> None:
    # Check if given path is a valid file or folder
    if isinstance(file_name, str) and isdir(file_name):
        if verbose > 0:
            print(f"\nLooking for .SFDB09 files inside {file_name}")

        file_list = list_sfdb_in_directory(file_name)

    elif isinstance(file_name, str) and isfile(file_name):
        if verbose > 0:
            print(f"\nLooking for {file_name}")
        file_list = [file_name]

    if verbose > 0:
        print(f"{len(file_list)} file(s) found")

    if verbose > 0:
        print("Opening files...")

    _header_database = []
    _periodogram_database = []
    _ar_spectrum_database = []
    _fft_spectrum_database = []

    # Begin cycling over all found files
    for sfdb_file_name in file_list:
        # Opening the first header to check for shape of spectrum, periodogram
        # and AR spectrum
        if verbose > 1:
            print(f"Checking header of {sfdb_file_name}")
        header = numpy.fromfile(sfdb_file_name, dtype=HEADER_dtype, count=1)

        lenght_of_averaged_time_spectrum = header["lenght_of_averaged_time_spectrum"]
        reduction_factor = header["reduction_factor"]
        unilateral_number_of_samples = header["unilateral_number_of_samples"]

        if lenght_of_averaged_time_spectrum > 0:
            periodogram_shape = lenght_of_averaged_time_spectrum
            ar_spectrum_shape = periodogram_shape
        else:
            periodogram_shape = reduction_factor
            ar_spectrum_shape = int(unilateral_number_of_samples / reduction_factor)

        spectrum_shape = 2 * unilateral_number_of_samples

        # Creating a custom dtype to read sfdb files
        if verbose > 1:
            print(f"Opening {sfdb_file_name}")
        sfdb_dtype = numpy.dtype(
            [
                ("header", HEADER),
                ("periodogram", "float32", periodogram_shape),
                ("autoregressive_spectrum", "float32", ar_spectrum_shape),
                ("fft_spectrum", "float32", spectrum_shape),
            ]
        )

        # Here the magic happens: opening sfdb file
        sfdb = dask.array.from_array(
            numpy.fromfile(sfdb_file_name, dtype=sfdb_dtype, count=fft_per_file),
            chunks=1,
        )
        _header_database.append(sfdb["header"].rechunk("auto"))
        _periodogram_database.append(sfdb["periodogram"].rechunk("auto"))
        _ar_spectrum_database.append(sfdb["autoregressive_spectrum"].rechunk("auto"))
        _fft_spectrum_database.append(sfdb["fft_spectrum"].rechunk("auto"))

    header_database = dask.array.concatenate(_header_database, axis=0).rechunk("auto")
    periodogram_database = dask.array.concatenate(
        _periodogram_database, axis=0
    ).rechunk("auto")
    ar_spectrum_database = dask.array.concatenate(
        _ar_spectrum_database, axis=0
    ).rechunk("auto")
    fft_spectrum_database = dask.array.concatenate(
        _fft_spectrum_database, axis=0
    ).rechunk(chunks=(1, -1))
    complex_fft_spectrum_database = (
        fft_spectrum_database[:, 0::2] + 1j * fft_spectrum_database[:, 1::2]
    )

    # Extracting frequency information from sfdb
    # !TODO qual'è il modo migliore di considerare diverse frequency resolutions?
    spectrum_frequency_index = dask.array.arange(
        0, complex_fft_spectrum_database.shape[1], 1, dtype="int32"
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
    gps_time_int = (
        header_database["gps_seconds"] + header_database["gps_nanoseconds"] * 1e-9
    )
    gps_time = delayed(
        astropy.time.Time(
            gps_time_int,
            format="gps",
            scale="utc",
        )
    )
    iso_time_values = gps_time.iso
    datetimes = pandas.to_datetime(iso_time_values.compute())

    # Saving to Zarr
    # !TODO ATTRIBUTES
    coordinates_names = ["time", "frequency"]

    spectrum_coordinate_values = [datetimes, spectrum_frequencies]
    periodogram_coordinate_values = [datetimes, periodogram_frequencies]

    spectrum = xarray.DataArray(
        data=complex_fft_spectrum_database,
        dims=coordinates_names,
        coords=spectrum_coordinate_values,
    )
    periodogram = xarray.DataArray(
        data=periodogram_database,
        dims=coordinates_names,
        coords=periodogram_coordinate_values,
    )
    ar_spectrum = xarray.DataArray(
        data=ar_spectrum_database,
        dims=coordinates_names,
        coords=periodogram_coordinate_values,
    )

    fft_data = xarray.Dataset(
        data_vars={
            "fft_complex_spectrum": spectrum.where(spectrum != 0).astype("complex64"),
        }
    )

    period_arSpectrum = xarray.Dataset(
        data_vars={
            "periodogram": periodogram.where(periodogram != 0),
            "autoregressive_spectrum": ar_spectrum.where(ar_spectrum != 0),
        }
    )

    return fft_data, period_arSpectrum
