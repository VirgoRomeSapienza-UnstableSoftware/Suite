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
from numpy.typing import NDArray, ArrayLike

# DASK
import dask.array
import dask.dataframe
from dask.distributed import Client
from dask.delayed import delayed

# ASTROPY
from astropy import time

# PANDAS
import pandas

# XARRAY
import xarray

# STANDARD MODULES
from typing import TextIO, NamedTuple
from os import walk
from os.path import isdir, isfile, join, getsize
from fnmatch import fnmatch
from dataclasses import dataclass, field, asdict
from time import time as t
from itertools import groupby


# =============================================================================
# *****************************************************************************
# =============================================================================

# All SFDB files have the same structure, so here we set a global variable for
# the reading
HEADER_ELEMENTS = [
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
    ("spare_1", "float64"),
    ("spare_2", "float64"),
    ("spare_3", "float64"),
    ("percentage_of_zeros", "float32"),
    ("spare_5", "float32"),
    ("spare_6", "float32"),
    ("lenght_of_averaged_time_spectrum", "int32"),
    ("scientific_segment", "int32"),
    ("spare_9", "int32"),
    # fft_data, periodogram and ARSpectrum have variable lengths
]

HEADER_DTYPE = numpy.dtype(HEADER_ELEMENTS)


# =============================================================================
# *****************************************************************************
# =============================================================================


@dataclass
class Vector3D:
    x: NDArray[numpy.float64]
    y: NDArray[numpy.float64]
    z: NDArray[numpy.float64]


# TODO: NON FA QUELLO CHE DOVREBBE
@delayed
def create_delayed_Vector3D(
    x: NDArray[numpy.float64], y: NDArray[numpy.float64], z: NDArray[numpy.float64]
):
    return Vector3D(x, y, z)


@dataclass
class Header:
    # voglio dividere gli argomenti unici da quelli time-dependant
    # inserisco anche termini di consistency
    # vedere study_of_header.ipynb per la distinzione tra indipendenti e dipendenti
    # time_independant_args
    count: list[numpy.float64]
    detector: list[numpy.int32]
    gps_nanoseconds: list[numpy.int32]
    fft_lenght: list[numpy.float64]
    starting_fft_sample_index: list[numpy.int32]
    unilateral_number_of_samples: list[numpy.int32]
    reduction_factor: list[numpy.int32]
    fft_interlaced: list[numpy.int32]
    scaling_factor: list[numpy.float32]
    window_type: list[numpy.int32]
    normalization_factor: list[numpy.float32]
    window_normalization: list[numpy.float32]
    starting_fft_frequency: list[numpy.float64]
    subsampling_time: list[numpy.float64]
    frequency_resolution: list[numpy.float64]
    sat_howmany: list[numpy.float64]
    spare_1: list[numpy.float64]
    spare_2: list[numpy.float64]
    spare_3: list[numpy.float64]
    spare_5: list[numpy.float32]
    spare_6: list[numpy.float32]
    lenght_of_averaged_time_spectrum: list[numpy.int32]
    scientific_segment: list[numpy.int32]
    spare_9: list[numpy.int32]
    # time_dependant_args
    gps_seconds: list[numpy.int32]
    number_of_flags: list[numpy.float32]
    fft_index: list[numpy.int32]
    mjd_time: list[numpy.float64]
    position_x: list[numpy.float64]
    position_y: list[numpy.float64]
    position_z: list[numpy.float64]
    velocity_x: list[numpy.float64]
    velocity_y: list[numpy.float64]
    velocity_z: list[numpy.float64]
    number_of_zeros: list[numpy.int32]
    percentage_of_zeros: list[numpy.float32]

    detector_name: str = field(init=False)
    window_normalization_name: str = field(init=False)
    fft_interlaced_name: str = field(init=False)
    samples_per_hertz: numpy.int32 = field(init=False)

    time_ind_args = [
        "count",
        "detector",
        "fft_lenght",
        "starting_fft_sample_index",
        "unilateral_number_of_samples",
        "reduction_factor",
        "fft_interlaced",
        "scaling_factor",
        "window_type",
        "normalization_factor",
        "window_normalization",
        "starting_fft_frequency",
        "subsampling_time",
        "frequency_resolution",
        "sat_howmany",
        "spare_1",
        "spare_2",
        "spare_3",
        "spare_5",
        "spare_6",
        "lenght_of_averaged_time_spectrum",
        "scientific_segment",
        "spare_9",
    ]
    time_dep_args = [
        "gps_seconds",
        "number_of_flags",
        "gps_nanoseconds",
        "mjd_time",
        "fft_index",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "number_of_zeroes",
        "percentage_of_zeroes",
    ]

    def __post_init__(self):
        # Checking for consistency
        #
        # Assure that time independant variables are unique
        # Than substituting the lists with nunmbers
        for arg in self.time_ind_args:
            attribute = getattr(self, arg)
            assert all_equal(attribute), f"{arg} is not unique"
            setattr(self, arg, attribute[0])

        # Creating human-readable attributes
        self.detector_name = extract_detector(self.detector)
        self.window_normalization_name = extract_window_type(self.window_type)
        self.fft_interlaced_name = extract_interlace_method(self.fft_interlaced)

        # DOCUMENT THIS: THIS NEEDS A DEEP EXPLANATION
        sampling_rate = 1 / self.subsampling_time
        nyquist = sampling_rate / 2
        coherence_time = 1 / self.frequency_resolution
        self.samples_per_hertz = int(((coherence_time * sampling_rate) / 2) / nyquist)

        # Consistency check
        assert coherence_time == self.fft_lenght, f"Coherence time is inconsistent"
        assert (
            int(coherence_time * sampling_rate / 2) == self.unilateral_number_of_samples
        ), f"Number of samples is inconsistent"

    @property
    def attributes(self):
        self.time_ind_args.extend(
            [
                "detector_name",
                "window_normalization_name",
                "fft_interlaced_name",
                "samples_per_hertz",
            ]
        )
        return {key: getattr(self, key) for key in self.time_ind_args}


# =============================================================================
# *****************************************************************************
# =============================================================================


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def list_files_in_directory(path: str, file_type: str = "SFDB09"):
    # DOCUMENT THIS
    file_names = []
    # Check if a directory was given
    for path, subdirs, files in walk(path):
        for name in files:
            if fnmatch(name, "*." + file_type):
                file_names.append(join(path, name))

    if len(file_names) == 0:
        raise ImportError(f"Given path does not contain any SFDB09 file")

    return file_names


# =============================================================================


def extract_detector(numerical: numpy.int32):
    if numerical == 0:
        return "Nautilus"
    elif numerical == 1:
        return "Virgo"
    elif numerical == 2:
        return "Ligo Hanford"
    elif numerical == 3:
        return "Ligo Livingston"
    else:
        raise ValueError("Unsupported detector")


def extract_window_type(numerical: numpy.int32):
    if numerical == 0:
        return "None"
    elif numerical == 1:
        return "Hanning"
    elif numerical == 2:
        return "Hamming"
    elif numerical == 3:
        return "Maria A. Papa"
    elif numerical == 4:
        return "Blackmann flatcos"
    elif numerical == 5:
        return "Flat top cosine edge"
    else:
        raise ValueError("Unsupported window type")


def extract_interlace_method(numerical: numpy.int32):
    if numerical == 1:
        return "Half interlaced"
    elif numerical == 2:
        return "Not interlaced"
    else:
        raise ValueError("Unsupported interlacing method")


def extract_periodogram_shape(
    lenght_of_averaged_time_spectrum: numpy.int32, reduction_factor: numpy.int32
):
    # DOCUMENT THIS!
    if lenght_of_averaged_time_spectrum > 0:
        return lenght_of_averaged_time_spectrum
    else:
        return reduction_factor


def extract_arSpectrum_shape(
    lenght_of_averaged_time_spectrum: numpy.int32,
    unilateral_number_of_samples: numpy.int32,
    reduction_factor: numpy.int32,
):
    # DOCUMENT THIS!
    if lenght_of_averaged_time_spectrum > 0:
        return lenght_of_averaged_time_spectrum
    else:
        return int(unilateral_number_of_samples / reduction_factor)


# =============================================================================


# Returns delayed obj containing all the first headers of list of files
def scan_first_headers(file_name_list: list[str], dtype: numpy.dtype):
    # DOCUMENT THIS!
    header_list = []
    for file_name in file_name_list:
        sfdb_scan = dask.array.from_array(
            numpy.memmap(file_name, dtype=dtype, mode="r", shape=1)
        )
        header_list.append(sfdb_scan)

    first_header_database = dask.array.concatenate(header_list, axis=0)

    # the list of first headers should be very lightweight, 1 chunck should be enough
    return first_header_database.rechunk(-1)


def load_first_headers(file_name_list: list[str], dtype: numpy.dtype):
    # DOCUMENT THIS!
    return scan_first_headers(file_name_list, dtype).compute()


def build_header_from_arr(header_list: numpy.ndarray):
    # To build the header, i need to construct a dictionary from the arraylike obj
    header_dict = {
        attr_name: header_list[attr_name] for attr_name in header_list.dtype.names
    }
    return Header(**header_dict)


# =============================================================================


def scan_sfdb09(file_name: str | TextIO, verbose: int = 0) -> list:
    # DOCUMENT THIS
    # -------------------------------------------------------------------------
    # Checking files for consistency
    #
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
    else:
        raise ImportError(f"Given path is not a file nor a folder")

    if verbose > 0:
        print(f"{len(file_list)} file(s) found")

    if verbose > 0:
        print("Opening files...")

    # Checking if datasets of different shapes were loaded
    # In case process is aborted
    first_headers_arr = load_first_headers(file_list, HEADER_DTYPE)

    # Header construction does consistency check
    first_headers = build_header_from_arr(first_headers_arr)

    # -------------------------------------------------------------------------
    # If no problem was found on the files to load, the process starts.

    # Begin cycling over all found files
    #
    # Since we opened all header files we my preallocate memory for the upcoming
    # reading of sfdbs. We will have
    lenght_of_averaged_time_spectrum = first_headers.lenght_of_averaged_time_spectrum
    reduction_factor = first_headers.reduction_factor
    unilateral_number_of_samples = first_headers.unilateral_number_of_samples
    periodogram_shape = extract_periodogram_shape(
        lenght_of_averaged_time_spectrum, reduction_factor
    )
    ar_spectrum_shape = extract_arSpectrum_shape(
        lenght_of_averaged_time_spectrum,
        unilateral_number_of_samples,
        reduction_factor,
    )
    spectrum_shape = unilateral_number_of_samples

    # Creating a custom dtype to read sfdb files
    sfdb_dtype = numpy.dtype(
        [
            ("header", HEADER_ELEMENTS),
            ("periodogram", "float32", periodogram_shape),
            ("ar_spectrum", "float32", ar_spectrum_shape),
            ("fft_spectrum", "complex64", spectrum_shape),
        ]
    )

    _header_database = []
    _periodogram_database = []
    _ar_spectrum_database = []
    _fft_spectrum_database = []

    for sfdb_file_name in file_list:
        if verbose > 1:
            print(f"Opening {sfdb_file_name}")

        sfdb = dask.array.from_array(
            numpy.memmap(sfdb_file_name, sfdb_dtype, mode="readonly"),
            chunks=1,
        )

        _header_database.append(sfdb["header"])
        _periodogram_database.append(sfdb["periodogram"])
        _ar_spectrum_database.append(sfdb["ar_spectrum"])
        _fft_spectrum_database.append(sfdb["fft_spectrum"])

    # ============================ HEADER =====================================
    # We want the header to be immediately computed, so that the resulting dataset
    # has all the useful informations.
    header_arr_database = dask.array.concatenate(_header_database, axis=0).compute()
    header_database = build_header_from_arr(header_arr_database)

    # Other objects can be lazy
    # ======================= REGRESSIVE STUFF ================================
    periodogram_database = dask.array.concatenate(_periodogram_database, axis=0)
    ar_spectrum_database = dask.array.concatenate(_ar_spectrum_database, axis=0)

    periodogram_frequency_index = dask.array.arange(
        0, periodogram_database.shape[1], 1, dtype="int32"
    )
    periodogram_frequencies = (
        periodogram_frequency_index
        * header_database.frequency_resolution
        * header_database.reduction_factor
    )

    # ============================= SPECTRUM ==================================
    frequency_chunk_size = 64 * header_database.samples_per_hertz
    # TODO: QUESTO CONTO VA SPIEGATO
    # every chunk is 2 ** 8 * coherence_time / 2 seconds
    time_chunk_size = 64

    fft_spectrum_database = dask.array.concatenate(
        _fft_spectrum_database, axis=0
    ).rechunk(
        chunks=(time_chunk_size, frequency_chunk_size),
    )

    # Extracting frequency information from sfdb
    spectrum_frequency_index = dask.array.arange(
        0, fft_spectrum_database.shape[1], 1, dtype="int32"
    )
    spectrum_frequencies = (
        header_database.frequency_resolution * spectrum_frequency_index
    )

    # ================================ TIME ===================================
    _gps_time = header_database.gps_seconds + header_database.gps_nanoseconds * 1e-9
    gps_time = delayed(
        time.Time(
            _gps_time,
            format="gps",
            scale="utc",
        )
    )
    iso_time_values = gps_time.iso
    datetimes = pandas.to_datetime(iso_time_values.compute())

    # ========================= BUILDING XARRAY ===============================
    # Saving to Xarray and Datasets
    coordinates_names = ["frequency", "time"]

    # TODO: QUESTA COSA è LENTISSIMA
    position_vector = create_delayed_Vector3D(
        header_database.position_x,
        header_database.position_y,
        header_database.position_z,
    )
    position = xarray.DataArray(
        position_vector.compute(),
        dims=["time"],
        coords=[datetimes],
    )
    # the attributes will be shared between alla datasets, they contain the time
    # independent values of the header
    # TODO: GLI ATTRIBUTI VENGONO GENERATI ANCORA A PARTIRE DAL PRIMO HEADER, DECIDERE QUALI ATTRIBUTI TENERE COME TALI
    # TODO: E QUALI FAR DIVENTARE DELLE VARIABILI, COME PER POSIZIONE E VELOCITA'
    attributes = header_database.attributes

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
            "spectrum": spectrum.astype("complex64"),
            # "position": position,
        },
        attrs=attributes,
    )
    regressive_data = xarray.Dataset(
        data_vars={
            "periodogram": periodogram,
            "ar_spectrum": ar_spectrum,
        },
        attrs=attributes,
    )

    return (fft_data, regressive_data)


def load_sfdb09(file_name: str, verbose: int = 0):
    fft_data, regressive_data = scan_sfdb09(file_name=file_name, verbose=verbose)
    return (fft_data.compute(), regressive_data.compute())


def scan_database() -> dask.array:
    ...


def load_database():
    return scan_database.compute()


def convert_database():
    ...


def slice_database():
    ...
