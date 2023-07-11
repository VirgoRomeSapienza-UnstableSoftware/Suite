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
class HeaderAttributes:
    # count: numpy.float64
    detector: numpy.int32
    fft_lenght: numpy.float64
    # starting_fft_sample_index: numpy.int32
    unilateral_number_of_samples: numpy.int32
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
    number_of_zeros: numpy.float64
    sat_howmany: numpy.float64
    percentage_of_zeros: numpy.float32
    lenght_of_averaged_time_spectrum: numpy.int32
    scientific_segment: numpy.int32

    detector_name: str = field(init=False)
    window_normalization_name: str = field(init=False)
    fft_interlaced_name: str = field(init=False)

    def __post_init__(self):
        self.detector_name = [extract_detector(detector) for detector in self.detector]
        self.window_normalization_name = [
            extract_window_type(wink) for wink in self.window_type
        ]
        self.fft_interlaced_name = [
            extract_interlace_method(interlaced) for interlaced in self.fft_interlaced
        ]

    # TODO: AGGIUNGERE LE PROPERTIES
    # sampling rates, nyquist, coherence time, half coherence time (time delta between interlaced coherence time)
    # TODO: ASSERT, CONTROLLARE CHE LE PROPERTIES COINCIDANO CON QUELLO CHE SI INSERISCE
    # TODO: COMPUTE CHUNKSIZE


class PiaHeader(NamedTuple):
    # This gives read-only privileges
    # DOCUMENT THIS
    count: list[numpy.float64]
    detector: list[numpy.int32]
    gps_seconds: list[numpy.int32]
    gps_nanoseconds: list[numpy.int32]
    fft_lenght: list[numpy.float64]
    starting_fft_sample_index: list[numpy.int32]
    unilateral_number_of_samples: list[numpy.int32]
    reduction_factor: list[numpy.int32]
    fft_interlaced: list[numpy.int32]
    number_of_flags: list[numpy.float32]
    scaling_factor: list[numpy.float32]
    mjd_time: list[numpy.float64]
    fft_index: list[numpy.int32]
    window_type: list[numpy.int32]
    normalization_factor: list[numpy.float32]
    window_normalization: list[numpy.float32]
    starting_fft_frequency: list[numpy.float64]
    subsampling_time: list[numpy.float64]
    frequency_resolution: list[numpy.float64]
    position_x: list[numpy.float64]
    position_y: list[numpy.float64]
    position_z: list[numpy.float64]
    velocity_x: list[numpy.float64]
    velocity_y: list[numpy.float64]
    velocity_z: list[numpy.float64]
    number_of_zeros: list[numpy.int32]
    sat_howmany: list[numpy.float64]
    spare_1: list[numpy.float64]
    spare_2: list[numpy.float64]
    spare_3: list[numpy.float64]
    percentage_of_zeros: list[numpy.float32]
    spare_5: list[numpy.float64]
    spare_6: list[numpy.float64]
    lenght_of_averaged_time_spectrum: list[numpy.int32]
    scientific_segment: list[numpy.int32]
    spare_9: list[numpy.float64]


# =============================================================================
# *****************************************************************************
# =============================================================================


def list_files_in_directory(path: str, file_type: str):
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
    if lenght_of_averaged_time_spectrum > 0:
        return lenght_of_averaged_time_spectrum
    else:
        return reduction_factor


def extract_arSpectrum_shape(
    lenght_of_averaged_time_spectrum: numpy.int32,
    unilateral_number_of_samples: numpy.int32,
    reduction_factor: numpy.int32,
):
    if lenght_of_averaged_time_spectrum > 0:
        return lenght_of_averaged_time_spectrum
    else:
        return int(unilateral_number_of_samples / reduction_factor)


# =============================================================================


def flag_check(flags: list[str], header: PiaHeader):
    for flag in flags:
        assert min(getattr(header, flag)) == max(getattr(header, flag))


# =============================================================================


def load_first_headers(file_list: list[str], HEADER_DTYPE: numpy.dtype):
    list_of_first_headers = pandas.DataFrame(
        numpy.zeros((len(file_list), len(HEADER_DTYPE))),
        columns=HEADER_DTYPE.names,
    )
    # opening files files to read the first header
    for i, sfdb_file_name in enumerate(file_list):
        first_header_of_file = dask.array.from_array(
            numpy.fromfile(sfdb_file_name, dtype=HEADER_DTYPE, count=1)
        )

        # Unraveling header
        list_of_first_headers.iloc[i] = np_header_to_series(first_header_of_file)

    # Converting the pandas dataframe into a dictionary of arrays
    dictionary_of_headers = list_of_first_headers.astype(
        {dtype[0]: dtype[1] for dtype in HEADER_ELEMENTS}
    ).to_dict(orient="list")
    # Pia header has the right custom python type
    return PiaHeader(**dictionary_of_headers)


def np_header_to_series(header: HEADER_DTYPE):
    return pandas.Series({key: header[key] for key in header.dtype.names})


def header_extract_attributes(header: PiaHeader):
    attribute_list = [
        "detector",
        "fft_lenght",
        "unilateral_number_of_samples",
        "reduction_factor",
        "fft_interlaced",
        "number_of_flags",
        "scaling_factor",
        "fft_index",
        "window_type",
        "normalization_factor",
        "window_normalization",
        "starting_fft_frequency",
        "subsampling_time",
        "frequency_resolution",
        "number_of_zeros",
        "sat_howmany",
        "percentage_of_zeros",
        "lenght_of_averaged_time_spectrum",
        "scientific_segment",
    ]
    return HeaderAttributes(**{key: getattr(header, key) for key in attribute_list})


# =============================================================================


def load_sfdb09(file_name: str | TextIO, verbose: int = 0) -> list:
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

    # List of flags that have to be the same accross all opened files
    database_flags = numpy.array(
        [
            "detector",
            "window_type",
            "lenght_of_averaged_time_spectrum",
            "fft_interlaced",
            "unilateral_number_of_samples",
            "reduction_factor",
        ],
        dtype="str",
    )

    # Checking if datasets of different shapes were loaded
    # In case process is aborted
    first_header_per_file = load_first_headers(file_list, HEADER_DTYPE)
    try:
        flag_check(database_flags, first_header_per_file)
    except:
        raise ValueError(
            f"\
                \nGiven path contains multiple databases with different T_fft.\
                \nPlease select a path with SFDB of fixed lenght."
        )

    # -------------------------------------------------------------------------
    # If no problem was found on the files to load, the process starts.

    # Begin cycling over all found files
    #
    # Since we opened all header files we my preallocate memory for the upcoming
    # reading of sfdbs. We will have
    lenght_of_averaged_time_spectrum = (
        first_header_per_file.lenght_of_averaged_time_spectrum[0]
    )
    reduction_factor = first_header_per_file.reduction_factor[0]
    unilateral_number_of_samples = first_header_per_file.unilateral_number_of_samples[0]
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

    # TODO: RIEMPIRE è PIù LENTO CHE FARE L'APPEND
    """
    # Allocating a list to store timestamp indexes

    client = Client(n_workers=2, threads_per_worker=4)

    size_of_one_fft = sfdb_dtype.itemsize
    n_ffts_per_file = [0] * (len(file_list) + 1)

    for i, file_name in enumerate(file_list):
        file_size = getsize(file_name)
        n_ffts_per_file[i + 1] = n_ffts_per_file[i] + int(file_size / size_of_one_fft)
    n_timestamps = numpy.sum(n_ffts_per_file, dtype=numpy.int32)

    _periodogram_database = dask.array.zeros(
        (n_timestamps, periodogram_shape), dtype=numpy.float64
    )
    _ar_spectrum_database = dask.array.zeros(
        (n_timestamps, ar_spectrum_shape), dtype=numpy.float64
    )
    _fft_spectrum_database = dask.array.zeros(
        (n_timestamps, spectrum_shape), dtype=numpy.complex64
    )
    _header_database = dask.array.zeros(n_timestamps, dtype=HEADER_DTYPE)
    """
    _header_database = []
    _periodogram_database = []
    _ar_spectrum_database = []
    _fft_spectrum_database = []

    for i, sfdb_file_name in enumerate(file_list):
        if verbose > 1:
            print(f"Opening {sfdb_file_name}")

        sfdb = dask.array.from_array(
            numpy.memmap(sfdb_file_name, sfdb_dtype, mode="readonly"),
            chunks=1,
        )

        """
        for j in range(n_ffts_per_file[i + 1]):
            # _header_database[n_ffts_per_file[i] + j] = headers[j]

            _periodogram_database[n_ffts_per_file[i] + j] = sfdb["periodogram"][j, :]
            _ar_spectrum_database[n_ffts_per_file[i] + j] = sfdb["ar_spectrum"][j, :]
            _fft_spectrum_database[n_ffts_per_file[i] + j] = sfdb["fft_spectrum"][j, :]

        """
        _header_database.append(sfdb["header"])
        _periodogram_database.append(sfdb["periodogram"])
        _ar_spectrum_database.append(sfdb["ar_spectrum"])
        _fft_spectrum_database.append(sfdb["fft_spectrum"])

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
    # TODO: QUESTO CONTO VA SPIEGATO
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
    coordinates_names = ["frequency", "time"]

    # TODO: QUESTA COSA è LENTISSIMA
    position_vector = create_delayed_Vector3D(
        header_database["position_x"],
        header_database["position_y"],
        header_database["position_z"],
    )
    position = xarray.DataArray(
        position_vector.compute(),
        dims=["time"],
        coords=[datetimes],
    )
    coordinate_values = dict(
        time=datetimes,
        v_x=(("time"), header_database["velocity_x"]),
        v_y=(("time"), header_database["velocity_y"]),
        v_z=(("time"), header_database["velocity_z"]),
    )

    # the attributes will be shared between alla datasets, they contain the time
    # independent values of the header
    # TODO: GLI ATTRIBUTI VENGONO GENERATI ANCORA A PARTIRE DAL PRIMO HEADER, DECIDERE QUALI ATTRIBUTI TENERE COME TALI
    # TODO: E QUALI FAR DIVENTARE DELLE VARIABILI, COME PER POSIZIONE E VELOCITA'
    attributes = asdict(header_extract_attributes(first_header_per_file))

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


def scan_database() -> dask.array:
    ...


def load_database():
    return scan_database.compute()


def convert_database():
    ...


def slice_database():
    ...
