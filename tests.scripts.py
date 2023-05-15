# %%
from src.virgoSuite import sfdb

import importlib

importlib.reload(sfdb)


import matplotlib.pyplot as plt

location = (
    "C:/Coding/Suite/H1_DCS-CALIB_STRAIN_GATED_SUB60HZ_C01_20190401_000000.SFDB09"
)
data = sfdb.load_file_sfdb(location)[0]

# %%

from pathlib import Path

save_path = "." + f"/DATABASE/netcdf/{data.detector.values[0]}/O3/C01/"
Path(save_path).mkdir(parents=True, exist_ok=True)
data.to_netcdf(
    save_path + "power_spectrum.netCDF4",
    mode="w",
    engine="NETCDF4",
    invalid_netcdf=True,
)
