# %%
from src.virgoSuite import sfdb

import importlib

importlib.reload(sfdb)


import matplotlib.pyplot as plt

location = (
    "C:/Coding/Suite/H1_DCS-CALIB_STRAIN_GATED_SUB60HZ_C01_20190401_000000.SFDB09"
)
sfdb.convert_sfdb(location, "C:/Coding/Suite", "netcdf")
