# %%
from src.virgoSuite import sfdb

import importlib

importlib.reload(sfdb)

location = (
    "C:/Coding/Suite/SFDBS/L1_DCS-CALIB_STRAIN_GATED_SUB60HZ_C01_20200324_181158.SFDB09"
)
sfdb.convert_sfdb(
    location,
    "C:/Coding/Suite/",
    "netcdf",
    telegram_notifications=True,
    token="5944315082:AAEJWn__lrrwUi7inaI0-y5JrkhB4Ovb_gg",
    chat_id="652707762",
)
