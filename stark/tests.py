import pandas as pd
import numpy as np
import os

from .utils import fetch_basepath

def validate_lte(testlte, lines, window):
    goodpath = f"{fetch_basepath()}/data/processed/spy/mask_nlte_1d_h{lines}/spy_h{lines}_{window}.csv"
    if os.path.isfile(goodpath):
        goodlte = pd.read_csv(goodpath).query("logg > 7.1 and logg < 8.9 and e_radial_velocity < 10 and redchi < 10")
        combined = pd.merge(testlte, goodlte, on='filename')
        test = np.allclose(combined.radial_velocity, combined.lte_rv)
        if not test:
            idx = np.where(~np.isclose(combined.radial_velocity, combined.lte_rv))
            print(list(idx))
            print(combined.iloc[combined.index[idx]])
        return test, True, len(combined)
    else:
        print(f'validation: no such file {goodpath}')
        return True, False

def validate_nlte(testnlte, lines, window):
    goodpath = f"{fetch_basepath()}/data/processed/spy/reference_windows/spy_{lines}_reference_15a.csv"
    if os.path.isfile(goodpath):
        goodlte = pd.read_csv(goodpath)
        combined = pd.merge(testnlte, goodlte, on='filename')
        return np.allclose(combined.radial_velocity, combined.lte_rv), True, len(combined)
    else:
        print(f'validation: no such file {goodpath}')
        return True, False
