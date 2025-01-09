import pandas as pd
import numpy as np
import os

def validate_lte(testlte, lines, window):
    goodpath = f"/mnt/d/arsen/research/proj/spyquadraticstark/data/processed/spy/mask_nlte_1d_h{lines}/spy_h{lines}_{window}.csv"
    if os.path.isfile(goodpath):
        goodlte = pd.read_csv(goodpath)
        combined = pd.merge(testlte, goodlte, on='filename')
        return np.allclose(combined.radial_velocity, combined.lte_rv), True
    else:
        print(f'validation: no such file {goodpath}')
        return True, False

def validate_nlte(testnlte, lines, window):
    goodpath = f"/mnt/d/arsen/research/proj/spyquadraticstark/data/processed/spy/reference_windows/spy_{lines}_reference_15a.csv"
    if os.path.isfile(goodpath):
        goodlte = pd.read_csv(goodpath)
        combined = pd.merge(testnlte, goodlte, on='filename')
        return np.allclose(combined.radial_velocity, combined.lte_rv), True
    else:
        print(f'validation: no such file {goodpath}')
        return True, False