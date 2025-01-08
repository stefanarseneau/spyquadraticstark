import pandas as pd
import numpy as np

def read_clean_nlte(lines, size):
    goodsystems = pd.read_csv('../data/processed/good_spy.csv').FileName.values
    nlte = pd.read_csv(f'../data/processed/nlte/{size}angstrom/{lines}.csv')
    nlte = nlte.query("nlte_logg > 7.1 and nlte_logg < 8.9 and nlte_e_rv < 10 and nlte_redchi < 10")
    return  nlte.loc[nlte['filename'].isin(goodsystems)]

def read_clean_lte(lines, size, window):
    goodsystems = pd.read_csv('../data/processed/good_spy.csv').FileName.values
    lte = pd.read_csv(f'../data/processed/lte/{size}angstrom/{lines}/window_{window}.csv')
    lte = lte.query("nlte_logg > 7.1 and nlte_logg < 8.9 and nlte_e_rv < 10 and nlte_redchi < 10")
    return  lte.loc[lte['filename'].isin(goodsystems)]

def read_stark_effect(lteargs, nlteargs = None, sigmaclip=True):
    nlteargs = {'lines' : 'ab', 'size' : 15} if nlteargs is None else nlteargs
    nlte = read_clean_nlte(**nlteargs)
    lte = read_clean_lte(**lteargs)
    df = pd.merge(nlte, lte, on='filename')
    # compute stark velocities
    df['vstark'] = df['lte_rv'] - df['nlte_rv']
    df['e_vstark'] = np.sqrt(df['lte_rv']**2 + df['nlte_rv']**2)
    # return
    if sigmaclip:
        sigma_to_clip = 3
        mean = np.mean(df.vstark)
        stddev = np.std(df.vstark)
        return df.query(f"abs(vstark - {mean}) < {sigma_to_clip * stddev}")
    else:
        return df
        
