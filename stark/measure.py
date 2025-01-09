import corv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.use('Agg')

from astropy.io import ascii
from contextlib import contextmanager
import sys, os

from .utils import air2vac
from .tests import validate_lte, validate_nlte

def read_spectrum(file, specpath = '/mnt/d/arsen/research/proj/spyquadraticstark/data/raw/sp'):
    # find, download, or skip the file
    path = os.path.join(specpath, file)
    table = ascii.read(path)
    wl = air2vac(table['Table'].data)
    fl = table[':'].data
    mask = (5260 < wl) * (wl < 5280) # continuum region
    snr = np.nanmean(fl[mask]) / np.nanstd(fl[mask])
    snr = snr if ~np.isnan(snr) else 1
    ivar =  snr**2 / (table[':'].data + 1e-6)**2
    return wl, fl, ivar

def get_windows(i, base_wavl):
    steps = np.array([ 0.        ,  7.77777778, 15.55555556, 23.33333333, 31.11111111,
                        38.88888889, 46.66666667, 54.44444444, 62.22222222, 70.        ])
    window = dict(a = 30, b = 30, g = 30, d = 30)
    window['a'] += steps[i]
    window['b'] += steps[i]
    window['g'] += steps[i] * 0.786
    window['d'] += steps[i] * 0.643
    return window

def get_mask(wl, mask_size):
    centres =  dict(a = 6564.61, b = 4862.68, g = 4341.68, d = 4102.89)
    mask_ha = ((centres['a'] - mask_size) < wl) * (wl < (centres['a'] + mask_size))
    mask_hb = ((centres['b'] - mask_size) < wl) * (wl < (centres['b'] + mask_size))
    mask_hg = ((centres['g'] - mask_size) < wl) * (wl < (centres['g'] + mask_size))
    mask_hd = ((centres['d'] - mask_size) < wl) * (wl < (centres['d'] + mask_size))
    mask = np.logical_or(mask_ha, mask_hb)
    mask = np.logical_or(mask, mask_hg)
    mask = np.logical_or(mask, mask_hd)
    return ~mask

def measure_spectrum(wavl, flux, ivar, i, lines = ['a','b','g','d'], lte_mask_size = 8, nlte_core_size = 15, modeltype='1d_da_nlte', mode = 'lte'):
    assert modeltype in ['1d_da_nlte', 'voigt'], 'Model type not recognized'
    assert mode in ['lte', 'nlte'], 'Mode note recognized'
    edges = {'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0}

    if mode == 'lte':
        mask = get_mask(wavl, lte_mask_size)
        window = get_windows(i, wavl)
    elif mode == 'nlte':
        mask = np.ones(wavl.shape)
        window = {'a' : nlte_core_size, 'b' : nlte_core_size, 'g' : nlte_core_size, 'd' : nlte_core_size}

    if modeltype == '1d_da_nlte':
        model = corv.models.WarwickDAModel(model_name='1d_da_nlte', names=lines, resolution=0.0637, windows=window, edges=edges).model
    elif modeltype == 'voigt':
        model = corv.models.make_balmer_model(nvoigt=1, names=lines, windows=window, edges=edges)

    rv, e_rv, redchi, param_res = corv.fit.fit_corv(wavl[mask], flux[mask], ivar[mask], model)
    figure = corv.utils.lineplot(wavl[mask], flux[mask], ivar[mask], model, param_res.params)
    plt.close()

    results = {f'{mode}_rv' : [rv], f'{mode}_e_rv' : [e_rv], 
               f'{mode}_teff' : [param_res.params['teff'].value], f'{mode}_logg' : [param_res.params['logg'].value], f'{mode}_redchi' : [param_res.redchi],}
    return results, figure

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def ltemain(lines_to_test = ['abgd', 'ab', 'a', 'b', 'g', 'd'], lte_mask_sizes = [8], modeltypes = ['1d_da_nlte', 'voigt'], windows = [9,8,7,6,5,4,3,2,1,0], verbose=True):
    names = pd.read_csv('/mnt/d/arsen/research/proj/spyquadraticstark/data/processed/good_spy.csv').FileName.values
    for a, model in enumerate(modeltypes):
        for b, lines in enumerate(lines_to_test):
            for c, mask_size in enumerate(lte_mask_sizes):
                for d, window in enumerate(windows):
                    print(f'model: {model} ({a}/{len(modeltypes)}) || lines: {lines} ({b}/{len(lines)}) || lte_mask_size: {mask_size} ({c}/{len(lte_mask_sizes)}) || window: {window} ({d}/{len(windows)})')
                    prefix = f"lte/{model}/{mask_size}angstrom/{lines}/window_{window}"
                    # if available, read in existing data; otherwise create a new file
                    try:
                        existingdata = pd.read_csv(f'/mnt/d/arsen/research/proj/spyquadraticstark/data/processed/{prefix}.csv')
                    except:
                        existingdata = pd.DataFrame({'filename' : [], 'lte_rv' : [], 'lte_e_rv' : [], 'lte_teff' : [], 'lte_logg' : [], 'lte_redchi' : [],})
                    # find files that have not already been calculated
                    unique_names = list(set(names) - set(existingdata.filename))
                    for e, name in enumerate(tqdm(unique_names)):
                        #try:
                        # read the spectrum
                        wavl, flux, ivar = read_spectrum(name)
                        with suppress_stdout():
                            # perform the measurement in lte mode
                            dat, ltefig = measure_spectrum(wavl, flux, ivar, window, lines = lines, lte_mask_size = 8, 
                                                           nlte_core_size = 15, modeltype='1d_da_nlte', mode = 'lte')
                            dat = pd.DataFrame(dat)
                        dat['filename'] = name
                        # combine with existing data and save everything
                        existingdata = pd.concat([existingdata, dat])
                        ltefig.savefig(f"/mnt/d/arsen/research/proj/spyquadraticstark/figures/diagnostic/{prefix}/{name}.png")
                        plt.close()
                        if e % 10 == 0:
                            existingdata.to_csv(f'/mnt/d/arsen/research/proj/spyquadraticstark/data/processed/{prefix}.csv', index=False)
                            validation_state, validation_succeeded = validate_lte(existingdata, lines, window)
                            assert validation_state, "Validation failed!"
                            if validation_succeeded and verbose:
                                print(f"Validation succeeded on {len(existingdata)} datapoints. Continuing!")
                        
                    if (lines != 'abgd') and (window <= 8):
                        break

if __name__ == "__main__":
    ltemain()