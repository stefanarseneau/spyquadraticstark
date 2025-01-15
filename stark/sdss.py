import os
import json

from astropy.io import fits
from astroquery.sdss import SDSS
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

#import interpolator as interp
import corv
from . import measure
from . import utils

class SDSSHandler:
    def __init__(self, table, source_key, sdss5_path):
        self.table = table
        self.source_key = source_key
        self.filenames = self.table.wd_source_id

        self.sdss5_path = sdss5_path
        self.specfinder = {'sdss4' : self.fetch_sdss4, 'sdss5' : self.fetch_sdss5}

    def analyze_table(self, outfile, lines = ['a', 'b'], resolution = 1, from_cache=False, window_indx = -1):
        if from_cache:
            try:
                results = pd.read_csv(outfile)
            except:
                results = pd.DataFrame({'wd_source_id' : [], 'wd_rv' : [], 'wd_e_rv' : [],
                        'redchi': [], 'teff' : [], 'logg' : []})
        else:
            results = pd.DataFrame({'wd_source_id' : [], 'wd_rv' : [], 'wd_e_rv' : [],
                        'redchi': [], 'teff' : [], 'logg' : []})
        processed_files = results.wd_source_id.tolist()
        to_analyze = list(set(self.filenames) - set(processed_files))
        subset = self.table.query(f"wd_source_id in {tuple(to_analyze)}")

        for i, row in tqdm(subset.iterrows(), total=subset.shape[0]):
            if str(row.wd_source_id) not in results.keys():
                wavl, flux, ivar = self.specfinder[row.wd_rv_from](row)

                row_vals = fit_lines(str(row.wd_source_id), wavl, flux, ivar, lines, resolution, window_indx = window_indx)

                results = pd.concat([results, row_vals])
                results.to_csv(outfile, index=False)
        return results

    def fetch_sdss4(self, row):
        plate = row.wd_plate
        mjd = row.wd_mjd
        fiberid = row.wd_fiberid
        spec = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberid)[0]

        wavl = 10**spec[1].data['LOGLAM']
        flux = spec[1].data['FLUX']
        ivar = spec[1].data['IVAR']
        return wavl, flux, ivar
    
    def fetch_sdss5(self, row):
        filepath = os.path.join(self.sdss5_path, row.wd_filepath)
        spec = fits.open(filepath)

        wavl = 10**spec[1].data['LOGLAM']
        flux = spec[1].data['FLUX']
        ivar = spec[1].data['IVAR']
        return wavl, flux, ivar

def fit_lines(name, wavl, flux, ivar, lines = ['a', 'b'], resolution = 1, window_indx = -1):
    try:
        window = utils.get_windows(window_indx)
        edges = {'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0}
        corvmodel = corv.models.WarwickDAModel(model_name='1d_da_nlte', names = lines, resolution = resolution, windows=window, edges=edges)
        ref_rv, ref_e_rv, ref_redchi, ref_param_res = corv.fit.fit_corv(wavl, flux, ivar, corvmodel.model)

        row_vals = pd.DataFrame()                    
        row_vals['wd_source_id'] = [name]
        row_vals['radial_velocity'] = [ref_rv]
        row_vals['e_radial_velocity'] = [ref_e_rv]
        row_vals['redchi'] = [ref_redchi]
        row_vals['teff'] = [ref_param_res.params['teff'].value]
        row_vals['logg'] = [ref_param_res.params['logg'].value]
    except:
        print(f"Fit failed: {name}")
    return row_vals
    