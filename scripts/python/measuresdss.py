import numpy as np
import pandas as pd
from tqdm import tqdm
import corv
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'stark', 'stefan.mplstyle'))

from astroquery.sdss import SDSS
from astropy.io import fits

def get_windows(i, base_wavl):
    steps = np.linspace(0, 70, 10)
    window = dict(a = 30, b = 30, g = 30, d = 30)
    window['a'] += steps[i]
    window['b'] += steps[i]
    window['g'] += steps[i] * 0.786
    window['d'] += steps[i] * 0.643
    return window

class SDSSHandler:
    def __init__(self, table, source_key, sdss5_path):
        self.table = table
        self.source_key = source_key
        self.filenames = self.table.wd_source_id

        self.sdss5_path = sdss5_path
        self.specfinder = {'sdss4' : self.fetch_sdss4, 'sdss5' : self.fetch_sdss5}

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
        filepath = os.path.join(self.sdss5_path, row.paths)
        spec = fits.open(filepath)

        wavl = 10**spec[1].data['LOGLAM']
        flux = spec[1].data['FLUX']
        ivar = spec[1].data['IVAR']
        return wavl, flux, ivar

def measurerv(modelname, lines, win, table):
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    filepath = os.path.join(f"{basepath}", "data", "sdss5", f"{modelname}", f"{lines}_window_{win}.csv")
    figpath = os.path.join(f"{basepath}", "figures", "diagnostic", "sdss", modelname, f"{lines}_window_{win}")
    sp = SDSSHandler(table, 'wd_rv_from', f'{basepath}/data/raw/')
    
    try:
        gooddata = pd.read_csv(filepath)
    except:
        gooddata = pd.DataFrame(columns=['wd_source_id', 'wd_rv', 'wd_e_rv', 'wd_teff', 'wd_logg', 'wd_redchi'])

    remaining = list(set(table.wd_source_id) - set(gooddata.wd_source_id))
    subset = table.query(f"wd_source_id in {tuple(remaining)}")

    for i, row in tqdm(subset.iterrows(), total=subset.shape[0]):
        wavl, flux, ivar = sp.specfinder[row.wd_rv_from](row)
        window = get_windows(win, wavl)

        if modelname == '1d_da_nlte':
            model = corv.models.WarwickDAModel(model_name='1d_da_nlte', names=lines, resolution=1, windows=window, 
                                               edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0}).model
        elif modelname == 'voigt':
            model = corv.models.make_balmer_model(nvoigt=2, names=lines, windows=window, 
                                                  edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0})
            
        try:
            rv, e_rv, redchi, param_res = corv.fit.fit_corv(wavl, flux, ivar, model)
            figure = corv.utils.lineplot(wavl, flux, ivar, model, param_res.params, printparams=False)
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            figure.savefig(f"{figpath}/{row.wd_source_id}.png")

            if modelname == '1d_da_nlte':
                gooddata.loc[len(gooddata)] = {'wd_source_id' : row.wd_source_id, 'wd_rv' : rv, 'wd_e_rv' : e_rv, 'wd_teff' : param_res.params['teff'].value, 
                                'wd_logg' : param_res.params['logg'].value, 'wd_redchi' : param_res.redchi}
            elif modelname == 'voigt':
                gooddata.loc[len(gooddata)] = {'wd_source_id' : row.wd_source_id, 'wd_rv' : rv, 'wd_e_rv' : e_rv, 'wd_teff' : np.nan, 
                                'wd_logg' : np.nan, 'wd_redchi' : param_res.redchi}
                
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            gooddata.to_csv(filepath, index=False)

        except Exception as e:
            print(f"{row.wd_source_id} failed to fit: {e}")
        

if __name__ == "__main__":
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    sdssids = pd.read_csv(f'{basepath}/data/raw/sdss_commonpm.csv').query("snr > 15 & R_chance_align < 0.1")
    windows = {'abgd' : [9,8,7,6,5,4,3,2,1,0], 'ab': [9,8], 'a' : [9,8],
            'b' : [9,8], 'g' : [9,8], 'd' : [9,8]}    
    
    for line, wins in windows.items():
        for win in wins:
            print(f'LTE line : {line} || window : {win}')
            measurerv('1d_da_nlte', line, win, sdssids)
    #measurerv('voigt', 'abgd', 9, sdssids.SOURCE_ID)