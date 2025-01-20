import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append('/mnt/d/arsen/research/proj/spyquadraticstark')
sys.path.append('/usr3/graduate/arseneau/spyquadraticstark')

from stark import utils
from stark import measure 
import corv

def read_spectrum(source_id):
    basepath = utils.fetch_basepath()
    return pd.read_csv(os.path.join(basepath, 'data', 'raw', 'coadd_smooth', f'{source_id}.csv'))

def measure_lte(modelname, lines, win, source_ids):
    basepath = utils.fetch_basepath()
    ltepath = os.path.join(f"{basepath}", "data", "coadd", "smooth", modelname, lines, f"window_{win}.csv")
    try:
        gooddata = pd.read_csv(ltepath)
    except:
        gooddata = pd.DataFrame(columns=['source_id', 'lte_rv', 'lte_e_rv', 'lte_teff', 'lte_logg', 'lte_redchi'])

    remaining = list(set(source_ids) - set(gooddata.source_id))
    for source in tqdm(remaining):
        spec = read_spectrum(source)
        wavl, flux, ivar = spec.wavl.values, spec.flux.values, spec.ivar.values
        window = measure.get_windows(win, wavl)

        if modelname == '1d_da_nlte':
            model = corv.models.WarwickDAModel(model_name='1d_da_nlte', names=lines, resolution=1, windows=window, 
                                               edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0}).model
        elif modelname == 'voigt':
            model = corv.models.make_balmer_model(nvoigt=1, names=lines, windows=window, 
                                                  edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0})
        
        try:
            rv, e_rv, redchi, param_res = corv.fit.fit_corv(wavl, flux, ivar, model)
            figure = corv.utils.lineplot(wavl, flux, ivar, model, param_res.params)
            figpath = f"{basepath}/figures/coadd_diagnostic/smooth/{modelname}/{lines}/window_{win}"
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            figure.savefig(f"{basepath}/figures/coadd_diagnostic/smooth/{modelname}/{lines}/window_{win}/{source}.png")

            gooddata.loc[len(gooddata)] = {'source_id' : source, 'lte_rv' : rv, 'lte_e_rv' : e_rv, 'lte_teff' : param_res.params['teff'].value, 
                             'lte_logg' : param_res.params['logg'].value, 'lte_redchi' : param_res.redchi}
            if not os.path.exists(os.path.dirname(ltepath)):
                os.makedirs(os.path.dirname(ltepath))
            gooddata.to_csv(ltepath, index=False)
        except Exception as e:
            print(f"{source} failed to fit: {e}")

if __name__ == "__main__":
    goodcoadds = pd.read_csv(os.path.join(f"{utils.fetch_basepath()}", "data", "goodcoadds.csv"))
    windows = {'abgd' : [9,8,7,6,5,4,3,2,1,0], 'ab': [9,8], 'a' : [9,8],
            'b' : [9,8], 'g' : [9,8], 'd' : [9,8]}    
    
    for line, wins in windows.items():
        for win in wins:
            print(f'LTE line : {line} || window : {win}')
            measure_lte('1d_da_nlte', line, win, goodcoadds.SOURCE_ID)
    measure_lte('voigt', 'abgd', 9, goodcoadds.SOURCE_ID)

