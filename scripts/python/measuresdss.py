import numpy as np
import pandas as pd
from tqdm import tqdm
import corv
import sys
import os
sys.path.append('/mnt/d/arsen/research/proj/spyquadraticstark')
sys.path.append('/usr3/graduate/arseneau/spyquadraticstark')

from stark import sdss, utils, measure

def measurerv(modelname, lines, win, table):
    sp = sdss.SDSSHandler(table, 'wd_rv_from', f'{utils.fetch_basepath()}/data/raw/sdss5')
    basepath = utils.fetch_basepath()
    filepath = os.path.join(f"{basepath}", "data", "sdss", f"{modelname}", f"{lines}", f"window_{win}.csv")
    try:
        gooddata = pd.read_csv(filepath)
    except:
        gooddata = pd.DataFrame(columns=['wd_source_id', 'wd_rv', 'wd_e_rv', 'wd_teff', 'wd_logg', 'wd_redchi'])

    remaining = list(set(table.wd_source_id) - set(gooddata.wd_source_id))
    subset = table.query(f"wd_source_id in {tuple(remaining)}")

    for i, row in tqdm(subset.iterrows(), total=subset.shape[0]):
        wavl, flux, ivar = sp.specfinder[row.wd_rv_from](row)
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
            figpath = f"{basepath}/figures/coadd_diagnostic/sdss/{modelname}/{lines}/window_{win}"
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            figure.savefig(f"{basepath}/figures/coadd_diagnostic/sdss/{modelname}/{lines}/window_{win}/{row.wd_source_id}.png")

            gooddata.loc[len(gooddata)] = {'source_id' : row.wd_source_id, 'lte_rv' : rv, 'lte_e_rv' : e_rv, 'lte_teff' : param_res.params['teff'].value, 
                             'lte_logg' : param_res.params['logg'].value, 'lte_redchi' : param_res.redchi}
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            gooddata.to_csv(filepath, index=False)
        except Exception as e:
            print(f"{row.wd_source_id} failed to fit: {e}")
        

if __name__ == "__main__":
    sdssids = pd.read_csv(f'{utils.fetch_basepath()}/data/raw/sdss_commonpm.csv').query("snr > 15 & R_chance_align < 0.1")
    windows = {'abgd' : [9,8,7,6,5,4,3,2,1,0], 'ab': [9,8], 'a' : [9,8],
            'b' : [9,8], 'g' : [9,8], 'd' : [9,8]}    
    
    for line, wins in windows.items():
        for win in wins:
            print(f'LTE line : {line} || window : {win}')
            measurerv('1d_da_nlte', line, win, sdssids)
    #measurerv('voigt', 'abgd', 9, sdssids.SOURCE_ID)
