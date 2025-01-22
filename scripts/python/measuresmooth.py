import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
from Payne.utils import smoothing
import corv

import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'stark', 'stefan.mplstyle'))

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

def get_windows(i, base_wavl):
    steps = np.linspace(0, 70, 10)
    window = dict(a = 30, b = 30, g = 30, d = 30)
    window['a'] += steps[i]
    window['b'] += steps[i]
    window['g'] += steps[i] * 0.786
    window['d'] += steps[i] * 0.643
    return window

def read_lte_spectrum(source_id):
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    return pd.read_csv(os.path.join(basepath, 'data', 'raw', 'spy_epoch', 'sp_csv_smooth', f'{source_id}.csv'))

def read_nlte_spectrum(source_id):
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    return pd.read_csv(os.path.join(basepath, 'data', 'raw', 'spy_epoch', 'sp_csv', f'{source_id}.csv'))

def measure_lte(modelname, lines, win, source_ids, smooth=True):
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    if smooth:
        ltepath = os.path.join(f"{basepath}", "data", "processed", "lte_smooth", modelname, f"{lines}_window_{win}.csv")
        figpath = os.path.join(f"{basepath}", "figures", "diagnostic", "lte_smooth", modelname, f"{lines}_window_{win}")
    else:
        ltepath = os.path.join(f"{basepath}", "data", "processed", "lte", modelname, f"{lines}_window_{win}.csv")
        figpath = os.path.join(f"{basepath}", "figures", "diagnostic", "lte", modelname, f"{lines}_window_{win}")

    try:
        gooddata = pd.read_csv(ltepath)
    except:
        gooddata = pd.DataFrame(columns=['obsname', 'lte_rv', 'lte_e_rv', 'lte_teff', 'lte_logg', 'lte_redchi'])

    remaining = list(set(source_ids) - set(gooddata.obsname))
    for source in tqdm(remaining):
        """read in the correct spectrum and apply the correct mask"""
        if smooth:
            spec = read_lte_spectrum(source)
            wavl, flux, ivar = spec.wavl, spec.flux, spec.ivar
            resolution = 1
        else:
            spec, mask = read_nlte_spectrum(source), get_mask(wavl, mask_size=4)
            wavl, flux, ivar = spec.wavl, spec.flux, spec.ivar
            wavl, flux, ivar = wavl[mask], flux[mask], ivar[mask]
            resolution = 0.0637
        window = get_windows(win, wavl)

        """create the correct model (either voigt or 1d_da_nlte)"""
        if modelname == '1d_da_nlte':
            model = corv.models.WarwickDAModel(model_name='1d_da_nlte', names=lines, resolution=resolution, windows=window, 
                                               edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0}).model
        elif modelname == 'voigt':
            model = corv.models.make_balmer_model(nvoigt=2, names=lines, windows=window, 
                                                  edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0})
            
        """try to fit the model to the spectrum and save the results"""
        try:
            rv, e_rv, redchi, param_res = corv.fit.fit_corv(wavl, flux, ivar, model)
            figure = corv.utils.lineplot(wavl, flux, ivar, model, param_res.params, printparams=False)
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            figure.savefig(f"{figpath}/{source}.png")
            plt.close()

            if modelname == '1d_da_nlte': 
                gooddata.loc[len(gooddata)] = {'obsname' : source, 'lte_rv' : rv, 'lte_e_rv' : e_rv, 'lte_teff' : param_res.params['teff'].value, 
                            'lte_logg' : param_res.params['logg'].value, 'lte_redchi' : param_res.redchi}
            elif modelname == 'voigt': 
                gooddata.loc[len(gooddata)] = {'obsname' : source, 'lte_rv' : rv, 'lte_e_rv' : e_rv, 'lte_teff' : np.nan, 
                            'lte_logg' : np.nan, 'lte_redchi' : param_res.redchi}
                
            if not os.path.exists(os.path.dirname(ltepath)):
                os.makedirs(os.path.dirname(ltepath))
            gooddata.to_csv(ltepath, index=False)
        except Exception as e:
            print(f"{source} failed to fit: {e}")

def measure_nlte(modelname, lines, size, source_ids):
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    nltepath = os.path.join(f"{basepath}", "data", "processed", "nlte", modelname, f"{lines}_{size}aa.csv")
    figpath = os.path.join(f"{basepath}", "figures", "diagnostic", "nlte", modelname, f"{lines}_{size}aa")
    try:
        gooddata = pd.read_csv(nltepath)
    except:
        gooddata = pd.DataFrame(columns=['obsname', 'nlte_rv', 'nlte_e_rv', 'nlte_teff', 'nlte_logg', 'nlte_redchi'])

    remaining = list(set(source_ids) - set(gooddata.obsname))
    for source in tqdm(remaining):
        print(source)
        spec = read_nlte_spectrum(source)
        wavl, flux, ivar = spec.wavl, spec.flux, spec.ivar
        window = {'a' : size, 'b' : size, 'g' : size, 'd' : size}

        if modelname == '1d_da_nlte':
            model = corv.models.WarwickDAModel(model_name='1d_da_nlte', names=lines, resolution=0.0637, windows=window, 
                                               edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0}).model
        elif modelname == 'voigt':
            model = corv.models.make_balmer_model(nvoigt=2, names=lines, windows=window, 
                                                  edges={'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0})
            
        try:
            rv, e_rv, redchi, param_res = corv.fit.fit_corv(wavl, flux, ivar, model)
            figure = corv.utils.lineplot(wavl, flux, ivar, model, param_res.params, printparams=False)
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            figure.savefig(f"{figpath}/{source}.png")
            plt.close()

            if modelname == '1d_da_nlte': 
                gooddata.loc[len(gooddata)] = {'obsname' : source, 'nlte_rv' : rv, 'nlte_e_rv' : e_rv, 'nlte_teff' : param_res.params['teff'].value, 
                            'nlte_logg' : param_res.params['logg'].value, 'nlte_redchi' : param_res.redchi}
            if modelname == 'voigt': 
                gooddata.loc[len(gooddata)] = {'obsname' : source, 'nlte_rv' : rv, 'nlte_e_rv' : e_rv, 'nlte_teff' : np.nan, 
                            'nlte_logg' : np.nan, 'nlte_redchi' : param_res.redchi}
                
            if not os.path.exists(os.path.dirname(nltepath)):
                os.makedirs(os.path.dirname(nltepath))
            gooddata.to_csv(nltepath, index=False)
        except Exception as e:
            print(f"{source} failed to fit: {e}")


if __name__ == "__main__":
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    goodobjs = pd.read_csv(os.path.join(basepath, "data", "raw", "spy_epoch", "reference_objs.csv"))
    windows = {'abgd' : [9,8,7,6,5,4,3,2,1,0], 'ab': [9,8], 'a' : [9,8],
            'b' : [9,8], 'g' : [9,8], 'd' : [9,8]}    
    
    """fit smoothed lte rvs
    """
    for line, wins in windows.items():
        for win in wins:
            print(f'LTE line : {line} || window : {win}')
            measure_lte('1d_da_nlte', line, win, goodobjs.obsname)
    measure_lte('voigt', 'abgd', 9, goodobjs.obsname)

    """fit nlte rvs
    """
    sizes = {'ab': [15, 8], 'abgd' : [15, 8], 'a' : [15, 8],
            'b' : [15, 8], 'g' : [15, 8], 'd' : [15, 8]}
    for line, wins in sizes.items():
        for win in wins:
            print(f'LTE line : {line} || window : {win}')
            measure_nlte('1d_da_nlte', line, win, goodobjs.obsname)
    measure_nlte('voigt', 'abgd', 9, goodobjs.obsname)

    """fit normal lte rvs
    """
    for line, wins in windows.items():
        for win in wins:
            print(f'LTE line : {line} || window : {win}')
            measure_lte('1d_da_nlte', line, win, goodobjs.obsname, smooth=False)
    measure_lte('voigt', 'abgd', 9, goodobjs.obsname, smooth=False)

