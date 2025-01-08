import corv
import numpy as np
import matplotlib

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

def measure_spectrum(wavl, flux, ivar, i, lines = ['a','b','g','d'], lte_mask_size = 8, nlte_core_size = 15, modeltype='1d_da_nlte'):
    assert modeltype in ['1d_da_nlte', 'voigt'], 'Model type not recognized'
    edges = {'a' : 0, 'b' : 0, 'g' : 0, 'd' : 0}

    ltemask = get_mask(wavl, lte_mask_size)
    ltewindow = get_windows(i, wavl)
    nltewindow = {'a' : nlte_core_size, 'b' : nlte_core_size, 'g' : nlte_core_size, 'd' : nlte_core_size}

    if modeltype == '1d_da_nlte':
        ltemodel = corv.models.WarwickDAModel(model_name='1d_da_nlte', names = lines, resolution = 0.0637, windows=ltewindow, edges=edges).model
        nltemodel = corv.models.WarwickDAModel(model_name='1d_da_nlte', names = lines, resolution = 0.0637, windows=nltewindow, edges=edges).model
    elif modeltype == 'voigt':
        ltemodel = corv.models.make_balmer_model(nvoigt=1, names = lines, windows=ltewindow, edges=edges)
        nltemodel = corv.models.make_balmer_model(nvoigt=1, names = lines, windows=nltewindow, edges=edges)

    lte_rv, lte_e_rv, lte_redchi, lte_param_res = corv.fit.fit_corv(wavl[ltemask], flux[ltemask], ivar[ltemask], ltemodel)
    nlte_rv, nlte_e_rv, nlte_redchi, nlte_param_res = corv.fit.fit_corv(wavl, flux, ivar, nltemodel)

    #lte_fig = corv.utils.lineplot(wavl[ltemask], flux[ltemask], ivar[ltemask], ltemodel, lte_param_res.params)
    #nlte_fig = corv.utils.lineplot(wavl, flux, ivar, nltemodel, nlte_param_res.params)

    results = {'lte_rv' : [lte_rv], 'lte_e_rv' : [lte_e_rv], 
               'lte_teff' : [lte_param_res.params['teff'].value], 'lte_logg' : [lte_param_res.params['logg'].value], 'lte_redchi' : [lte_param_res.redchi],
               'nlte_rv' : [nlte_rv], 'nlte_e_rv' : [nlte_e_rv], 
               'nlte_teff' : [nlte_param_res.params['teff'].value], 'nlte_logg' : [nlte_param_res.params['logg'].value], 'nlte_redchi' : [nlte_param_res.redchi],
               'vstark' : [lte_rv - nlte_rv], 'e_vstark' : [np.sqrt(lte_e_rv**2 + nlte_e_rv**2)]}
    return results