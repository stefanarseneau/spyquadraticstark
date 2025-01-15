import pandas as pd
import numpy as np
import pyvo
import sys
sys.path.append('/mnt/d/arsen/research/proj')
sys.path.append('/usr3/graduate/arseneau')

from astroquery.gaia import Gaia
import WD_models

from .utils import read_stark_effect, fetch_basepath

### Functions For Analyzing Data

def bin_data(data, n):
    quantiles = np.linspace(0, 1, n + 1)
    bin_edges = np.quantile(data, quantiles[:-1])
    return np.digitize(data, bin_edges)

def calculate_bins(clean_df, binname, colname):
    bin_mean, stark, err = [], [], []
    for n in np.unique(clean_df[f"{binname}"]):
        subset = clean_df.query(f"{binname} == {n}")
        bin_mean.append(np.mean(subset[f"{colname}"]))
        stark.append(np.median(subset.vstark))
        err.append(1.2533*np.std(subset.vstark) / np.sqrt(len(subset)))
    return bin_mean, stark, err

def calculate_parameter_bins(dataframe, teffcol, loggcol, n=8):
    # bin the data
    dataframe['teff_bin'] = bin_data(dataframe[teffcol], n)
    dataframe['logg_bin'] = bin_data(dataframe[loggcol], n)
    # calculate the median parameters within bins
    teff_mean, teff_stark, teff_err = calculate_bins(dataframe, 'teff_bin', teffcol)
    logg_mean, logg_stark, logg_err = calculate_bins(dataframe, 'logg_bin', loggcol)
    return (teff_mean, teff_stark, teff_err), (logg_mean, logg_stark, logg_err)

def windowdata(source = '1d_da_nlte', nlteargs = {'coresize' : 15, 'lines' : 'ab'}):
    window, vstark, e_vstark = [], [], []
    shift = np.array([ 0.        ,  7.77777778, 15.55555556, 23.33333333, 31.11111111,
                        38.88888889, 46.66666667, 54.44444444, 62.22222222, 70.        ])
    for i in range(10):
        df = read_stark_effect(lteargs = {'model' : source, 'lines' : 'abgd', 'window' : i}, 
                               nlteargs = nlteargs, sigmaclip=False)
        window.append(i)
        vstark.append(np.median(df.vstark))
        e_vstark.append(1.2533*np.std(df.vstark) / np.sqrt(len(df)))
    window = shift + 30
    return window, vstark, e_vstark

def windowcomparison():
    window, vstark, e_vstark = [], [], []
    shift = np.array([ 0.        ,  7.77777778, 15.55555556, 23.33333333, 31.11111111,
                        38.88888889, 46.66666667, 54.44444444, 62.22222222, 70.        ])
    for i in range(10):
        df = pd.read_csv(f'{fetch_basepath()}/data/comparison/1d_da_nlte/abgd/window_{i}.csv')
        df = df.query("nlte_logg > 7.1 and nlte_logg < 8.9 and nlte_e_rv < 10 and nlte_redchi < 15")
        df = df.query("lte_logg > 7.1 and lte_logg < 8.9 and lte_e_rv < 10 and lte_redchi < 15")
        window.append(i)
        vstark.append(np.median(df.vstark))
        e_vstark.append(1.2533*np.std(df.vstark) / np.sqrt(len(df)))
    window = shift + 30
    return window, vstark, e_vstark

def gravz_from_logg_teff(loggarray, teffarray, Hlayer = 'H'):
    mass_sun = 1.9884e30
    radius_sun = 6.957e8
    newton_G = 6.674e-11
    pc_to_m = 3.086775e16
    speed_light = 299792458 #m/s

    font_model = WD_models.load_model('f', 'f', 'f', Hlayer)
    g_acc = (10**font_model['logg'])/100
    rsun = np.sqrt(font_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    logg_teff_to_r = WD_models.interp_xy_z_func(x = font_model['logg'], y = 10**font_model['logteff'],\
                                                z = rsun, interp_type = 'linear')
    
    radius = logg_teff_to_r(loggarray, teffarray) * radius_sun
    rv = (10**loggarray * radius) / (100 * speed_light)
    return rv*1e-3