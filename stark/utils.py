import pandas as pd
import numpy as np
import pyvo
import sys
sys.path.append('/mnt/d/arsen/research/proj')
sys.path.append('/usr3/graduate/arseneau')

from astroquery.gaia import Gaia
import WD_models

def fetch_basepath():
    return '/mnt/d/arsen/research/proj/spyquadraticstark'
    #return '/usr3/graduate/arseneau/spyquadraticstark'

def fetch_goodspypath():
    return f'{fetch_basepath()}/data/processed/good_spy.csv'

def fetch_ltepath(model, lines, window):
    return f'{fetch_basepath()}/data/coadd/lte/{model}/{lines}/window_{window}.csv'

def fetch_nltepath(coresize, lines):
    return f'{fetch_basepath()}/data/coadd/nlte/{lines}_{coresize}angstrom.csv'


def air2vac(wv):
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

def read_clean_nlte(coresize, lines):
    goodsystems = pd.read_csv(fetch_goodspypath()).FileName.values
    nlte = pd.read_csv(f'{fetch_nltepath(coresize, lines)}')
    nlte = nlte.query("nlte_logg > 7.1 and nlte_logg < 8.9 and nlte_e_rv < 10 and nlte_redchi < 10")
    return  nlte.loc[nlte['filename'].isin(goodsystems)]

def read_clean_lte(model, masksize, lines, window):
    goodsystems = pd.read_csv(fetch_goodspypath()).FileName.values
    #lte = pd.read_csv(f'{fetch_ltepath(model, masksize, lines, window)}')
    lte = pd.read_csv(f'../data/processed/spy/mask_nlte_1d_h{lines}/spy_h{lines}_{window}.csv')
    print('temporarily using the wrong lte path! remember to change this.')
    lte = lte.query("logg > 7.1 and logg < 8.9 and e_radial_velocity < 10 and redchi < 10")
    return  lte.loc[lte['filename'].isin(goodsystems)]

def read_stark_effect(lteargs, nlteargs = None, sigmaclip=True):
    nlteargs = {'coresize' : 15, 'lines' : 'ab'} if nlteargs is None else nlteargs
    nlte = read_clean_nlte(**nlteargs)
    lte = read_clean_lte(**lteargs)
    df = pd.merge(nlte, lte, on='filename')
    # merge in Gaia IDs
    goodsystems = pd.read_csv(fetch_goodspypath())
    df = pd.merge(goodsystems, df, left_on='FileName', right_on='filename')
    df = df.drop(columns=['FileName'])
    # compute stark velocities
    df['vstark'] = df['radial_velocity'] - df['nlte_rv']
    df['e_vstark'] = np.sqrt(df['e_radial_velocity']**2 + df['nlte_e_rv']**2)
    # return
    if sigmaclip:
        sigma_to_clip = 3
        mean = np.mean(df.vstark)
        stddev = np.std(df.vstark)
        return df.query(f"abs(vstark - {mean}) < {sigma_to_clip * stddev}")
    else:
        return df
    
def merge_gaia(dataframe, sourcecol):
    sources = dataframe[sourcecol].values
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select *
            from \"J/MNRAS/508/3877/maincat\" as ngf
            where ngf.GaiaEDR3 in {tuple(sources)}"""
    gaia_data = tap_service.search(QUERY).to_table().to_pandas()
    mergeddata = pd.merge(dataframe, gaia_data, left_on = sourcecol, right_on = 'GaiaEDR3')
    return mergeddata
    
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

        
