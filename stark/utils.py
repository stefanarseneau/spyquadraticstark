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
    return f'{fetch_basepath()}/data/goodcoadds.csv'

def fetch_ltepath(model, lines, window):
    return f'{fetch_basepath()}/data/coadd/lte/{model}/{lines}/window_{window}.csv'

def fetch_nltepath(coresize, lines):
    return f'{fetch_basepath()}/data/coadd/nlte/{lines}_{coresize}angstrom.csv'
    #return f'{fetch_basepath()}/data/processed/nlte/{coresize}angstrom/{lines}.csv'


def air2vac(wv):
    _tl=1.e4/np.array(wv)
    return (np.array(wv)*(1.+6.4328e-5+2.94981e-2/\
                          (146.-_tl**2)+2.5540e-4/(41.-_tl**2)))

def read_clean_nlte(coresize, lines):
    nlte = pd.read_csv(f'{fetch_nltepath(coresize, lines)}')
    nlte = nlte.query("nlte_logg > 7.1 and nlte_logg < 8.9 and nlte_e_rv < 10 and nlte_redchi < 15")
    return  nlte

def read_clean_lte(model, lines, window):
    lte = pd.read_csv(f'{fetch_ltepath(model, lines, window)}')
    #lte = pd.read_csv(f'../data/processed/spy/mask_nlte_1d_h{lines}/spy_h{lines}_{window}.csv')
    #print('temporarily using the wrong lte path! remember to change this.')
    lte = lte.query("lte_logg > 7.1 and lte_logg < 8.9 and lte_e_rv < 10 and lte_redchi < 15")
    return  lte

def read_stark_effect(lteargs, nlteargs = None, sigmaclip=True):
    nlteargs = {'coresize' : 15, 'lines' : 'ab'} if nlteargs is None else nlteargs
    nlte = read_clean_nlte(**nlteargs)
    lte = read_clean_lte(**lteargs)
    df = pd.merge(nlte, lte, on='source_id')
    # merge in Gaia IDs
    goodsystems = pd.read_csv(fetch_goodspypath())
    df = pd.merge(goodsystems, df, left_on='SOURCE_ID', right_on='source_id')
    #df = df.drop(columns=['FileName'])
    # compute stark velocities
    df['vstark'] = df['lte_rv'] - df['nlte_rv']
    df['e_vstark'] = np.sqrt(df['lte_e_rv']**2 + df['nlte_e_rv']**2)
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
    


        
