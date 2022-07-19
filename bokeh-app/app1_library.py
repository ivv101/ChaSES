# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:rapids]
#     language: python
#     name: conda-env-rapids-py
# ---

# +
import numpy as np

import alphashape

from scipy.stats import skellam
from scipy.special import ndtri
from scipy.ndimage import gaussian_filter

from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats 

from collections import Counter

# from cuml.cluster import DBSCAN
# from cuml.metrics.cluster import silhouette_samples

from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import silhouette_samples

# from sklearn.cluster import DBSCAN, OPTICS #, KMeans, SpectralClustering
# from sklearn.preprocessing import StandardScaler

import extended_library as ext_lib


# -

def db_sort(db, n_min=1):
    
    '''
    delete clusters < n_min; clusters ordered so that 0 - largest cluster
    '''
    
    n_clusters = np.max(db) + 1    
    if n_clusters == 0:
        return db

    a = list(Counter(db).items())
    a = np.array([_ for _ in a if _[0]!=-1])
    a = a[a[:, 1].argsort()][::-1]
    small = [_[0] for _ in a if _[1]<n_min]
    a = dict(zip(a[:, 0], np.arange(len(a))))
    for i in small:
        a[i] = -1
    a[-1] = -1

    db = np.array([a[i] for i in db])

    return db


def get_data(obsid, ccd, fits_dir='', holes=True):
    
    hls = '_holes' if holes else ''
        
    evt2_data, head = ext_lib.process_fits(f'{fits_dir}/{obsid}/{ccd}/{obsid}_{ccd}{hls}_evt2_05_8keV.fits')

    if len(evt2_data)==0:
        print(f'{obsid}_{ccd} empty')
        # msg.text = f'{obsid}_{ccd} empty'
        return 'empty'

    xy = ext_lib.xy_filter_evt2(evt2_data)[f'ccd_{ccd}']

    scaled_xy = ext_lib.scale(*xy.T)
    
    scaled_xy['head'] = head
    
    return scaled_xy


def get_hull(clusters, alpha):

    xs = []
    ys = []
    
    areas = []

    for c in clusters:

        # edges, hull_vertices = cxo_lib.concave_hull(c, 0.2)
        # pgon = Polygon(zip(*hull_vertices)) 
        # dens = 100 * (len(c) / (pgon.area * len(X)) - 1)
        
        ashape = alphashape.alphashape(c, alpha)
        
        x, y = ashape.exterior.xy
        
        area = ashape.area

        # x, y = cxo_lib.concave_hull(c, alpha)[1]

        xs.append([[x.tolist()]])
        ys.append([[y.tolist()]])
        areas.append(area)

    return {'xs': xs, 'ys': ys, 'area': areas}   


def nbins_sigma_func(X, nbins, sigma):

    # k = scipy.stats.gaussian_kde([x,y], sigma)
    # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))    
    # d = zi.reshape(xi.shape).T

    H, xe, ye = np.histogram2d(*X.T, bins=nbins)  

    H = gaussian_filter(H, sigma=sigma)            

    mean, median, std = sigma_clipped_stats(H, sigma=3.0)
    # print((mean, median, std), bkg_dens) 

    bkg_dens = median * nbins**2

    # print('len_X:', len(X_source.data['X']))
    # print('bkg_dens:', bkg_dens)

    return H, bkg_dens


def process_ccd(obsid, ccd, holes=True, n_lim=True, n_max='all', 
                args_func={}, nbins=100, sigma=3, alpha=1, fits_dir=''):

    scaled_xy = get_data(obsid, ccd, fits_dir, holes)
    
    if scaled_xy=='empty':
        return 'empty'
    
    # import pickle as pkl
    # pkl.dump(scaled_xy, open('/home/ivv101/oyk/Extended_sources/2022/app_cache/scaled_xy.pkl', 'wb'))

    X = scaled_xy['X'].copy()

    if len(X)==0:
        return 'empty'

    len_X_orig = len(X)

    if n_lim and n_max!='all':
        np.random.shuffle(X)
        X = X[:np.min([n_max, len_X_orig])]

    args = args_func.copy()    

    if 'eps' in args:
        eps0 = float(1 / np.sqrt(len(X)))                       
        args['eps'] *= eps0

    db = DBSCAN(**args).fit_predict(X)                
    db = db_sort(db, n_min=4)

    n_clusters = db.max() + 1

    noise = X[db==-1]
    clusters = [X[db==_].tolist() for _ in range(n_clusters)] 

    # cp.cuda.stream.get_current_stream().synchronize()

    # hulls, center of mass, area, silhouette, #cluster, n-n_bkg/area, significance 

    data = {}

    xs_ys_areas = get_hull(clusters, alpha)        
    data.update(xs_ys_areas)
        
    try:
        silhs = silhouette_samples(X, db)   
        
        if silhouette_samples.__module__.split('.')[0] == 'cuml':
        
            silhs = [np.mean(silhs[db==_]).get().tolist() for _ in range(n_clusters)]  
        else:
            silhs = [np.mean(silhs[db==_]).tolist() for _ in range(n_clusters)]  
                        
    except Exception as e:
        print(e)
        print(f'no silhs for {obsid}/{ccd}')
        silhs = [-1]*n_clusters
        
    data['silhouette'] = silhs    

    H, bkg_dens = nbins_sigma_func(X, nbins, sigma)

    com = np.transpose([np.mean(c, 0).tolist() for c in clusters])  
    data['x_scaled'], data['y_scaled'] = com
    
    com = ext_lib.unscale(*com, scaled_xy['pars'])

    data['x'], data['y'] = com

    data['n-n_bkg'] = [len(c) - bkg_dens * a for c, a in zip(clusters, data['area'])]

    data['signif.'] = [1 - skellam.cdf(x, len(X) * a, bkg_dens * a) for x, a in zip(data['n-n_bkg'], data['area'])]

    data['sigmas'] = [ndtri(1-_/2) for _ in data['signif.']]

    h = scaled_xy['head']

    w = WCS(naxis=2)
    w.wcs.crpix = [h['TCRPX11'], h['TCRPX12']]
    w.wcs.cdelt = [h['TCDLT11'], h['TCDLT12']]
    w.wcs.crval = [h['TCRVL11'], h['TCRVL12']]
    w.wcs.ctype = [h['TCTYP11'], h['TCTYP12']]
    w.wcs.cunit = [h['TCUNI11'], h['TCUNI12']]
    w.wcs.radesys = 'ICRS'
    w.wcs.mjdobs = h['MJD-OBS']
    w.wcs.dateobs = h['DATE-OBS']

    data['ra'], data['dec'] = w.wcs_pix2world(com.T, 1).T

    data.update(dict(zip(['X', 'len_X_orig', 'db', 'n_clusters', 'bkg_dens', 'clusters', 'H'], 
                         [X, len_X_orig, db, n_clusters, bkg_dens, clusters, H])))

    return data # no filtering by sigma_min
