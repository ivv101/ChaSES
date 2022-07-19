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
#     display_name: Python [conda env:p39] *
#     language: python
#     name: conda-env-p39-py
# ---

# same as test2, added Nikolay Oskolkov https://towardsdatascience.com/how-to-cluster-in-high-dimensions-4ef693bacc6

# + active=""
# environment.yml
#
# channels:
#   - conda-forge
# dependencies:
#   - python=3.9
#   - bokeh
#   - pandas
#   - scikit-learn
#   - shapely
#   - alphashape
#   - astropy
#   - bs4
#   - python-magic
#   - kneed
#   - hdbscan
#   - pip
#   - pip:
#     - pca
# -

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


# +
with open('bokeh_modules.py', 'wt') as _:
    _.write(
'''
from bokeh.embed import file_html, json_item, autoload_static, components
from bokeh.events import Tap
from bokeh.io import curdoc, output_notebook, export_png, export
from bokeh.layouts import layout, column, row, gridplot
from bokeh.models import ColumnDataSource, CustomJS, Slider, Legend, \
        Button, CheckboxButtonGroup, RadioButtonGroup, RadioGroup, CheckboxGroup, Label, Spacer, Title, Div, \
        PanTool, WheelZoomTool, SaveTool, ResetTool, HoverTool, TapTool, \
        BasicTicker, Scatter, CustomJSHover, FileInput, Toggle, TableColumn, DataTable, TextAreaInput, \
        Panel, Tabs, DateFormatter, LogColorMapper, LinearColorMapper, ColorBar, Select, PreText, \
        HTMLTemplateFormatter, NumberFormatter, ScientificFormatter
from bokeh.plotting import figure, output_file, show, save
from bokeh.resources import CDN
from bokeh.themes import Theme
from bokeh.util.compiler import TypeScript
from bokeh.util.hex import hexbin
from bokeh.transform import linear_cmap, log_cmap
''')
import bokeh_modules as bk 
import importlib
importlib.reload(bk)
import bokeh.palettes as bkp

if is_interactive():
    bk.output_notebook()

# +
import os
# import os.path
import numpy as np
import pandas as pd

# import scipy.stats


import itertools  
# from collections import Counter

import pickle as pkl

import json
from glob import glob
import re

# import cupy as cp

# from cuml.cluster import DBSCAN
# from cuml.metrics.cluster import silhouette_score
# from cuml.metrics.cluster import silhouette_samples

# import extended_library as ext_lib
import cxo_cluster_4_library as cxo_lib

import app1_library as app1_lib

# runs only once, safe to execute when in different folder
try:
    notebook_dir # does a exist in the current namespace
except NameError:
    notebook_dir = os.getcwd()
    
notebook_dir    


# -

def get_table(data):

    '''
    data is dict
    '''

    props = ['color', 'silhouette', 'area', 'n-n_bkg', 'signif.', 'sigmas', 'x', 'y', 'ra', 'dec']

    tbl_data = {k: data[k] for k in props}

    tbl_data['n-n_bkg'] = [int(_) for _ in tbl_data['n-n_bkg']]

    color_template = '''             
        <p style="color:<%= 
            (function f(){                    
                    return(color)
                }()) %>;"> 
            <%= "&#9608;" %>
        </p>
        '''

    cols = []
    for _ in tbl_data.keys():

        if _ == 'color':
            frmt = bk.HTMLTemplateFormatter(template=color_template)
        elif _ == 'n-n_bkg':
            frmt = bk.NumberFormatter()       
        elif _ in ['area', 'signif.']:
            frmt = bk.ScientificFormatter(precision=2, power_limit_low=-2)   
        elif _ in ['ra', 'dec']:
            frmt = bk.NumberFormatter(format='0.0000')                  
        else:
            frmt = bk.NumberFormatter(format='0.00')    

        c = bk.TableColumn(field=_, title=_, formatter=frmt)    

        cols.append(c)  

    return tbl_data, cols   


# +
cache_dir = 'cache' # 'app_cache'

fits_dir = 'data' #'/mnt/data_ssd/holes'

obsids = np.sort([int(_) for _ in os.listdir(fits_dir)]).astype(str).tolist()

n_max = 20_000   

# +
# bk.curdoc().clear()

# +
# from sklearn import datasets
# n_samples = 10_000
# noisy_moons_X, noisy_moons_y = datasets.make_moons(n_samples=n_samples, noise=0.)

# no_structure = np.random.rand(n_samples, 2)

# scaled_xy = ext_lib.scale(*noisy_moons_X.T)

# X = np.transpose([scaled_xy['x'], scaled_xy['y']])

# X = np.concatenate([X, no_structure])

# X_moon = (X - X.min(0)) / (X.max(0) - X.min(0))

# p = bk.figure()
# p.scatter(*X_moon.T)
# bk.show(p)

# +
# X, db = pkl.load(open('tmp.pkl', 'rb'))
                
# silhs = app1_lib.get_silhouette_samples(X, db)  
# silhs

# +
# glb = ''

# +
get_folders = lambda x : np.sort([int(_) for _ in os.listdir(f'{fits_dir}/{x}')]).astype(str).tolist()

clusters_info = []

def modify_doc(doc):
    
    msg = bk.PreText(text='')

    select_obsid = bk.Select(title='obsid', 
                             value='755' if '755' in obsids else obsids[0], 
                             options=obsids,
                             width=100)
    
    ccds = get_folders(select_obsid.value)

    select_ccd = bk.Select(title='ccd', 
                           value='7' if '7' in ccds else ccds[0], 
                           options=ccds,
                           width=50)

    def select_obsid_callback(attr, old, new):

        ccds = get_folders(new)

        select_ccd.options = ccds  
        select_ccd.value = ccds[0] 

    select_obsid.on_change('value', select_obsid_callback)
         
    cb_group = bk.CheckboxButtonGroup(labels=['holes', 'n_max', 'cache'], active=[0, 1], width=100)
    
    len_pre = bk.PreText(text='')
    
    eps_slider = bk.Slider(start=2, end=4, value=2.86, step=.1, title='eps', width=400)
    
    slider_min_samples = bk.Slider(start=30, end=60, value=46, step=2, title='min_samples', width=400)
    
    pallettes_dict = {'Viridis256': bkp.Viridis256, 'Spectral11': bkp.Spectral11}
    
    pals = list(pallettes_dict.keys())
    
    select_pallette = bk.Select(title='pallette', 
                           value=pals[0], 
                           options=pals,
                           width=200)
    
    TOOLTIPS = [
        ('index', '$index'),
        ('(x,y)', '@center_of_mass'),
        ('silhouette', '@silhouette')
    ]
        
    p = bk.figure(title='', tools=['hover'])
    
    p.toolbar.autohide = True
    
    hov = p.select_one(bk.HoverTool)
    
    hov.tooltips = TOOLTIPS
        
    p.x_range.start = p.y_range.start = 0
    p.x_range.end = p.y_range.end = 1    
    p.x_range.range_padding = p.y_range.range_padding = 0
    
    opacity_slider = bk.Slider(start=0, end=1, value=0.5, step=.01, title='opacity', width=100)
    
    img = p.image(image=[], x=0, y=0, dw=1, dh=1, palette=pallettes_dict[select_pallette.value], level='image')
        
    clus_source = bk.ColumnDataSource({'xs': [], 'ys': [], 'color': []})
    
    clus_source_filtered = bk.ColumnDataSource({'xs': [], 'ys': [], 'color': []})
    
    clus = p.multi_polygons(xs='xs', ys='ys', color='color', source=clus_source_filtered, fill_alpha=opacity_slider.value)   
    
    hov.renderers = [clus]
    
    text_source = bk.ColumnDataSource(dict(x=[], y=[], text=[]))

    txts = p.text(x='x', y='y', text='text', angle=0, text_color='white', source=text_source)
    
    tbl_source = bk.ColumnDataSource()

    tbl = bk.DataTable(source=tbl_source)    
        
    opacity_slider.js_link('value', clus.glyph, 'fill_alpha')
    
    alpha_slider = bk.Slider(start=0, end=10, value=1, step=.01, title='alpha', width=100)
        
    def alpha_slider_callback(attr, old, new):
        
        data = clus_source.data.copy()  
        
        hull_dict = app1_lib.get_hull(data['clusters'], new)
        
        data.update(hull_dict)        
        
        data = pd.DataFrame(data)
            
        clus_source.data = data.copy()
                        
        clus_source_filtered.data = data[data['sigmas'] >= min_sigma_slider.value]
            
    alpha_slider.on_change('value_throttled', alpha_slider_callback)
        
    p.grid.visible = False
    
    # p.toolbar_location = None
    p.axis.visible = False
    
    p.border_fill_color = 'white' #border_fill_color
    p.title.background_fill_color = 'white'
        
    def select_pallette_callback(attr, old, new):
        
        img.glyph.color_mapper.palette = pallettes_dict[new]
                     
    select_pallette.on_change('value', select_pallette_callback)
    
    X_source = bk.ColumnDataSource({'X': [], 'db': []})
        
    sigma_slider = bk.Slider(start=0, end=3, value=1, step=0.01, title='sigma', width=100)
    nbins_slider = bk.Slider(start=1, end=200, value=100, step=1, title='nbins', width=100)
                
    def nbins_sigma_slider_callback(attr, old, new):
        
        # k = scipy.stats.gaussian_kde([x,y], sigma)
        # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))    
        # d = zi.reshape(xi.shape).T
        
        H, bkg_dens = app1_lib.nbins_sigma_func(X_source.data['X'], nbins_slider.value, sigma_slider.value)
              
        img.data_source.data['image'] = [H.T]
                
    sigma_slider.on_change('value', nbins_sigma_slider_callback)                       
    nbins_slider.on_change('value', nbins_sigma_slider_callback)   
    
    min_sigma_slider = bk.Slider(start=0, end=10, value=5, step=1, title='min sigma', width=100)
    
    def min_sigma_slider_callback(attr, old, new):
        
        data = pd.DataFrame(clus_source.data)
        
        new_data = data[data['sigmas'] >= new]
        
        clus_source_filtered.data = new_data
        
        text_source.data = {
            'x': new_data['x_scaled'],
            'y': new_data['y_scaled'],
            'text': np.arange(len(new_data['clusters'])).tolist()            
        }
        
        tbl_source.data, tbl.columns = get_table(new_data) 
                
    min_sigma_slider.on_change('value', min_sigma_slider_callback)     
    
    pal_reverse_checkbox = bk.CheckboxGroup(labels=['reverse'], active=[])
    
    def pal_reverse_checkbox_callback(attr, old, new):
        
        _ = pallettes_dict[select_pallette.value]
                
        img.glyph.color_mapper.palette = _[::-1] if len(new)==1 else _
            
    pal_reverse_checkbox.on_change('active', pal_reverse_checkbox_callback)
            
    apply_button = bk.Button(label='Apply', button_type='success', width=100)     
   
    def apply_button_callback():
        
        # global glb
        
        obsid = select_obsid.value
        ccd = select_ccd.value
        
        p.title = f'{obsid}/{ccd}'
        
        msg.text = f'{cb_group.active}'
                       
        holes = True if 0 in cb_group.active else False        
        n_lim = True if 1 in cb_group.active else False
        
        args_func = {
            'eps': eps_slider.value, 
            'min_samples': slider_min_samples.value
        }
                        
        res = app1_lib.process_ccd(obsid, ccd, holes=holes, n_lim=n_lim, n_max=n_max, args_func=args_func, 
                          nbins=nbins_slider.value, 
                          sigma=sigma_slider.value,
                          alpha=alpha_slider.value,
                          fits_dir=fits_dir)    
                
        if res=='empty':
            img.data_source.data['image'] = []            
            empty = {'xs': [], 'ys': [], 'color': []}            
            clus_source.data = empty            
            clus_source_filtered.data = empty            
            msg.text = f'{obsid}_{ccd} empty'
            
            return 'empty'
        
        new_data = res

        X = new_data['X']
        len_X_orig = new_data['len_X_orig']
        db = new_data['db']
        n_clusters = new_data['n_clusters']
        bkg_dens = new_data['bkg_dens']
        clusters = new_data['clusters']
        H = new_data['H']
        
        img.data_source.data['image'] = [H.T]
                                            
        len_pre.text = f'{len(X)} events' if len_X_orig==len(X) else f'{len(X)}/{len_X_orig} events'     
                    
        X_source.data = {'X': X, 'db': db}  
        
        hls = '_holes' if 0 in cb_group.active else '' 
        
        pkl.dump(X, open(f'{cache_dir}/X_{obsid}_{ccd}{hls}.pkl', 'wb'))
                                   
        n_clusters_text = f'\n{n_clusters} clusters'
        if n_clusters==1:
            n_clusters_text = n_clusters_text[:-1]
        
        len_pre.text += n_clusters_text
                       
        palette = itertools.cycle(bkp.Category20[20]) 
        
        # start = 0
        # palette = itertools.islice(palette, start, None)

        colors = np.array(['']*n_clusters, dtype='U7')
            
        for i in range(n_clusters): 
        
            colors[i] = next(palette)
            
        colors[0] = cxo_lib.rgb2hex(0, 255, 255) # turquoise   
        
        msg.text = f'len_X: {len(X)}, bkg: {bkg_dens}'
                                    
        new_data.update({'clusters': clusters, 'color': colors.tolist()})
        
        excl = ['X', 'len_X_orig', 'db', 'n_clusters', 'bkg_dens', 'H']
        
        h = [_ for _ in new_data.keys() if _ not in excl]                
        new_data = pd.DataFrame(new_data, columns=h).fillna('')
                                
        clus_source.data = new_data.copy()
        
        new_data = new_data[new_data['sigmas'] >= min_sigma_slider.value]
        
        clus_source_filtered.data = new_data
        
        text_source.data = {
            'x': new_data['x_scaled'],
            'y': new_data['y_scaled'],
            'text': np.arange(len(new_data['clusters'])).tolist()            
        }
                        
        tbl_source.data, tbl.columns = get_table(new_data) 
                        
        # msg.text = f'{obsid}_{ccd} done'
        
        return 'success'
                        
    apply_button.on_click(apply_button_callback)
        
    process_all_button = bk.Button(label='Process all', button_type='primary', width=100) 
    
    def process_all_button_callback():
        
        # global glb
        
        bad_obsids_ccd = []
        
        all_dir = f'{cache_dir}/all'
        
        loop_progress = cxo_lib.loop_class(obsids)
        
        for obsid in obsids:
            ccds = get_folders(obsid)
            
            # process_all_button.label = f'{i}/{len(obsids)}''
            
            for ccd in ccds:
                
                select_obsid.value = obsid
                select_ccd.value = ccd
                
                try:
                    res = apply_button_callback()
                except:
                    bad_obsids_ccd.append([obsid, ccd, 'error'])
                    continue
                
                if res == 'empty':
                #     # print(f'{obsid}_{ccd} empty')
                    bad_obsids_ccd.append([obsid, ccd, 'empty'])
                    continue
                
                hls = '_holes' if 0 in cb_group.active else ''
                                                
                os.system(f'cd {notebook_dir}; mkdir -p {all_dir}')
                
                tbl_json = json.loads(pd.DataFrame(tbl_source.data).to_json(orient='split', index=False, indent=4))
                json.dump(tbl_json, open(f'{all_dir}/{obsid}_{ccd}{hls}.json', 'wt'))
                
                # glb = tbl_json
                  
                _ = clus_source_filtered.data.copy()    
                pkl.dump(_, open(f'{all_dir}/{obsid}_{ccd}{hls}.pkl', 'wb'))
                
                bk.export_png(p, filename=f'{all_dir}/{obsid}_{ccd}{hls}.png')  
                
                # print(f'{obsid}_{ccd} done')
                
            loop_progress()
            
        pkl.dump(bad_obsids_ccd, open(f'{cache_dir}/bad_obsids_ccd.pkl', 'wb'))    
        
    process_all_button.on_click(process_all_button_callback)    
        
    row1 = bk.row([select_obsid, select_ccd, len_pre], width=800)
    
    settings_column = bk.column([msg, bk.row(apply_button, process_all_button), row1, cb_group, eps_slider, slider_min_samples, tbl])
    
    layout = bk.row([bk.column([p, bk.row(select_pallette, pal_reverse_checkbox, min_sigma_slider), 
                                bk.row(opacity_slider, alpha_slider), bk.row(nbins_slider, sigma_slider)]), settings_column])
    
    doc.add_root(layout)
    
    doc.title = 'Sliders'
    
    apply_button_callback()
    
    
    # bk.export_png(p, filename=f'{save_dir}/{obsid_ccd}{hls}_blur.png')

# +
# curdoc().add_root(row(plot, controls))
# curdoc().title = "Weather"
# -

if is_interactive():
    bk.show(modify_doc, notebook_url='localhost:1111', port=8905)
else:
    modify_doc(bk.curdoc())

# +
# https://mybinder.org/v2/gh/ivv101/tst/main?urlpath=/proxy/5006/bokeh-app
# -














