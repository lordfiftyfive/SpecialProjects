# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:49:42 2022

@author: subar
"""

from sklearn.preprocessing import Binarizer
import argparse
import logging
import numpy as np
import torch
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.contrib.cevae import CEVAE
import quandl
from numba import jit
from sklearn.model_selection import train_test_split
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)
import matplotlib.pyplot as plt 
Scalar = MinMaxScaler(feature_range=(0,1))
scalar = MinMaxScaler(feature_range=(-1,1))
@jit
def data():
    """
    This implements the generative process of [1], but using larger feature and
    latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    """

    #inflation = quandl.get("YALE/SP_CPI", authtoken="DNMZo2iRzVENxpxqHBKF")    
    #y = pd.read_excel('D:\\data\\API_NY.GDP.MKTP.KD_DS2_en_excel_v2_3358761.xls',sheet_name='Data')
    #xx = pd.read_csv('D:\\data\\API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_3852477.csv',on_bad_lines='skip')
    #x = pd.read_excel('D:\\data\\API_IT.NET.USER.ZS_DS2_en_excel_v2_3358811.xls',sheet_name='Data')
    #inflation= pd.read_csv('D:\\data\\API_FP.CPI.TOTL_DS2_en_csv_v2_3852520.csv',on_bad_lines='skip')
    x = quandl.get("RATEINF/CPI_USA", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1947-01-01", end_date="2021-10-01")
    y = quandl.get("FRED/GDP", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly")
    
    #y = y.drop(['Country Name','Indicator Name','Indicator Code'],1)
    #,'UMC','LMC','LIC','HIC'])]#,'WLD','USA'])]#CHN
    #,'UMC','LMC','LIC','HIC'])]#,'WLD','USA'])]#CHN
    #y = y[y['Country Code'].isin(['USA'])]#,'UMC','LMC','LIC','HIC'])]#,'WLD','USA'])]#CHN
    #y = y.pivot_table(y,columns='Country Code')
    #x = x.pivot_table(x,columns='Country Code')
    #y = y.fillna(0)
    #x = np.array(x)
    #y = np.array(y)
    #x = x.pct_change()
    
    #y = y.pct_change()
    #x = x.fillna(0.1)
    #y = y.fillna(0.08)
    print(x.shape)
    print(y.shape)
    #y = scalar.fit_transform(y)
    #binarizer = Binarizer(threshold=0.03)
    #x = binarizer.fit_transform(x)
    #y = y['Value']
    #x = us_inflation['Value']
    #x = x.reset_index(drop=True)
    #y = y.reset_index(drop=True)
    #x = x.drop(['Year'],1)
    #y = y.drop(['Date'],1)
    
    #print(a)
    #x= np.array(us_inflation)
    #x = np.log10(x)
    #y = np.log10(y)
    #y = y.astype(np.float64)
    
    x = pd.DataFrame(x)#,columns=['x'])
    #y = pd.DataFrame(y)#,columns=['y'])
    #x = x.astype(np.float64)
    #us_inflation = us_inflation.astype(np.float64)
    print(x.shape)
    print(y.shape)
    dat = pd.concat([x,y],1)
    dat = dat.fillna(0)
    #print(dat.dtype())
    print(y)
    
    print(dat)
    print("fasd")
    dat = pd.DataFrame(dat)#.astype(np.float64)
    dat = np.array(dat)
    N = dat.shape[1]
    var_names = [r'$X^{%d}$' % j for j in range(N)]
    dat = pp.DataFrame(dat,var_names=var_names)
    
    parcorr = ParCorr(significance='analytic')
    
    pcmci = PCMCI(dataframe=dat, 
              cond_ind_test=parcorr,
              verbosity=1)
    print("asdf")
    correlations = pcmci.run_bivci(tau_max=20, val_only=True)['val_matrix']
    setup_args = {'var_names':var_names,
              'figsize':(10, 6),
              'x_base':5,
              'y_base':.5}
    
    lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, 
                                   setup_args=setup_args)
    
    lpcmci = LPCMCI(dataframe=dat, 
                cond_ind_test=parcorr,
                verbosity=1)

    # Define the analysis parameters.
    tau_max = 5
    pc_alpha = 0.01
    
    # Run LPCMCI
    results = lpcmci.run_lpcmci(tau_max=tau_max,
                                pc_alpha=pc_alpha)
    #lpcmi.print_significant_links()
    #plt.plot()
    #plt.plot(s)
    plt.show()
    # Plot time series graph
    tp.plot_time_series_graph(
        figsize=(8, 8),
        node_size=0.05,
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='MCI',
        ); 
    
data()