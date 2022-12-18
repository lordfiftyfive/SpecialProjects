# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 15:26:33 2021

@author: subar
"""
from alpaca_trade_api.stream import Stream
import alpaca_trade_api as tradeapi
import time
from numba import jit
import numpy as np 
import cupy as cp
import pandas as pd
import tensorflow as tf
import time
import py_vollib.black_scholes
import markowitzify
#pd.isnull(np.datetime64('NaT'))
#from pipeline_live.data.alpaca.factors import AverageDollarVolume
#from pipeline_live.data.alpaca.pricing import USEquityPricing

from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import (RSI, AverageDollarVolume, BollingerBands, DailyReturns,
ExponentialWeightedMovingAverage, MACDSignal, MaxDrawdown, Returns, RollingPearson,RollingSpearman,VWAP,CustomFactor)
from zipline.pipeline.filters import AtLeastN
from zipline.api import (
    attach_pipeline,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    date_rules,
    time_rules
)
#note alpaca has all of the same factors as zipline
"""
from pipeline_live.data.alpaca.factors import (
    RSI,MACDSignal,VWAP,AverageDollarVolume, SimpleMovingAverage,
)
"""
#from pylivetrader.api import order_target, symbol
#import numba_scipy
import alphalens as al
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.math import psd_kernels as tfk
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#from prophet.plot import plot_plotly, plot_components_plotly
#from prophet import Prophet
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent

from alpaca_trade_api.rest import TimeFrame
import polars as pl
import statistics
import sys
import time
import os
import pyfolio as pf
from datetime import datetime, timedelta
from pytz import timezone
from alpaca_trade_api.rest_async import gather_with_concurrency, AsyncRest
from tiingo import TiingoClient
from websocket import create_connection
import simfin as sf
from simfin.names import *
#from tiingo import TiingoWebsocketClient

#hyper parameter optimization
import keras_tuner as kt
from tune_sklearn import TuneGridSearchCV
import optuna
#alpha vantage api key is A7ATL022SWSHQFYL

"""
rw = 0.4 # rolling window
span = 0.5 # Lowess span
lags = [1,2,3] # autocorrelation lag times to compute
ews = ['var','sd','ac','skew','kurt','ac','smax','cf','aic'] # EWS to compute (let's do all of them)
ham_length = 80 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 #

ews_dic = ewstools.ews_compute(df_traj['x'], 
                          roll_window = rw, 
                          span = span,
                          lag_times = lags, 
                          ews = ews,
                          upto=tcrit)

"""
from zipline.api import (
            attach_pipeline,
            date_rules,
            order_target_percent,
            pipeline_output,
            record,
            schedule_function,
        )
config = {}

# To reuse the same HTTP Session across API calls (and have better performance), include a session key.
config['session'] = True

config['api_key'] = "292879b75e4ce811056b97bbbf8d72ca04e20de8"

client = TiingoClient(config)
API_KEY = "AKMLXVPKAQU1M9FWDYPZ"
API_SECRET = "oaCSrDz2Be4GYeEI8SSQGYEpdS1AiJvYOn3eUjcO"#oaCSrDz2Be4GYeEI8SSQGYEpdS1AiJvYOn3eUjcO
APCA_API_BASE_URL = "https://api.alpaca.markets"
alpaca = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
api = alpaca
assets = api.list_assets()
Scalar = MinMaxScaler(feature_range=(-1,1))
pca = PCA()
index = 0
api_time_format = '%Y-%m-%dT%H:%M:%S.%f-04:00'
stocks_to_hold = 150
ws = create_connection("wss://api.tiingo.com/iex")

from zipline.api import order_target, record, symbol
import matplotlib.pyplot as plt

active_assets = api.list_assets(status='active')
universe = AverageDollarVolume(window_length = 30).top(20)
#my_factor = MyFactor()
"""
pipeline = Pipeline(
    columns = {
            'MyFactor' : my_factor,
            'Sector' : ZiplineTraderSector(),
    }, domain=US_EQUITIES, screen=universe
)
"""
class ATLAS(object):
    def __init__(self):
        #print("asdf")
        self.base_url = 'https://api.alpaca.markets'#'https://paper-api.alpaca.markets'
        self.data_url = 'https://data.alpaca.markets'
        assets = api.list_assets()
        self.assets = [asset for asset in assets if asset.tradable ]
        self.batch_size = 250# The maximum number of stocks to request data for
        self.window_size = 1460 # The number of days of data to consider

    subscribe = {
            'eventName':'subscribe',
            'authorization':'292879b75e4ce811056b97bbbf8d72ca04e20de8',
            #see https://api.tiingo.com/documentation/websockets/iex > Request for more info
            'eventData': {
                'thresholdLevel':5
          }
    }
    @jit
    def data(self):
        print("data importation")
        
    @jit
    def make_pipeline():
        rsi = RSI()
        return Pipeline(
            columns={
                "longs": rsi.top(3),
                "shorts": rsi.bottom(3),
            },
        )
    @jit
    def preprocessing(self):
        print("commencing data preprocessing")
    @jit
    def scholes(self):
        print("commencing options trading")
    @jit
    def opt(self):
        print("Initiating Hyperparameter optimization")
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
          hp.Choice('units', [8, 16, 32]),
          activation='selu'))
        model.add(tf.keras.layers.Dense(1, activation='relu'))
        model.compile(loss='mse')

    @jit
    def order(self):
        """
        #for i in stocks:
            x,y = data[i]
            core(x,y)
            rfinal = self.rfinal
            if rfinal >= 0.2 and ub < 0.25
                #we will put the order here where the size of the order will be 1*(1/ub)
            elif rfinal < 0.2 and ua < 0.25:
                #open short position
            elif rfinal > 0.2 and ub > 0.25 or rfinal < 0.2 and ua > 0.25:
                #close all short and long positions
            else:
                print('Waiting...')
                time.sleep(1)
                
        """
        print("Warning: trading is starting")
        # Filter the assets down to just those on NASDAQ.
    
        """
     
        """
        
        """
        Currently three ideas:
            
        1. ATLAS: Uses traditional generic deep learning to predict stocks 
        2. NEMESIS: Pairs trading with hurst and limits on momentum
        3. NEPTUNE: Uses rogue wave models for stock market
        4. RANDOM: Baseline algorithm buys and sells random stocks at random time intervals (baseline)
        
        Here is how the Atlas system is going to work:
            
        we are going to use a deep learning algorithm to try to predict the following days stock price 
        
        
        """
        api.submit_order(
            symbol='AAPL',
            qty=0,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        
        # Submit a limit order to attempt to sell 1 share of AMD at a
        # particular price ($20.50) when the market opens
        api.submit_order(
            symbol='AMD',
            qty=0,
            side='sell',
            type='limit',
            time_in_force='opg',
            limit_price=20.50
        )
        
        #shorting
        
        symbol = 'TSLA'
        
        # Submit a market order to open a short position of one share
        order = api.submit_order(symbol, 1, 'sell', 'market', 'day')
        print("Market order submitted.")
        
        # Submit a limit order to attempt to grow our short position
        # First, get an up-to-date price for our symbol
        symbol_bars = api.get_barset(symbol, 'minute', 1).df.iloc[0]
        symbol_price = symbol_bars[symbol]['close']
        # Submit an order for one share at that price
        order = api.submit_order(symbol, 1, 'sell', 'limit', 'day', symbol_price)
        print("Limit order submitted.")
        
        # Wait a second for our orders to fill...
        print('Waiting...')
        time.sleep(1)
        
        # Check on our position
        position = api.get_position(symbol)
        if int(position.qty) < 0:
            print(f'Short position open for {symbol}')
        
        #portfolio
        
        # Get a list of all of our positions.
        portfolio = api.list_positions()
        
        # Print the quantity of shares for each position.
        for position in portfolio:
            print("{} shares of {}".format(position.qty, position.symbol))
            
#note: we Could use the D-wave for portfolio optimization            

atlas = ATLAS()
atlas.data()
