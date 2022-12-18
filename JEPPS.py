# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:50:28 2020

@author: subar
"""

from tensorflow import keras 
#import gpflow
import tensorflow as tf
#note: first time around runtime is going to throw a dependency error. What you have to do is run both the installation and importing cells then restart the runtime and run them again
import tensorflow_probability as tfp
import quandl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import cupy as cp
import shap
#import rpy2
#from rpy2.robjects.packages import importr
from sklearn.decomposition import PCA, KernelPCA
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
#from tensorflow_probability import sts
#import modin.pandas as pd
import pandas as pd
#import bootstrapped.bootstrap as bs
#import bootstrapped.stats_functions as bs_stats
from scipy.stats import ttest_ind, ttest_ind_from_stats,chisquare, mstats,levene
import seaborn as sns
import statsmodels.api
from statsmodels.tsa.seasonal import STL
import time
from sklearn.manifold import TSNE
from statsmodels.tsa.seasonal import seasonal_decompose
from hurst import compute_Hc, random_walk
import scipy.integrate as integrate
#from stldecompose import decompose
from tensorflow.python.client import device_lib
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
#from attention import Attention 
from numba import jit
from hyperspy.signals import Signal1D,Signal2D
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
#import econml
import umap
import nolds
#import EntropyHub as EH

from scipy import signal
from scipy import integrate
import spectrum
from spectrum import tools
from random import gauss, seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
from lightgbm import *
from hyperopt import hp
from hyperopt import Trials
#import hfda
#deployment
#import mlflow
#mlflow is breaking because we need h5py 3.1 and protobuf 3.14
#hyperparameter optimization libraries. Use ONLY these for all hyperparameter optimization tasks

import kerastuner as kt
import optuna
#from tune_sklearn import TuneSearchCV
#from shaphypetune import BoostSearch, BoostRFE, BoostRFA, BoostBoruta
tf.executing_eagerly()

#from gpflow.utilities import print_summary, positive\
#print(device_lib.list_local_devices())
print(tf.__version__)
"""
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
"""
#print(tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
"""
param_dist_hyperopt = {
'max_depth': 15 + hp.randint('num_leaves', 5), 
'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}
"""
WANDB_API_KEY = '0f4ae8015eb847e04a2344b0d5777c39ef1e06a1'
quandl.ApiConfig.api_key = 'DNMZo2iRzVENxpxqHBKF'
#@jit
def data():
    #te.getCalendarData(country=['united states', 'china'], category=['imports','exports'],
                       #initDate='2017-06-07', endDate='2017-12-31',
                       #output_type='df')
    datafour = quandl.get("YALE/SPCOMP", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30", end_date="2020-09-30")# earliest date is 1960-06-30 #quandl.get("YALE/CPIQ", authtoken="DNMZo2iRzVENxpxqHBKF", collapse="quarterly", start_date="1970-12-31", end_date="2016-03-31")#quandl.get("UNAE/GDPCD_USA", authtoken="DNMZo2iRzVENxpxqHBKF", end_date="2016-12-31")# quandl.get("UNAE/GDPCD_USA", authtoken="DNMZo2iRzVENxpxqHBKF", end_date="2016-12-31")
    data_to_predict = quandl.get("FRED/GDP", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1960-01-01", end_date="2021-03-31")#quandl.get("FRBP/GDPPLUS_042619", authtoken="DNMZo2iRzVENxpxqHBKF", collapse="quarterly")#quandl.get("FRBP/GDPPLUS_042619", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff")#quandl.get("FRBP/GDPPLUS", authtoken="DNMZo2iRzVENxpxqHBKF", collapse="quarterly", start_date="1960-06-30")
    d7 = quandl.get("FRED/GDP", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30", end_date="2020-09-30")
    #datafive = quandl.get("FRED/PCETRIM1M158SFRBDAL", authtoken="DNMZo2iRzVENxpxqHBKF", collapse="quarterly", start_date="1977-02-01",end_date="2016-03-31")#quandl.get("FRED/VALEXPUSM052N", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1960-09-30")#quandl.get("WWDI/USA_NE_GDI_TOTL_CD", authtoken="DNMZo2iRzVENxpxqHBKF", start_date="1970-12-31")
    #Data_To_predict = data_to_predict.values
    DESPAIR = quandl.get("USMISERY/INDEX", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30", end_date="2020-09-30")
    Debt_data_change_of_change = quandl.get("FRED/NCBDSLQ027S", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff",collapse="quarterly",start_date="1959-09-30",end_date="2020-09-30")
    income_increase =quandl.get("FED/FU156010001_Q", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30",end_date="2020-09-30")
    consumption_per_capita = quandl.get("FRED/A794RX0Q048SBEA", transform="rdiff", collapse="quarterly", start_date="1959-09-30", end_date="2020-09-30")
    sentiment =quandl.get("UMICH/SOC1", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30", end_date="2020-09-30")
    DataSix = quandl.get("FRED/ROWFDIQ027S", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30",end_date="2018-09-30")
    
    inflation = quandl.get("RATEINF/CPI_USA", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30", end_date="2020-09-30")
    print(DataSix)
    #Data_To_Predict = np.matrix(Data_To_predict)
    print("datasix")
    #print(Data_To_Predict)
    #print(Datafour)
    #early_stop = EarlyStopping(monitor='loss',patience=5, verbose=1)
    data_to_predict = data_to_predict#[-47::]#np.reshape(Data_TO_predict, (9,6)
    
    ECOG_V = quandl.get("FRED/GDP", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30", end_date="2020-09-30")
    """
    #Ask if my model is making good predictions or whether the predicitons are due to the loss of information from the nornmalization procedure of the outputs
    
    #create a plot of the data before this ß
    #split_date= pd.Timestamp('01-01-2011')
    #train = df.loc(:split_date,[''])
    #test = df.loc(split_date:, [''])
    
    """
    #data preprocessing
    #mstats.winsorize()
    #trainingtarget =pd.DataFrame(data_to_predict)#normalizer.transform(data_to_predict)
    """
    tabnet = TabNet(num_features = train_X_transformed.shape[1],
                output_dim = 128,
                feature_dim = 128,
                n_step = 2, 
                relaxation_factor= 2.2,
                sparsity_coefficient=2.37e-07,
                n_shared = 2,
                bn_momentum = 0.9245)
    
    """
    Scalar = MinMaxScaler(feature_range=(-1,1))
    
    dataSeven = pd.DataFrame(income_increase)
    print("a")
    print(len(dataSeven))
    #datafour = preprocessing.scale(datafour)
    #trainingtarget = preprocessing.scale(trainingtarget)
    datasix = pd.DataFrame(DataSix)
    sentiment = pd.DataFrame(sentiment)
    sentiment = sentiment.fillna(0.0496)
    datafour = pd.DataFrame(datafour)
    datasix = datasix.drop(datasix.index[0])
    #print("data six")
    #print(datasix)
    
    #class statsmodels.tsa.seasonal.STL(endog, period=None, seasonal=7, trend=None, low_pass=None, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=False, seasonal_jump=1, trend_jump=1, low_pass_jump=1)¶
    
    #datafour = datafour.fillna(0)
    #data_to_predict = pd.DataFrame(data_to_predict)
    print(data_to_predict.shape)
    #data
    dta_hamilton = quandl.get("MULTPL/SP500_REAL_PRICE_MONTH", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1958-06-30", end_date="2020-09-30")#quandl.get("MULTPL/SP500_REAL_PRICE_MONTH", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff", collapse="quarterly", start_date="1959-09-30",end_date="2020-09-30")#quandl.get("MULTPL/SP500_REAL_PRICE_MONTH", authtoken="DNMZo2iRzVENxpxqHBKF", transform="rdiff",start_date="1959-12-31" end_date="2020-05-01")
    print("dta")
    #print(dta_hamilton)
    #print(dta_hamilton.shape)
    #dta_hamilton.index = pd.DatetimeIndex(dta_hamilton.index, freq="QS")
    #dta_hamilton.index = np.linspace(1871.17,2020.583,1802)
    #dta_hamilton = dta_hamilton.drop('Date',1)
    #dta_hamilton.index = pd.date_range('1959-09-30', '2020-09-30',freq="QS")#pd.DatetimeIndex(dta_hamilton.index,freq='MS')# order=4
    #dta_hamilton.index = dta_hamilton.index.asfreq('QS')
    
    mod_hamilton = sm.tsa.MarkovAutoregression(dta_hamilton, k_regimes=2, order=4, switching_ar=False,switching_trend=True,switching_variance=True)
    res_hamilton = mod_hamilton.fit()
    res_hamilton.smoothed_marginal_probabilities[1]#.plot(title='Probability of being in the high regime', figsize=(12,3));
    rr = res_hamilton.smoothed_marginal_probabilities[1]#res_hamilton.filtered_marginal_probabilities[0]#res_hamilton.filtered_marginal_probabilities[0]#1
    plt.plot(rr)
    #e = [0,0,0]
    #e = np.stack(e)
    #e = pd.DataFrame(e)
    #r = r.reset_index(drop=True, inplace=True)
    #r = r.drop('Date')
    #r = pd.concat([e,r],0)
    norm = True
    sides = 'centerdc'

    #p.plot(label='MA (15, 30)', norm=norm, sides=sides)

    print("r shape")
    #print(r)
    #r = np.array(r)
    rr = rr.values
    
    #print(r)
    r = np.stack(rr)
    r = r.reshape(-1, 1)
    print(r)
    #r = Scalar.fit_transform(r)
    
    #r= np.expand_dims(r,1)
    #print(r.shape)
    r = pd.DataFrame(r)
    print(r.shape)
    print(dta_hamilton.shape)
    #r.index = pd.date_range('1958-06-30', '2020-09-30',freq="QS")
    
    print(datafour.shape)
    print(datafour)
    #datafour = pd.merge([datafour,r],right=datafour)
    #datafour = pd.concat([datafour,r],axis=1)
    #datafour = datafour.merge([r])
    #datafour = datafour.fillna(0)
    #r = r.drop([r.index ],axis=1)
    #datafour = pd.concat([datafour,r],1)
    #print("r")
    #print(r.shape)
    """
    mod_hamilton = sm.tsa.MarkovRegression(dta_hamilton,k_regimes=2)
    
    num_params = mod_hamilton.k_params
    params = mod_hamilton.start_params
    print(params.shape)
    print(params)
    #b = mod_hamilton.from_formula("1+B*(t-xdata)**(m) *(1+(B*((t-x)**m)*np.cos(T+w*np.log10(t-x)))*np.cos(w*np.log10(t - x )+T)) - y",dta_hamilton)
    a = mod_hamilton.regime_transition_matrix(params=params)
    #dta_hamilton.index = pd.DatetimeIndex(dta_hamilton.date, freq='QS')
    #from_formula
    #print("res_hamilton")
    #print(res_hamilton)
    print(a)
    
    fig, axes = plt.subplots(2, figsize=(7,7))
    ax = axes[0]
    ax.plot(res_hamilton.filtered_marginal_probabilities[0])
    print("probabilities")
    print(res_hamilton.filtered_marginal_probabilities[0])
    
    ax = axes[1]
    ax.plot(res_hamilton.smoothed_marginal_probabilities[0])
    print(res_hamilton.expected_durations)
    #predictions = mod_hamilton.predict(params,start='2020-06-30',end='2025-01-01')
    
    #r = res_hamilton.filtered_marginal_probabilities[0]
    
    r = r.values
    print("r")
    print(r)
    print(r.shape)
    #r = np.stack(r)
    #r = r.reshape(-1, 1)
    #r = Scalar.fit_transform(r)
    #r = np.array(r)
    
    r = pd.DataFrame(r)
    
    #r = pd.DataFrame(r)
    #r.drop([r.index ],axis=1)
    datafour = pd.DataFrame(datafour)
    #datafour = pd.merge([datafour,r],right=datafour)
    print("asdf")
    print(r.shape)
    """
    #plt.plot(r)
    #print(datafour.shape)
    #datafour = pd.concat([datafour,r],axis=1)
    #r = pd.DataFrame(r)
    print(datafour.shape)
    datafour = pd.concat([datafour, Debt_data_change_of_change,DESPAIR,dataSeven,ECOG_V,inflation,consumption_per_capita,d7], axis=1)#just added consumption_per_capita
    print(data_to_predict.shape)
    #data = pd.concat([datafour,trainingtarget],axis=1)
    print(datafour.shape)
    datafour = datafour.fillna(0)
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #datafour = imp.fit_transform(datafour)
    datafour = np.array(datafour)
    
    Data_TO_predict = np.array(data_to_predict)
    H, c, data = compute_Hc(data_to_predict, kind='change', simplified=True)
    print(H)
    x= pd.DataFrame(datafour)#np.concatenate((datafour, Data_TO_predict), axis=1)
    x = pd.concat([x,r],1)
    x= np.array(x)
    print("checkpoint 3")
    
    #z = pd.concat([datafour,data_to_predict])
    #print(z.ndim)
    #z = Scalar.fit(z)
    #z = np.array(z)
    #z = np.vsplit(z,7)
    
    #datafour = z[0]
    
    #trainingtarget = z[1]
    
    #z = pd.concat(datafour,trainingtarget)
    #z = Scalar.fit_transform(z)
    pca = PCA()
    #a = decompose(trainingtarget)#seasonal_decompose(trainingtarget,model='additive',freq=1)
    #a.plot()
    #print(trainingtarget)
    
    #plt.plot(trainingtarget)
    #trainingtarget = a.trend
    x = Scalar.fit_transform(x)
    #print(ECOG_V.shape)
    print(x)
    x = Signal1D(x)
    x.decomposition(output_dimension=19,algorithm="MLPCA")
    #x = pca.fit_transform(x)
    my_augmenter = (
     #TimeWarp() * 2  # random time warping 5 times in parallel
     #+ Crop(size=50)  # random crop subsequences with length 300
     Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
     #+Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
     +Reverse() @ 0.5  # with 50% probability, reverse the sequence 
     )
    
    #x = np.array_split(x,14,axis=1)
    #X,y = x[0],x[1]
    x = np.array(x)
    y = np.array(data_to_predict)

    #X = x.reshape()
    #Y = y.flatten()
    #print(X.shape)
    #print(Y.shape)
    x = my_augmenter.augment(x)
    
    #print("hdfa")
    #print(hfda.measure(y,5))    
    #datafour = normalizer.fit(datafour)
    y = Scalar.fit_transform(y)
    #fractional dimensions
    
    #yh = y.ravel()
    
    #poly = PolynomialFeatures(2)
    #datafour = poly.fit_transform(X)
    #y = trainingtarget
    #X = datafour
    #trainingtarget = pd.DataFrame(trainingtarget)
    #trainingtarget = pd.DataFrame(trainingtarget)
    #time_index = pd.date_range(start=1959.6, end=2018.8, periods=237)#np.linspace(1959.6,2018.8,237)#np.linspace(0, 10, N, endpoint=True)
    #trainingtarget = trainingtarget.set_index(time_index)
    
    #trainingtarget = statsmodels.tsa.seasonal.STL(trainingtarget,period=None)
    #Reminder: look into possibility that the DATES column is being included into datafour
    
    #inputOne = len(dataThree)
    #print(inputOne)
    #print(dataThree)
    #x_train = Scalar.fit_transform(datafour[:220:])
    #y_train = Scalar.fit_transform(trainingtarget[:220:])
    #x_train = pca.fit(x_train)
    #x_test = Scalar.transform(datafour[220::])
    
    print("shape of x")
    print(x.shape)
    
    #x = TSNE(n_components=19,method='exact').fit_transform(x)
    #x = umap.UMAP(metric='euclidean',n_components=2).fit_transform(x,y)
    #x = cp.array(x)
    y = np.array(y)
    #X = y
    y =my_augmenter.augment(y) 
    #a = np.random.rand(500)#nolds.load_qrandom()
    a = [42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    #a has a extremely small correlation dimension close to 0. a also has a negative lypanov exponent
    #c = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    b = [3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3,6,9,6,3]
    
    # b has a correlation dimension which is very close to 0 but a lypanov of -inf. This implies that b is super-stable and deterministic
    #for an attracot rto be strange it must have atleast one positive lyapunov exponent
    #one must have a positive lypunov in order to be a chaotic system
    #y has a correlation dimension of 1.69
    #attractor exiists if embedding_dimension >= correlation_dimension*2 +1
    #c = dta_hamilton.values
    dta_hamilton = quandl.get("MULTPL/SP500_REAL_PRICE_MONTH", authtoken="DNMZo2iRzVENxpxqHBKF", collapse="monthly", start_date="1958-06-30", end_date="2020-09-30")
    dd = np.ravel(dta_hamilton)#data_to_predict)
    series = [gauss(0.0, 1.0) for i in range(1000)]
    series = np.stack(series)
    print(nolds.corr_dim(dd,10))#y has correlation-dimension of 1.68 up to 3 embedding-dimensions
    print(nolds.lyap_e(dd,9,matrix_dim=3))
    print(nolds.lyap_r(dd,9))#y has a very small positive lypunov dimension at up to 3 embedding-dimensions
    #p = pma(dta_hamilton, 15, 30, NFFT=4096)
    p,f = signal.periodogram(dd)
    a = integrate.cumtrapz(f,np.sqrt(p),0)
    print("fasd")
    print(a)
    #plt.plot(a)
    #p,f = signal.welch(series)
    #plt.semilogy(np.sqrt(p),f)
    #spectral density distribution is most similar to the lorenz equations
    #plt.psd(series)
    """
    def my_emd(signal):
        emd = CEEMDAN(ensemble_size=50, random_seed=42, cores=4)
        emd(signal, progress=True)
        emd.postprocessing()
        return emd.c_modes
    hht = HHT(frequency, emd=my_emd, method="NHT", norm_type="lmd", smooth_width=5)
    spectrogram = hht(sun)
    spectrogram.sum("frequency").plot()
    """
    """
    interpretation of economic growth: based on the maximum lypunov exponent of -0.0013546540018613332the average behavior is that 
    it is a steady-state conservative system with extremely small dissaptive tendencies. 
    it has lypunov exponents of [ 0.05176133 -0.0347021  -0.20290983] which imply that although its average behavior is steady state
    it undergoes bifurcations and is similar to a lorentz attractor system albeit with a weak attractor. correlation dimension of 0.0011
    
    for the stock market maximum lypunov is 0.065. Implies that it is very close to a steady state with slightly chaotic tendencies. 
    [ 0.06782889 -0.02888527 -0.15514901] which means that this is also similar to a lorentz attractor system. 
    
    positive maximum lypunov corresponds to non-linear system with no chaos, attractors or self-similarity. 
    conservative which corresponds to 0 maximum lypunov allows chaos but no attractors 
    dissapative which corresponds to -maximum lypunov system allows for chaos and attractos including chaotic attractors 
    
    brownian motion is an example of a system with a positive maximum lypunov exponent 
    
    it is also worth noting that non conservative systems do not have a constant lypunov across entire system 
    Although the system is deterministic, there is no order to the orbit that ensues
    """
    #it is worth noting that lypunov of 0 is conservative system, negative lypunov is dissapative and positive lypunov is chaotic 
    # maximal lypunov above 0 and positive means is part of a chaotic system
    #note: lorentz attractor has correlation dimension of 2.055
    #x = np.array(datafour[:,None])
    """
    model = BoostBoruta(
        clf_lgbm, param_grid=param_dist_hyperopt, max_iter=200, perc=100,
        importance_type='shap_importances', train_importance=False
    )
    #TODO: add test train split from sklearn before uncommenting out bottom line
    #model.fit(x, y, eval_set=[(X_clf_valid, y_clf_valid)], early_stopping_rounds=6, verbose=0)
    """
    x = np.expand_dims(x,1)
    return x,y
x,y = data()
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.math import psd_kernels as tfk

#cycle, trend = sm.tsa.filters.hpfilter(y,1600)
#y = trend
#y = b
#Y = a.trend
#y = Y

num_inducing_points = 45
#custom time series kernel  integrate.quad(lambda x: special.jv(2.5,x), -π, π)e^(ih*lambda)*f(lambda)d(lambda)
#default RBF kernel k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
"""
  def kernel(self):
    return 1/2*np.absolute(t)**(2H)+np.absolute(s)**(2H) - (np.absolute(t)-np.absolute(s))**2H
    t = x
    s = y
    Calculating the average value, Xm, of the X1..Xn series
    Calculating the standard series deviation, S
    Normalization of the series by deducting the average value, Zr (where r=1..n), from each value
    Creating a cumulative time series Y1=Z1+Zr, where r=2..n
    Calculating the magnitude of the cumulative time series R=max(Y1..Yn)-min(Y1..Yn)
    Dividing the magnitude of the cumulative time series by the standard deviation (S).

"""

#Time series gaussian kernel with long term dependence: E[BSubSccript(H)(H)BsubscriptH(s)] = 1/2|t|^2H-|t-s|&2H where H is the hearst exponent If we assume there is a long term dependence that means 1 > H > 1/2

x = x.astype(np.float64)#tf.dtypes.cast(x, tf.int32) #
#x = tf.cast(x, tf.float32)
#x = tensor_util.convert_nonref_to_tensor(x, dtype=x.dtype)

class RBFKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)
    self.amplitude = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='amplitude')
    self.length_scale = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='length_scale')
    
  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    #print(dtype)
    return x

  @property
  def kernel(self):
   
    return tfk.ExponentiatedQuadratic(
      amplitude=self.amplitude,
      length_scale=self.length_scale)
    observation_noise_variance = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='observation_noise_variance')

dtype = np.float64
amplitude = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='amplitude')
length_scale = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='length_scale')
kernel = tfk.ExponentiatedQuadratic(
      amplitude=amplitude,
      length_scale=length_scale)
observation_noise_variance = tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='observation_noise_variance')

"""    
    tfp.math.psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.nn.softplus(0.1 * self._amplitude), length_scale=tf.nn.softplus(5. * self._length_scale), feature_ndims=1, validate_args=False,
    name='ExponentiatedQuadratic'
)
"""
#MaternOneHalf might be better then exponential quadratic RBF is too smooth

#x1 = x[0]
#x2 = x[1]

#return tf.convert_to_tensor(1/2*(np.absolute(x1))**(2*H)+(np.absolute(x2))**(2*H) - (np.absolute(x1)-np.absolute(x2))**2*H)#tf.as_dtype(1/2*(np.absolute(x))**(2*H)+(np.absolute(y))**(2*H) - (np.absolute(x)-np.absolute(y))**2*H)
x_tst = x[189::]
x_range = 237
num_distributions_over_Functions = 1
tf.keras.backend.set_floatx('float64')
#kernel = Brownian #tfp.positive_semidefinite_kernels.ExponentiatedQuadratic#MaternOneHalf()
kernels = tfp.math.psd_kernels.MaternThreeHalves(
    amplitude=tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='amplitude'), length_scale=tfp.util.TransformedVariable(
      1., tfb.Softplus(), dtype=dtype, name='length_scale'), feature_ndims=1, validate_args=False,
    name='MaternThreeHalves'
)
from tensorflow.keras.optimizers import Adam, Nadam
"""
import tensorflow as tf
import kerasncp as kncp
from tcn import TCN, tcn_full_summary
from kerasncp.tf import LTCCell
ncp_wiring = kncp.wirings.NCP(
    inter_neurons=20,  # Number of inter neurons
    command_neurons=10,  # Number of command neurons
    motor_neurons=50,  # Number of motor neurons
    sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
    inter_fanout=5,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=6,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=4,  # How many incoming synapses has each motor neuron
)
ncp_cell = LTCCell(
    ncp_wiring,
    initialization_ranges={
        # Overwrite some of the initialization ranges
        "w": (0.2, 2.0),
    },
)
y = y.astype(np.float64)
wiring = kncp.wirings.FullyConnected(8, 1)  # 8 units, 1 motor neuron
ltc_cell = LTCCell(wiring) # Create LTC model
"""
"""
tfp.sts.impute_missing_values(
    model, observed_time_series, parameter_samples, include_observation_noise=False
)

"""
"""
def objective(trial):

    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f'n_units_l{i}', 4, 128, log=True)
        model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
    model.add(tf.keras.layers.Dense(CLASSES))
    ...
    return accuracy

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


"""

def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,19), dtype=x.dtype),
        #tf.keras.layers.RNN(ltc_cell, return_sequences=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(25,kernel_initializer='ones',activation='tanh', dtype = x.dtype, use_bias=True)),
        #Attention(),
        #tf.keras.layers.InputLayer(input_shape=(10),dtype=x.dtype),#put a 1 before the 9 later
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(50,kernel_initializer='ones', use_bias=False),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(75,kernel_initializer='ones', use_bias=False),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(100,kernel_initializer='ones', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(125,kernel_initializer='ones', use_bias=False),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(150,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(175,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(200,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(225,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(250,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(225,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(200,kernel_initializer='ones',use_bias=False),
        #goal is to eventually replace the first dense layer with an LSTM layer
        #tf.keras.layers.LSTM
        #tf.keras.layers.TimeDistributed(Dense(vocabulary)))
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(150,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(125,kernel_initializer='ones', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(100,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(75,kernel_initializer='ones', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(50,kernel_initializer='ones',use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(25, activation='elu',kernel_initializer='zeros',use_bias=False,),
        tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points, kernel_provider=RBFKernelFn(dtype=x.dtype) , event_shape=(1,),
        inducing_index_points_initializer=tf.compat.v1.constant_initializer(
                np.linspace(0,x_range, num=1125,
                            dtype=x.dtype)[..., np.newaxis]), unconstrained_observation_noise_variance_initializer=(tf.compat.v1.constant_initializer(np.log(np.expm1(1.)).astype(x.dtype))),variational_inducing_observations_scale_initializer=(tf.compat.v1.constant_initializer(np.log(np.expm1(1.)).astype(x.dtype))), mean_fn=None,
        jitter=1e-06, convert_to_tensor_fn=tfp.distributions.Distribution.sample)
    
      
        #in unconstrained thing replace astype with tf.dtype thing.    #tf.initializers.constant(-10.0)
        ])
    batch_size =19
    #use a different loss other than the variational gaussian loss
    loss = lambda y, rv_y: rv_y.variational_loss(
        y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])
    #tf.keras.optimizers.Adam(1e-4) tf.optimizers.Adam(learning_rate=0.011)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.011), loss=loss)#tf.optimizers.Adam(learning_rate=0.01)
    return model
tuner = kt.Hyperband(
    hypermodel = build_model,
    objective='val_loss',
    max_epochs=10,
    hyperband_iterations=3,overwrite=True
)
tuner.search(x, y, epochs=5,verbose=True,validation_split=0.1,)
best_hps=tuner.get_best_hyperparameters(num_trials=10)#[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(x, y, epochs=5, validation_split=0.1)

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#print(best_hps)

print("spinning up core")
time_steps = 1
input_dim = 18
tcn_layer = TCN(input_shape=(time_steps, input_dim))
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,19), dtype=x.dtype),
    #tf.keras.layers.RNN(ltc_cell, return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(25,kernel_initializer='zeros',activation='tanh', dtype = x.dtype, use_bias=True)),
    #Attention(50)#note: only use attention when Gaussian process layer is not being used
    #tf.keras.layers.InputLayer(input_shape=(10),dtype=x.dtype),#put a 1 before the 9 later
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50,kernel_initializer='zeros', use_bias=False),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(75,kernel_initializer='zeros', use_bias=False),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100,kernel_initializer='zeros', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(125,kernel_initializer='zeros', use_bias=False),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(150,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(175,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(200,kernel_initializer='ones',use_bias=False),
    
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(225,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(250,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(225,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(200,kernel_initializer='ones',use_bias=False),
    #goal is to eventually replace the first dense layer with an LSTM layer
    #tf.keras.layers.LSTM
    #tf.keras.layers.TimeDistributed(Dense(vocabulary)))
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(150,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(125,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(100,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(75,kernel_initializer='ones', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(50,kernel_initializer='ones',use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(25, activation='elu',kernel_initializer='zeros',use_bias=False,),
    tfp.layers.VariationalGaussianProcess(
    num_inducing_points=num_inducing_points, kernel_provider=RBFKernelFn(dtype=x.dtype) , event_shape=(1,),
    inducing_index_points_initializer=tf.compat.v1.constant_initializer(
            np.linspace(0,x_range, num=1125,
                        dtype=x.dtype)[..., np.newaxis]), unconstrained_observation_noise_variance_initializer=(tf.compat.v1.constant_initializer(np.log(np.expm1(1.)).astype(x.dtype))),variational_inducing_observations_scale_initializer=(tf.compat.v1.constant_initializer(np.log(np.expm1(1.)).astype(x.dtype))), mean_fn=None,
    jitter=1e-06, convert_to_tensor_fn=tfp.distributions.Distribution.sample)

  
    #in unconstrained thing replace astype with tf.dtype thing.    #tf.initializers.constant(-10.0)
    ])
batch_size =19
#use a different loss other than the variational gaussian loss

loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])

#tf.keras.optimizers.Adam(1e-4) tf.optimizers.Adam(learning_rate=0.011)
model.compile(optimizer=tf.keras.optimizers.Adam(0.011), loss=loss)#tf.optimizers.Adam(learning_rate=0.01)

model.summary()
model.fit(x,y,epochs=55,verbose=True,validation_split=0.1)#5
"""
from sklearn.model_selection import train_test_split
X = np.squeeze(x,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#X_train.shape = X_train.shape[0]
#print(X_train.shape)
#explainer = shap.Explainer(model)

e = shap.GradientExplainer(model
)


shap_values =  e.shap_values(X_train)#e(X_test)
shap.plots.beeswarm(shap_values)
#shap_values,indexes = e.shap_values(model, ranked_outputs=2)
"""
yhat = model(x_tst)#Note that this is a distribution not a tensor
num_samples = 13 #note: num_samples refers to how many generated future evolutions of economic growth you want it to generate over the specified period of time
Models = []
#probability =
average = []
"""
note: the below code works for turning from a tensorflow distribution back to a tensor
a = tf.compat.v1.convert_to_tensor(
    yhat,
    dtype=None,
    name=None,
    preferred_dtype=None,
    dtype_hint=None
)
"""
#plt.plot(a)
#plt.plot(yhats)#Note: yhat is supposedly a distribution and not a tensor explore the possibility that it is outputting probabilities and not tensors
r = []
for i in range(num_samples):
  #plt.plot(yhat)
  sample_ = yhat.sample().numpy()
  #probability = yhat.prob(0.7).numpy()
  #Model = sample
  #sample_[..., 0].T,
  #print(model.summary)
  mean = yhat.mean().numpy()
  #print(mean)
  #variance = yhat.variance().numpy()
  #std = yhat.stddev().numpy()
  #print(yhat.sample().numpy)

  plt.plot((sample_[..., 0].T),
           'r',
           linewidth=0.2,
           label='ensemble means' if i == 0 else None);
  e = sample_[..., 0].T
  #print("asdf")
  #print(len(e))
  r.append(e)
  print(len(r[0]))

"""
(variational_loss,
variational_distributions) = tfp.sts.build_factored_variational_loss(
model=model, observed_time_series=observed_time_series,
init_batch_shape=[10])


"""
#a = yhat.sample_[..., 0].T

print("Predictions")
#print(predictions)
#plt.plot(predictions)

#forecast_dist = tfp.sts.forecast(model, observed_time_series,parameter_samples=sample_,num_steps_forecast=5)
print(sample_)
print(x.shape)
#r = pd.DataFrame(r)

#sns.kdeplot(sample_[..., 0].T)
from git import Repo
"""
repo = Repo(https://github.com/lordfiftysix/Special_projects.git)
repo.git.add(update=True)
repo.index.commit("")
origin = repo.remote(name='origin')
origin.push()

"""
rfinal = (r[0]+r[1]+r[2]+r[3]+r[4]+r[5]+r[6]+r[7]+r[8]+r[9]+r[10]+r[11]+r[12])/13
plt.plot(rfinal)
u1 = r[0]
u2 = r[1]
u3 = r[2]
u4 = r[3]
u5 = r[4]
u6 = r[5]
u7 = r[6]
u8 = r[7]
u9 = r[8]
u10=r[9]
u11=r[10]
u12 = r[11]
u13 = r[12]
ufinal = np.sqrt((((u1 - rfinal) + (u2-rfinal) + (u3-rfinal) + (u4-rfinal) + (u5 - rfinal) + (u6-rfinal)+(u7-rfinal)+(u7-rfinal)+(u8-rfinal)+(u9-rfinal)+(u10-rfinal)+(u11-rfinal)+(u12-rfinal)+(u13-rfinal))**2)/13)
#from shapely.geometry import LineString

#note: use shapely to fine intersection points
#first_line = LineString(np.column_stack((x, f)))
#second_line = LineString(np.column_stack((x, g)))
#intersection = first_line.intersection(second_line)

#this is for plotting the intersection points
#plt.plot(*LineString(intersection).xy, 'o')

#this is for getting the x and y of the intersection
#x, y = LineString(intersection).xy

print(ufinal)
print("ufinal")
print(len(ufinal))#note ufinal len is 48
print(len(rfinal))
plt.plot(y[189::])
print(ufinal.shape)
ufinal = np.expand_dims(ufinal,axis=1)
rfinal = np.expand_dims(rfinal,axis=1)
print(rfinal.shape)
ua = rfinal[...,0]+2*ufinal[...,0]
ub = rfinal[...,0]-2*ufinal[...,0]
q = np.linspace(0,55,55)
#for i in range(len(rfinal)):

#plt.fill_between(q,ua,ub,color='blue',alpha=0.2,)
plt.fill_between(q,rfinal[:,0],ua,color='blue',alpha=0.2,)
plt.fill_between(q,rfinal[:,0],ub,color='blue',alpha=0.2)
#plt.fill_between(rfinal[:,0],rfinal[:,0]+ufinal[:,0],rfinal[:,0]-ufinal[:,0],color='yellow')
print("average")
#st_dev = average.mean()#stddev()
#average = mean#.mean()#.numpy()

#plt.plot(sample_[20::])
print("samples")
#print(sample_.T[20::])
print(sample_)
print(sample_.shape)
error = (rfinal[...,0]-y[189::])/55

rmse = np.sum(tf.keras.losses.MSE(y[189::],
rfinal[...,0]
))/48

ys = pd.DataFrame(y)
ys = ys.head(189)
mean = np.mean(ys)
rmseb = np.sum(tf.keras.losses.MSE(y[189::],np.full(shape=48,fill_value=mean)))/48

print("rmse")
print(np.sqrt(rmseb))
print(np.sqrt(rmse))
print("JEPPS outperforms random guessing by "+ str(((rmseb/rmse)-1)*100) + " %")

#plt.plot(average[20::])
#print(len(average))#Note that supposedly the average error rate for the Fed's current forcasting model called gdpNow is 0.56%
#print(std)
#if this was a standard normal distribution than 95% would be 0.252 and 0.968

print("error")
#print(error)
print("average error")
print(np.sum(error)/len(error))

print(levene([rfinal[...,0],y[189::]],center='median') )
#c = np.corrcoef(rfinal, y[:-189:])[0, 1]
#print(c)
"""
plt.plot(X, Y, "kx", mew=2)
(line,) = plt.plot(xx, mean, lw=2)
_ = plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 2 * np.sqrt(var[:, 0]),
    mean[:, 0] + 2 * np.sqrt(var[:, 0]),
    color=line.get_color(),
    alpha=0.2,
)

name 	

"""