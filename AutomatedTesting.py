# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:18:26 2022

@author: subar
"""

import requests #for making rest api requests
import json
from requests_oauthlib import OAuth1
#RL
import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
#active inference with relative distance strategy being the way we will implement diversity sampling 

from pyrelational.strategies.task_agnostic.relative_distance_strategy import (
    RelativeDistanceStrategy,
)
from transformers import GPT2Tokenizer
from pyrelational.models import LightningModel
from pyrelational.data import GenericDataManager
from pytorch_widedeep.self_supervised_training import (
    ContrastiveDenoisingTrainer,
)
#asdfasdfasdfa
from pytorch_widedeep.models import TabFastFormer, WideDeep, FTTransformer
from pytorch_widedeep.preprocessing import TabPreprocessor, TextPreprocessor
from pytorch_widedeep.datasets import load_adult
from scipy.spatial import distance
from numba import jit
import urllib3
from PIL import Image
from io import BytesIO
import get_response
import numpy as np
from numba import cuda
from hummingbird.ml import convert, load #we will be using this for conversion of our clustering algo
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from transformers import AutoTokenizer
import lox

import taichi as ti
import taichi.math as tm
"""
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = English()
tokenizer = Tokenizer(nlp.vocab)
"""
#import grequests
http = urllib3.PoolManager()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')#.to(device)
ti.init(arch=ti.gpu)
num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)
print(torch.cuda.get_device_name(device=device))
ppo_config = {'batch_size': 1, 'forward_batch_size': 1,"ppo_epochs": 4,"lr": 1.41e-5,"horizon":10000}
"""
ppo_config = {

    "steps": 51200,
    #"batch_size": 256,
    #"forward_batch_size": 16,
    "ppo_epochs": 4,   
    "txt_in_len": 5,
    "txt_out_len": 20,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
    "seed": 1,
}

df = load_adult(as_frame=True)
print(df)
target_col = "income"
target = df[target_col].values
print(target)
"""
gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')#.to(device)
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')#.to(device)
auth = OAuth1('de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2')
    
#gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#_ = gpt2_model.to(device)
#_ = sentiment_model.to(device)
#_ = gpt2_model_ref.to(device)
#_ = gpt2_tokenizer.to(device)
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **ppo_config)
print("checkpoint 1")
#@jit(target_backend='cuda')
#@lox.thread(14)

#@ti.func
def one():
    print("starting")
    #j
    #if dtype(u) == np.int: 
    """
    parameters = {
        "a": 's',
        "b": 'subarnos@'
    }#GET /api/v4/{accessKey}/carts/customer
    p = {
        'ExternalId':"25"
        }
    a = json.dumps(parameters)
    #print(a)
    """
    auth = OAuth1('de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2')
    
    #t = requests.get('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Clubs/Changes',auth=auth)#,data=parameters) 
    #r = requests.get('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Carts/GetCartCustomer',auth=auth,data=parameters)#auth=('user', 'pass'))
    #s = requests.post('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/Coupons/Changes',auth=auth)#,data=parameters)
    #r = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=('user', 'pass'),data=json.dumps(p))
    #,data=parameters)
    #u = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/CustomerReferrals/',auth=('user', 'pass'))
    #n = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/carts/coupons',auth=auth)
    #parser = get_response.get_json(r.text)
    #r.status_code()
    #print(r.status_code)
    #u =n
    #a = np.array(dic['data']).ravel()
    #print(a)
    #p = dict(zip(a))
    
    a = {'customerToken': 1,'searchCustomer.externalId': 1}

    b = ['customerToken','searchCustomer.externalId','searchCustomer.firstName2','searchCustomer.lastName2','searchCustomer.companyName','earchCustomer.alertCount','searchCustomer.referralCustomer1Uid']
    c = ['searchCustomer.firstName2','searchCustomer.lastName2','searchCustomer.companyName','earchCustomer.alertCount']
    r = requests.get('https://loyaltyengine1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/resolvedcustomers?',params=a,auth=auth,)
    d = {"CouponCodes": "a"}
    a=r.json()
    #b = r.params()
    print(r.content)
    print(a)
    l = requests.post('https://loyaltyengine1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/carts/commands/calculates',params=d,auth=auth)
    p = []#dic['data']
    l1 = []
    success = 0
    data = []
    p = []#dic['data']
    l1 = []
    success = 0
    data = []
    y1 = []
    y2=[]
    d2 = []
    d3 = []
    #ti.loop_config(parallelize=8)#, block_dim=16)
    for i in range(len(c)):
        #while success != 'success':
        for i in range(5):
                
            query_txt = "aaaaaaaaaaaaaaaaaaa"
            query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")
            #print("checkpoint 2")
            response_tensor  = respond_to_batch(gpt2_model, query_tensor)
            response_txt = gpt2_tokenizer.decode(response_tensor[0,:])
            p.append(response_txt)
            #l1.append(u)
    
            #p.update(u=response_txt)
            print(p)
            print("a")
            #L1 = dic['data']#np.array(dic['data']).ravel()#['a','b','c','d']
            #L1 = np.array(L1)
            #L2 = [1,2,3,4]
            #d = dict(zip(l1,L2))
            ff = dict(zip(c,p))
            #r = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(d))
            #f =  requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(ff))#('user', 'pass')
            
            #r = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(p))
            #f = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(ff))
            
            #s = requests.post('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/Coupons/Changes',auth=auth,data=json.dumps(ff))
            #v = r.json()
            #l = requests.post('https://loyaltyengine1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/carts/commands/calculates',params=d,auth=auth)
            s = requests.get('https://loyaltyengine1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/resolvedcustomers?',params=ff,auth=auth)
            #ss = requests.get('https://loyaltyengine1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/resolvedcustomers?',params=ff,auth=auth)
            d2.append(response_tensor)
            f = s.json()
            print("ff")
            print(ff)
            print(l1)
            print(f)
            #fail = f['status']
            print("fai")
            #print(fail)
            success = f['status']
            print(success)
            print("d")
            y1.append(success)
            y2.append(f)
            reward1 = 0
            dat = s.text
            d3.append(dat)
            if success == 'success':#'error':#
                reward1=500
            else:
                reward1=-500
            reward1 = [torch.tensor(reward1)]

            #reward1 = rewardOne()
            print(reward1)
            
            train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward1)
            data.append(response_txt)
    return data, d2, y1,y2,d3
            
    """
    we will probably need to wrap this entire function into a while loop
    
    the while loop will terminate when a certain minimum number of sucessful test cases have been
    generated 
    
    we need to include exploration in the form of entropy in our reward function
    """
    
    """
    we should try customers/commands/saves api where we save data and get the data back 
    
    we need to verify that all the parameters that are returned are withen expectations with our ai algorithm
    and not just the status parameter 
    
    pseudo code
    
    for i in parameters:
        gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
        gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        _ = gpt2_model.to(device)
        #_ = sentiment_model.to(device)
        _ = gpt2_model_ref.to(device)
        query_txt = "this morning I went to "#"00000000-0000-0000-0000-000000000000 "
        
        query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")
    #3 rewards
    
    1st reward = sucess
    
    2nd reward = fail + maxent term
    
    3rd reward = fail + category from SSL algorithm =1 
    
    we can also take the closest and furthest points in our generated data and compare the similarity distance between them. 
    
    We can then say the following: given for instance that the euclidian distance between these two points is 400 we can say that we need to sample 130 points i.e 
    do 130 iterations in order to  conclude that the chances of missing a test case are 2.5% given that a 1% or less from our sample will be treated as irrelavent
    . so max iterations for adversarial stage with 3rd reward should be no more then then 130 for given each parameter. so for 130 parameters no more then 15600 iterations
    130 iterations 
    
    note: we should round down to the nearest thousand if it is above a thousand and round to the nearest 100 or 10 if below a thousand because we are assuming our scheme 
    is more effecient then a random sampler. 
    
    report as of 1/13/2022
    
    we cannot get negative examples for even ssl. Therefore we are going to use unsupervised learning
    Specifically we are going to use stumpy to cluster the inputs and outputs from the API request
    that we know works. We will then have our TRL algorithm generate test cases for an API request
    that we dont know works and use stumpys anomoly detection to determine whether any output deviates
    If none of the output for the new api deviate we will assume the New api is correct. 
    
    we can also use scikit learn with hummingbird-ml for the clustering and the outlier detection
    
    we may use stumpy later but scikitlearn with hummingbirdd-ml should be easier to implement
    for both clustering and outlier detection
    
    we dont even need to do the clustering step. Just fit on perfectly working data with isoforest
    and predict on data that comes from API that we have not verified works 
    """
    #print(r.encoding)
    #print(r.text())
    #print("c")
    #return data,y1,y2
    #rr = requests.head('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Carts/GetCartCustomer',auth=('user', 'pass')) #this line is causing problems
    
#r.text
#@jit(target_backend='cuda')
#@ti.kernel
def two():
    
    # get models

    #gpt2_model.to(device);
    #gpt2_model_ref.to(device);
    #_ = gpt2_model.to(device)
    #_ = sentiment_model.to(device)
    #_ = gpt2_model_ref.to(device)
    #query_txt = "this morning I went to "#"00000000-0000-0000-0000-000000000000 "
    
    #query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

    # get model response
    
    """
    Current plan of action
    
    stage 1 query -> trl -> pyrelational 
    (which will do either evaluation or regression) -> api
    
    note: we just need to tack on a entropy term to the reward function to maximize
    exploration
    
    the reward will have  two terms: one to denote sucess or failiure and another
    to entropy term to maximize exploration
    
    """
    
    #data = []
    """
    def pos_logit_to_reward(logit, task):
        
        #Take the positive sentiment logit and scale it for the task.
            #task [negative]: reward = -logit
            #task [neutral]: reward = -2*abs(logit)+4
            #task [positive]: reward = logit

        for i in range(len(logit)):
            if task[i]=='[negative]':
                logit[i] = -logit[i]
            elif task[i]=='[neutral]':
                logit[i] = -2*torch.abs(logit[i])+4
            elif task[i]=='[positive]':
                pass
            else:
                raise ValueError('task has to be in [0, 1, 2]!')
        return logit
    response_tensor  = respond_to_batch(gpt2_model, query_tensor)
    response_txt = gpt2_tokenizer.decode(response_tensor[0,:])
    ep = 5
    #reward is going to be -cosine similarity 
    reward = -distance.cosine(query_tensor,response_tensor)# *0.5*correctness#-distance.euclidean(query_tensor,response_tensor)
    #reward = -4*reward**2 + 8*reward
    reward = [torch.tensor(reward)] 
    """
    
    """
    
    + we may also want to consider adding a max ent regularizer that rewards exploration. 
    
    we may want to consider doing two stages of training with trl before this: one where trl tries generating correct sequences with its own reward and one with a reward
    that incentivizes wrong sequences with maximum exploration
    
    print(reward)
    
    # initialize trainer
    ppo_config = {'batch_size': 2, 'forward_batch_size': 1,'ppo_epochs':20}
    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **ppo_config)
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
    response_tensor  = respond_to_batch(gpt2_model, query_tensor)
    response_tx = gpt2_tokenizer.decode(response_tensor[0,:])
    
    """
#

#@ti.kernel
#@lox.thread(14)
@ti.data_oriented
def three():
    """
    tab_preprocessor = TabPreprocessor(
    cat_embed_cols=cat_embed_cols,
    continuous_cols=continuous_cols,
    with_attention=True,
    with_cls_token=True,  # this is optional
    )
    X_tab = tab_preprocessor.fit_transform(df_tr)
    """
    
    
    #y1 is whether there was a sucess or failiure y2 is the paramter returned, d2 is
    
    #clf = LocalOutlierFactor(n_neighbors=len(y1),novelty=True)
    #clf.fit(y1)
    #clf2 = LocalOutlierFactor(n_neighbors=len(y1),novelty=True)
    #test = np.zeros(20).reshape(-1, 1)
    #a = clf.predict(test)
    #Beginning of final component of final stage
    data,d2,y1,y2,d3 = one()
    c = ['searchCustomer.firstName2','searchCustomer.lastName2','searchCustomer.companyName','earchCustomer.alertCount']
    pp = pd.DataFrame(d3,columns=['text_column'])
    data = gpt2_tokenizer.encode(pp, return_tensors="pt")
    #gpt2_tokenizer.encode(query_txt, return_tensors="pt")
    
    #pp = pd.DataFrame(d3,columns=['text_column'])
    
    #text_preprocessor = TextPreprocessor(text_col='text_column')
    #data = text_preprocessor.fit_transform(pp)
    print("checkpoint 2")
    clf = LocalOutlierFactor(n_neighbors=len(data),novelty=True)
    clf.fit(data)
    test = []
    p=[]
    
    ti.loop_config(block_dim=16,parallelize=8)
    
    #val = ti.field(ti.i32, shape=len(c))
    #vall = ti.field(ti.i32, shape=20)
   
    for i in range(len(c)):
        #while success != 'success':
        for u in range(20):
            print("f")
            query_txt = "aaaaaaaaaaaaaaaaaaa"
            query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")
            #print("checkpoint 2")
            response_tensor  = respond_to_batch(gpt2_model, query_tensor)
            response_txt = gpt2_tokenizer.decode(response_tensor[0,:])
            p.append(response_txt)
            #l1.append(u)
    
            #p.update(u=response_txt)
            print(p)
            print("a")
            #L1 = dic['data']#np.array(dic['data']).ravel()#['a','b','c','d']
            #L1 = np.array(L1)
            #L2 = [1,2,3,4]
            #d = dict(zip(l1,L2))
            ff = dict(zip([c[i]],[p]))
            #r = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(d))
            #f =  requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(ff))#('user', 'pass')
            
            #r = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(p))
            #f = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=auth,data=json.dumps(ff))
            
            #s = requests.post('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/Coupons/Changes',auth=auth,data=json.dumps(ff))
            #v = r.json()
            #s = requests.get('https://loyaltyengine1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/resolvedcustomers?',params=ff,auth=auth)
            s = requests.get('https://loyaltyengine1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/resolvedcustomers?',params=ff,auth=auth)
            f = s.json()
            text = s.text
            text = pd.DataFrame(text[i],columns =['text_column'])
            text_preprocessor = TextPreprocessor(text_col='text_column')
            data = text_preprocessor.fit_transform(text)
            print("ff")
            print(ff)
            print(f)
            #fail = f['status']
            print("fai")
            #print(fail)
            success = f['status']
            print(success)
            print("d")#filelist = glob.glob(path + "/*.csv")
            print("data")
            reward1 = 0
            #tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            #text_preprocessor = TextPreprocessor(text_col='text_column')
            #data = tokenizer.fit_transform(f[i])

            print(data)
            #we only want to select the y[ith] parameter on each epoch
            a = clf.predict(data)
            if success == 'success' and a==1: #'error':#
                reward1=-500
            else:
                reward1=500
            reward1 = [torch.tensor(reward1)]

            #reward1 = rewardOne()
            print(reward1)
            
            train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward1)  
    
    """

    #ft_transformer = FTTransformer()
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=tab_preprocessor.continuous_cols,
    input_dim=32,
    kv_compression_factor=0.5,
    n_blocks=3,
    n_heads=4,)
    
    contrastive_denoising_trainer = ContrastiveDenoisingTrainer(
    model=ft_transformer,
    preprocessor=tab_preprocessor,
    )
    contrastive_denoising_trainer.pretrain(X_tab, n_epochs=5, batch_size=256)
    """
    
    """
    our two options are fastformer or SAINT. SAINT is the best for performance and fastformer is the most effecient 
    
    we will start with SAINTcfip
    
    """
    
"""
def three():
    dataset = 0
    gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [400, 22, 20])
    train_indices = train_ds.indices
    val_indices = val_ds.indices
    test_indices = test_ds.indices
    
    data_manager = GenericDataManager(
        dataset=dataset,
        train_indices=train_indices,
        validation_indices=val_indices,
        test_indices=test_indices,
        hit_ratio_at=5,
    )
    
    model = LightningModel(gpt2_model,model_config={}, trainer_config={"epochs": 4,"gpus":0})# this line compiles
    strategy = RelativeDistanceStrategy(data_manager=data_manager, model=model)
    
    # New data to be annotated, followed by an update of the data_manager and model
    to_annotate = strategy.active_learning_step(num_annotate=100)
    strategy.active_learning_update(indices=to_annotate, update_tag="Manual Update")
    
    # Annotating data step by step until the trainset is fully annotated
    strategy.full_active_learning_run(num_annotate=100)
    print(strategy)
"""
three()
#one()