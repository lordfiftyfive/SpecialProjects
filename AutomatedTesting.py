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

from pyrelational.models import LightningModel
from pyrelational.data import GenericDataManager

from scipy.spatial import distance
from numba import jit
import urllib3
from PIL import Image
from io import BytesIO
import get_response
#import grequests
http = urllib3.PoolManager()
@jit
def one():
    print("starting")

    #if dtype(u) == np.int: 
    parameters = {
        "a": 's',
        "b": 'subarnos@'
    }#GET /api/v4/{accessKey}/carts/customer
    p = {
        'ExternalId':"25"
        }
    a = json.dumps(parameters)
    print(a)
    auth = OAuth1('de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2')
    
    #s = http.post('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Carts/GetCartCustomer',auth=auth,data=parameters)#.get('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Clubs/Changes',auth=auth)
    #print(s.status_code)
    #rr = requests.head('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Carts/GetCartCustomer')
    #print(rr.headers)
    #rrr = requests.options('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Carts/GetCartCustomer')
    #print(rrr.content)
    #t = requests.get('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Clubs/Changes',auth=auth)#,data=parameters) 
    #r = requests.get('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Carts/GetCartCustomer',auth=auth,data=parameters)#auth=('user', 'pass'))
    #s = requests.post('http://GET/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/Coupons/Changes',auth=auth,data=parameters)
    r = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/ResolvedCustomers?quickSearch=subarno@bloyal.com',auth=('user', 'pass'),data=json.dumps(p))#,data=parameters)
    #u = requests.get('https://loyaltyengineus1.bloyal.com/api/v4/de4cf57cfc45184510c627de9d01324ad13d95ddafc751024597bffa74af43ae4f793bdc6a8c5eb1c6364eb2/CustomerReferrals/',auth=('user', 'pass'))
    #parser = get_response.get_json(r.text)
    #r.status_code()
    #print(r.status_code)
    u =r
    print("a")
    print(u.content)
    print("b")
    print(u.json())
    #t.status_code
    #print("c")
    #print(r.history)
    #print("d")
    #print(r.content())#
    #i = BytesIO(r.content)
    
    """
    we should try customers/commands/saves api where we save data and get the data back 
    
    we need to verify that all the parameters that are returned are withen expectations with our ai algorithm
    and not just the status parameter 
    
    
    """
    
    #print(i)
    #print(t.json())
    #print(r.encoding)
    #print(r.text())
    rr = requests.head('https://loyaltyengine.bloyal.io/swagger/ui/index#!/Carts/GetCartCustomer')
#r.text

one()

@jit
def two():
    
    # get models
    gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    query_txt = "this morning I went to "#"00000000-0000-0000-0000-000000000000 "
    
    query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")
    # get model response
    
    """
    Current plan of action
    
    stage 1 query -> trl -> pyrelational 
    (which will do either evaluation or regression) -> api
    
    """
    
    response_tensor  = respond_to_batch(gpt2_model, query_tensor)
    response_txt = gpt2_tokenizer.decode(response_tensor[0,:])
    ep = 5
    #reward is going to be -cosine similarity 
    reward = -distance.cosine(query_tensor,response_tensor)# *0.5*correctness#-distance.euclidean(query_tensor,response_tensor)
    #reward = -4*reward**2 + 8*reward
    reward = [torch.tensor(reward)]
    print(reward)
    
    # initialize trainer
    ppo_config = {'batch_size': 1, 'forward_batch_size': 1,'ppo_epochs':20}
    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **ppo_config)
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
    response_tensor  = respond_to_batch(gpt2_model, query_tensor)
    response_tx = gpt2_tokenizer.decode(response_tensor[0,:])
    print(train_stats)
    print("as")
    
    print(response_tx)
    print("fa")
    print(response_txt)
    
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
    
    model = LightningModel(gpt2_model,model_config={}, trainer_config={"epochs": 4})# this line compiles
    strategy = RelativeDistanceStrategy(data_manager=data_manager, model=model)

"""
from pymdp.agent import Agent
from pymdp import utils, maths

def active_inference_with_planning(A, B, C, D, n_actions, env, policy_len = 2, T = 5):

   #Initialize prior, first observation, and policies 
  
  prior = D # initial prior should be the D vector

  obs = env.reset() # get the initial observation

  policies = construct_policies([n_states], [n_actions], policy_len = policy_len)

  for t in range(T):

    print(f'Time {t}: Agent observes itself in location: {obs}')

    # convert the observation into the agent's observational state space (in terms of 0 through 8)
    obs_idx = grid_locations.index(obs)

    # perform inference over hidden states
    qs_current = infer_states(obs_idx, A, prior)
    plot_beliefs(qs_current, title_str = f"Beliefs about location at time {t}")

    # calculate expected free energy of actions
    G = calculate_G_policies(A, B, C, qs_current, policies)

    # to get action posterior, we marginalize P(u|pi) with the probabilities of each policy Q(pi), given by \sigma(-G)
    Q_pi = softmax(-G)

    # compute the probability of each action
    P_u = compute_prob_actions(actions, policies, Q_pi)

    # sample action from probability distribution over actions
    chosen_action = utils.sample(P_u)

    # compute prior for next timestep of inference
    prior = B[:,:,chosen_action].dot(qs_current) 

    # step the generative process and get new observation
    action_label = actions[chosen_action]
    obs = env.step(action_label)
  
  return qs_current
"""