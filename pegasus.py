# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 03:10:56 2022

@author: subar
"""
import gym
from gym.spaces import Discrete, MultiDiscrete,Box, Dict, MultiBinary
#utilities 
import numpy as np
import random
#these libraries have to do with the agents 
import ray
"""
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.qmix import QMixConfig
from ray import air, tune
import argparse
from gym.spaces import Dict, Discrete, Tuple, MultiDiscrete
import logging
import os
#import pathpy as pp

from ray.tune import register_env
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
"""
import mne# preprocessing and brain importation and utilities library including acessing and preprocessing the EEG data
#these libraries have to do with the free energy principle
import pymdp
from pymdp import utils
from pymdp.agent import Agent
#from gym.spaces import 

#from ray.rllib.algorithms.qmix.qmix_policy import QMixTorchPolicy

#optimization of deep learning and RL aspects of algorithm these will allow the algorithm to run faster with less memory 
#from composer import Trainer
#from nebullvm.api.functions import optimize_model
from numba import jit
"""
dependency network

Qmix.py - has qmixpolicy.py as a dependency 
Qmixpolicy.py has  mixers.py and Model.py dependencies
Model.py -base
mixers.py -base

"""

import tensorflow as tf
import ivy# library for interoperable across all deep learning frameworks 
#import tensorflow_datasets as tfds
import tensorflow_probability as tfp
#from nebulgym.decorators.torch_decorators import accelerate_model, accelerate_dataset


#below libraries are core libraries for q-mix Rllib algorithm
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple

import ray

import argparse
from math import ceil

import gym
import torch
import dyn_rl_benchmarks
import matplotlib.pyplot as plt
from graph_rl.graphs import HiTSGraph
from graph_rl.models import SACModel
from graph_rl.subtasks import DictInfoHidingTGSubtaskSpec, EnvSPSubtaskSpec
from graph_rl import Session

#device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#b = mne.io.read_raw_bdf("sub-hc1 ses-hc eeg sub-hc1_ses-hc_task-rest_eeg.bdf")
"""
import pathpy as pp
t = pp.TemporalNetwork()
t.add_edge('a', 'b', 1)
t.add_edge('b', 'a', 3)
t.add_edge('b', 'c', 3)
t.add_edge('d', 'c', 4)
t.add_edge('c', 'd', 5)
t.add_edge('c', 'b', 6)
print(t)
t

style = {    
  'ts_per_frame': 1, 
  'ms_per_frame': 200,
  #'look_ahead': 2, 
  #'look_behind': 2, 
  'node_size': 15, 
  #'inactive_edge_width': 2,
  #'active_edge_width': 4, 
  'label_color' : '#ffffff',
  'label_size' : '24px',
  'label_offset': [0,5]
  }
pp.visualisation.plot(t, **style)
"""
@jit
def a():
    print("initializing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAC on Obstacle-v1.")
    parser.add_argument(
        "--max_n_timed_subgoals", type=int, default=5, help="Timed subgoal budget."
    )
    parser.add_argument(
        "--render", default=False, action="store_true", help="Render while training."
    )
    parser.add_argument(
        "--hidden_layers", default=2, type=int, help="Number of hidden layers in MLPs."
    )
    parser.add_argument("--learning_rate", default=1.0e-4, type=float)
    args = parser.parse_args()

    # create environment
    env = gym.make("Drawbridge-v1")
    #env.render()  
    ###########################################################################
    # define subtask specifications (i.e. specify objective in each node/layer)
    ###########################################################################

    # Layer 0 (lower layer):
    # Objective is to reach a timed goal (TG).
    # The observation space of the environment is a dict space so keys can be 
    # used to select what the layer is to see and what the subgoal space should 
    # contain.
    partial_obs_keys = {"ship_pos", "ship_vel", "sails_unfurled", "bridge_phase"}
    goal_keys = {"ship_pos", "ship_vel", "sails_unfurled"}
    goal_achievement_threshold = {
        "ship_pos": 0.05,
        "ship_vel": 0.2,
        "sails_unfurled": 0.1
    }
    s_spec_0 = DictInfoHidingTGSubtaskSpec(
        goal_achievement_threshold=goal_achievement_threshold, 
        partial_obs_keys=partial_obs_keys, 
        goal_keys=goal_keys, 
        env=env, 
        delta_t_max=env.max_episode_length/args.max_n_timed_subgoals, 
    )

    # Layer 1 (higher layer):
    # Higher layer pursues a conventional environment goal.
    s_spec_1 = EnvSPSubtaskSpec(
        max_n_actions=args.max_n_timed_subgoals,
        env=env,
        map_to_env_goal=lambda partial_obs: partial_obs["ship_pos"]
    )
    subtask_specs = [s_spec_0, s_spec_1]

    ################################################################
    # specify model (i.e. actor and critic) and HiTS hyperparameters
    ################################################################

    algo_kwargs = []
    # entropy coefficient of SAC
    alphas = [0.01, 0.02]
    buffer_sizes = [400000, 10000]

    for i in range(2):
        # Flat algo refers to the off-policy RL algorithm used in 
        # individual layers. Can choose between SAC and DDPG.
        flat_algo_kwargs = {"alpha": alphas[i], "tau": 0.3}
        # NOTE: Input and output sizes are determined automatically
        model = SACModel(
            hidden_layers_actor=[16]*args.hidden_layers,
            hidden_layers_critics=[16]*args.hidden_layers,
            activation_fns_actor=[torch.nn.ReLU(inplace=False)]*args.hidden_layers,
            activation_fns_critics=[torch.nn.ReLU(inplace=False)]*args.hidden_layers,
            learning_rate_actor=args.learning_rate,
            learning_rate_critics=args.learning_rate,
            device='cpu',
            # q function of higher level should be restricted to < 0 because
            # of shortest path objective (reward <= 0)
            force_negative=True if i == 1 else False 
        )#.to(args.device)
        #model.to(device)
        algo_kwargs.append(
        {
            "model": model, 
            "flat_algo_name": "SAC",
            "flat_algo_kwargs": flat_algo_kwargs,
            "buffer_size": buffer_sizes[i],
            "batch_size": 256
        })


    ##############################
    # create graph via Graph class
    ##############################

    graph = HiTSGraph(
        name="HiTS_graph", 
        n_layers=2,
        env=env,
        subtask_specs=subtask_specs,
        HAC_kwargs=algo_kwargs[-1], # algorithm parameters of highest layer
        HiTS_kwargs=algo_kwargs[:-1], # algorithm parameters of all other layers
        update_tsgs_rendering=env.update_timed_subgoals,
        # highest layer sees cumulative env reward plus penalty for emitting 
        # timed subgoals
        env_reward_weight=0.01
    )

    ###################################
    # create and run session (training)
    ###################################

    sess = Session(graph, env)
    sess.run(
        n_steps=5000,#50000,#0,
        learn=True,
        render=args.render,
        success_reward=0.0
    )
    #print(graph)
    #plt.plot(graph)
    #graph.render()
    #env.render()
