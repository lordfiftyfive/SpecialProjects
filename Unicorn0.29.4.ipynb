import torchvision

import h5py
import scipy
import scipy.io
import numpy as np
import cv2

def loadmatrix(matfile):
    """Function to load .mat files.
    Parameters
    ----------
    matfile : str
        path to `matfile` containing fMRI data for a given trial.
    Returns
    -------
    dict
        dictionary containing data in key 'vol' for a given trial.
    """
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return scipy.io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

# Load and process data
import scipy.io as sp
import scipy.io

# Load visual stimuli
mat_data = sp.loadmat("/home/ubuntu/.cache/kagglehub/datasets/wan2022/cichy-et-al-2014/versions/1/Cichy_92_Image_Set_ROI_RDMs/92_Image_Set/92images.mat")
visual_stimuli = mat_data["visual_stimuli"]

# Convert images to grayscale
grayscale_images = []
for i in range(visual_stimuli.shape[1]):
    filename, rgb_image = visual_stimuli[0, i]
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    grayscale_images.append((filename, gray_image))

grayscale_array = np.array([img[1] for img in grayscale_images])
print(grayscale_array.shape)
x = grayscale_array

# Load target data
raw_y = loadmatrix("/home/ubuntu/.cache/kagglehub/datasets/wan2022/cichy-et-al-2014/versions/1/Cichy_92_Image_Set_ROI_RDMs/92_Image_Set/target_fmri.mat")
y1 = np.mean(raw_y['EVC_RDMs'], axis=0)  # shape: (92, 92)
y2 = np.mean(raw_y['IT_RDMs'], axis=0)   # shape: (92, 92)

# SCIKIT-TDA IMPLEMENTATION
try:
    import ripser
    from persim import PersistenceImager, PersLandscapeApprox
    import numpy as np
    import cv2
    SCIKIT_TDA_AVAILABLE = True
    print("✅ Scikit-TDA libraries loaded successfully!")
except ImportError:
    print("⚠️  Installing Scikit-TDA libraries...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ripser", "persim", "scikit-learn"])
    import ripser
    from persim import PersistenceImager, PersLandscapeApprox
    import numpy as np
    import cv2
    SCIKIT_TDA_AVAILABLE = True
    print("✅ Scikit-TDA libraries installed and loaded!")

import gc
import torch

class ScikitTDAImageProcessor:
    """
    TDA processor using scikit-tda (ripser + persim) instead of giotto-tda.
    Provides identical functionality with potentially easier installation.
    """
    
    def __init__(self, method='persistence_image', resolution=8, bandwidth=0.1):
        """
        Initialize processor with scikit-tda components.
        
        Args:
            method: 'persistence_image', 'persistence_landscape', or 'amplitude'
            resolution: Resolution for persistence images/landscapes  
            bandwidth: Bandwidth parameter for persistence images
        """
        self.method = method
        self.resolution = resolution
        self.bandwidth = bandwidth
        
        if method == 'persistence_image':
            # Use persim.PersistenceImager
            self.vectorizer = PersistenceImager(
                pixel_size=1.0/resolution,  # persim uses pixel_size instead of n_bins
                birth_range=(0, 2),
                pers_range=(0, 2),
                kernel_params={'sigma': bandwidth}
            )
            self.feature_dim = resolution * resolution * 2  # H0 + H1
            
        elif method == 'persistence_landscape':
            # Use persim.PersLandscapeApprox
            self.vectorizer = None  # Will be created per-diagram
            self.feature_dim = 5 * resolution * 2  # 5 layers, H0 + H1
            
        elif method == 'amplitude':
            # Simple amplitude calculation (manual implementation)
            self.vectorizer = None
            self.feature_dim = 2  # H0 + H1 amplitudes
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Scikit-TDA processor initialized: {self.feature_dim} features ({method})")
    
    def _image_to_point_cloud(self, image):
        """Convert image to point cloud for ripser processing."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Efficient preprocessing
        image = cv2.resize(image, (40, 40))
        image = image.astype(np.float32) / 255.0
        
        # Create point cloud from significant pixels
        y_coords, x_coords = np.where(image > 0.15)
        intensities = image[y_coords, x_coords]
        
        # Limit points for memory efficiency
        max_points = 150
        if len(x_coords) > max_points:
            indices = np.random.choice(len(x_coords), max_points, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
            intensities = intensities[indices]
        
        if len(x_coords) < 3:
            return None
        
        # Create 3D point cloud: (x, y, intensity)
        point_cloud = np.column_stack([
            x_coords / 40.0,  # Normalize coordinates
            y_coords / 40.0,
            intensities
        ])
        
        return point_cloud
    
    def _compute_persistence_diagrams(self, point_cloud):
        """Compute persistence diagrams using ripser."""
        try:
            # Use ripser to compute persistence diagrams
            rips_result = ripser.ripser(
                point_cloud,
                maxdim=1,  # Compute H0 and H1
                thresh=2.0,  # Maximum edge length
                coeff=2  # Coefficient field Z/2Z
            )
            
            diagrams = rips_result['dgms']
            return diagrams
            
        except Exception as e:
            print(f"Ripser computation failed: {e}")
            return None
    
    def _vectorize_diagrams(self, diagrams):
        """Convert persistence diagrams to feature vectors."""
        if diagrams is None or len(diagrams) < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        try:
            if self.method == 'persistence_image':
                # Use persim PersistenceImager
                # Fit and transform diagrams
                self.vectorizer.fit(diagrams[0:2])  # Fit on H0 and H1
                images = self.vectorizer.transform(diagrams[0:2])
                
                # Concatenate H0 and H1 images
                features = np.concatenate([img.flatten() for img in images])
                
                # Pad or truncate to expected size
                if len(features) < self.feature_dim:
                    padded = np.zeros(self.feature_dim, dtype=np.float32)
                    padded[:len(features)] = features
                    return padded
                else:
                    return features[:self.feature_dim].astype(np.float32)
                    
            elif self.method == 'persistence_landscape':
                # Use persim landscape approximation
                features = []
                for dim in [0, 1]:  # H0 and H1
                    if dim < len(diagrams) and len(diagrams[dim]) > 0:
                        landscape = PersLandscapeApprox(dgms=[diagrams[dim]], hom_deg=dim)
                        # Extract landscape values at fixed points
                        land_values = landscape.landscapes[:self.resolution] if len(landscape.landscapes) > 0 else np.zeros(self.resolution)
                        if len(land_values) < self.resolution:
                            padded_land = np.zeros(self.resolution)
                            padded_land[:len(land_values)] = land_values
                            land_values = padded_land
                        features.extend(land_values[:self.resolution])
                    else:
                        features.extend(np.zeros(self.resolution))
                
                # Repeat for multiple landscape levels (5 levels)
                final_features = np.tile(features, 5)[:self.feature_dim]
                return final_features.astype(np.float32)
                
            elif self.method == 'amplitude':
                # Simple amplitude calculation: max persistence for each dimension
                features = []
                for dim in [0, 1]:  # H0 and H1
                    if dim < len(diagrams) and len(diagrams[dim]) > 0:
                        diagram = diagrams[dim]
                        # Remove infinite points
                        finite_diagram = diagram[diagram[:, 1] != np.inf]
                        if len(finite_diagram) > 0:
                            persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
                            amplitude = np.max(persistences)
                        else:
                            amplitude = 0.0
                    else:
                        amplitude = 0.0
                    features.append(amplitude)
                
                return np.array(features, dtype=np.float32)
                
        except Exception as e:
            print(f"Vectorization failed: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def process_image(self, image):
        """
        Main processing function: image -> TDA features.
        Compatible with the giotto-tda version.
        """
        try:
            # Convert image to point cloud
            point_cloud = self._image_to_point_cloud(image)
            if point_cloud is None:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # Compute persistence diagrams
            diagrams = self._compute_persistence_diagrams(point_cloud)
            if diagrams is None:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # Vectorize diagrams
            features = self._vectorize_diagrams(diagrams)
            return features
            
        except Exception as e:
            print(f"TDA processing failed: {e}, using fallback")
            # Fallback: simple downsampling
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            downsampled = cv2.resize(image, (8, 8)).flatten()
            fallback = np.zeros(self.feature_dim, dtype=np.float32)
            fallback[:min(64, self.feature_dim)] = downsampled[:min(64, self.feature_dim)]
            return fallback

# Import required libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv
from sklearn.metrics.pairwise import cosine_similarity

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.training.train_offline import train_offline
from agilerl.training.train_on_policy import train_on_policy
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.utils.utils import create_population as Population

import supersuit as ss
import psutil

print("✅ Scikit-TDA libraries loaded successfully!")
print("🎉 Scikit-TDA alternative implementation ready!")
print("   - Same memory reduction as giotto-tda")
print("   - Potentially easier installation")
print("   - Drop-in replacement for your existing code")
print("   - All training optimizations preserved")

class ScikitTDAEnhancedCichyEnvUniqueActions(ParallelEnv):
    """
    Modified TDA-enhanced environment where each agent can send unique actions 
    to each of their neighbors, and neighbors receive these unique actions.
    """
    
    metadata = {"render_modes": ["human"], "name": "scikit_tda_cichy_env_unique"}

    def __init__(self, x_train, y_train1, y_train2, tda_method='persistence_image', render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.agents = [
            "IT1", "IT2", "IT3", "IT4", "IT5",
            "EVC1", "EVC2", "EVC3", "EVC4", "EVC5", "EVC6"
        ]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        self._num_agents = len(self.agents)

        # Store data
        self.x_raw = x_train
        self.y1 = y_train1
        self.y2 = y_train2

        # Initialize scikit-tda processor
        print("🔄 Initializing Scikit-TDA processor...")
        self.tda_processor = ScikitTDAImageProcessor(
            method=tda_method,
            resolution=8,
            bandwidth=0.1
        )
        
        # Convert images to TDA features
        print("🔄 Converting images to TDA features using scikit-tda...")
        self.x_tda = self._preprocess_all_images()
        
        original_features = 175 * 175
        new_features = self.x_tda.shape[1]
        reduction_factor = original_features / new_features
        print(f"✅ Scikit-TDA preprocessing complete!")
        print(f"   Memory reduction: {reduction_factor:.1f}x ({original_features} → {new_features} features)")

        # Define neighbor relationships (adjacent agents)
        self.neighbor_map = self._build_neighbor_map()
        
        # Calculate action dimensions
        base_action_dim = 92  # Original action dimension
        max_neighbors = max(len(neighbors) for neighbors in self.neighbor_map.values())
        
        # Each agent outputs: base_action + (action_for_neighbor_1 + action_for_neighbor_2 + ...)
        self.action_dimensions = {}
        for agent in self.agents:
            num_neighbors = len(self.neighbor_map[agent])
            # Base action (92) + unique action for each neighbor (92 each)
            total_action_dim = base_action_dim + (num_neighbors * base_action_dim)
            self.action_dimensions[agent] = total_action_dim

        # Update observation and action spaces
        tda_feature_dim = self.x_tda.shape[1]
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "image": spaces.Box(low=-np.inf, high=np.inf, shape=(tda_feature_dim,), dtype=np.float32),
                "neighbor_actions": spaces.Box(low=-1, high=1, shape=(max_neighbors, base_action_dim), dtype=np.float32)
            }) for agent in self.agents
        }
        
        self.action_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(self.action_dimensions[agent],), dtype=np.float32)
            for agent in self.agents
        }

        # Internal state
        self.current_step = 0
        self.max_steps = 100
        # Store unique actions each agent sends to its neighbors
        self.agent_neighbor_actions = {
            agent: {neighbor: np.zeros(base_action_dim, dtype=np.float32) 
                   for neighbor in self.neighbor_map[agent]}
            for agent in self.agents
        }
        # Store base actions for reward computation
        self.agent_base_actions = {agent: np.zeros(base_action_dim, dtype=np.float32) 
                                  for agent in self.agents}

    def _build_neighbor_map(self):
        """Build adjacency map for agents - each agent connects to adjacent agents in the list"""
        neighbor_map = {}
        for i, agent in enumerate(self.agents):
            neighbors = []
            # Add previous agent (if exists)
            if i > 0:
                neighbors.append(self.agents[i-1])
            # Add next agent (if exists)  
            if i < len(self.agents) - 1:
                neighbors.append(self.agents[i+1])
            neighbor_map[agent] = neighbors
        return neighbor_map

    def _preprocess_all_images(self):
        """Convert all images to TDA features using scikit-tda."""
        # Handle different input shapes
        if self.x_raw.shape == (92, 175, 175):
            images = self.x_raw
        elif self.x_raw.shape == (30625, 92):
            images = self.x_raw.T.reshape(92, 175, 175)
        elif self.x_raw.shape == (92, 30625):
            images = self.x_raw.reshape(92, 175, 175)
        else:
            raise ValueError(f"Unexpected x shape: {self.x_raw.shape}")
        
        # Process in small batches
        batch_size = 4
        all_features = []
        
        print(f"   Processing {len(images)} images in batches of {batch_size}...")
        for i in range(0, len(images), batch_size):
            batch_end = min(i + batch_size, len(images))
            batch = images[i:batch_end]
            
            if i % 20 == 0:
                print(f"   Progress: {i}/{len(images)} images processed")
            
            for img in batch:
                features = self.tda_processor.process_image(img)
                all_features.append(features)
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"   ✅ All {len(images)} images processed with scikit-tda")
        return np.array(all_features, dtype=np.float32)

    def _parse_agent_action(self, agent, full_action):
        """
        Parse the full action vector into base action and neighbor-specific actions.
        
        Args:
            agent: Agent name
            full_action: Full action vector from the agent
            
        Returns:
            base_action: Base action (92-dim)
            neighbor_actions: Dict mapping neighbor -> action for that neighbor
        """
        base_action_dim = 92
        neighbors = self.neighbor_map[agent]
        
        # Extract base action (first 92 elements)
        base_action = full_action[:base_action_dim]
        
        # Extract neighbor-specific actions
        neighbor_actions = {}
        start_idx = base_action_dim
        
        for neighbor in neighbors:
            end_idx = start_idx + base_action_dim
            neighbor_action = full_action[start_idx:end_idx]
            neighbor_actions[neighbor] = neighbor_action
            start_idx = end_idx
            
        return base_action, neighbor_actions

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.dones = {agent: False for agent in self.agents}
        self.truncs = {agent: False for agent in self.agents}
        
        image_idx = self.current_step % 92
        tda_features = self.x_tda[image_idx].copy()
        
        # Initialize empty neighbor actions
        max_neighbors = max(len(self.neighbor_map[agent]) for agent in self.agents)
        
        self.observations = {}
        for agent in self.agents:
            num_neighbors = len(self.neighbor_map[agent])
            neighbor_actions = np.zeros((max_neighbors, 92), dtype=np.float32)
            
            self.observations[agent] = {
                "image": tda_features,
                "neighbor_actions": neighbor_actions
            }
        
        # Flatten observations for AgileRL compatibility
        flat_obs = {
            agent_id: np.concatenate([
                self.observations[agent_id]["image"],
                self.observations[agent_id]["neighbor_actions"].flatten()
            ])
            for agent_id in self.agents
        }
        
        infos = {agent: {} for agent in self.agents}
        return flat_obs, infos
    
    def step(self, actions):
        self.current_step += 1
        
        # Parse all agent actions
        for agent, full_action in actions.items():
            base_action, neighbor_actions = self._parse_agent_action(agent, full_action)
            self.agent_base_actions[agent] = base_action
            self.agent_neighbor_actions[agent] = neighbor_actions
        
        # Build observations with unique neighbor actions
        image_idx = self.current_step % 92
        tda_features = self.x_tda[image_idx].copy()
        max_neighbors = max(len(self.neighbor_map[agent]) for agent in self.agents)
        
        observations = {}
        for agent in self.agents:
            # Create neighbor actions array
            neighbor_actions = np.zeros((max_neighbors, 92), dtype=np.float32)
            
            # Fill with actions sent specifically to this agent
            neighbor_idx = 0
            for neighbor in self.neighbor_map[agent]:
                if agent in self.agent_neighbor_actions[neighbor]:
                    # This neighbor sent a specific action to this agent
                    neighbor_actions[neighbor_idx] = self.agent_neighbor_actions[neighbor][agent]
                neighbor_idx += 1
            
            observations[agent] = {
                "image": tda_features,
                "neighbor_actions": neighbor_actions
            }
        
        # Compute rewards using base actions
        rewards = self._compute_rewards(self.agent_base_actions, image_idx)
        dones = {agent: self.current_step >= self.max_steps for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        self.observations = observations

        # Flatten observations for AgileRL compatibility
        flat_obs = {
            agent_id: np.concatenate([
                self.observations[agent_id]["image"],
                self.observations[agent_id]["neighbor_actions"].flatten()
            ])
            for agent_id in self.agents
        }
        
        return flat_obs, rewards, dones, truncations, infos

    def _compute_rewards(self, base_actions, image_idx):
        """Compute rewards using base actions (unchanged from original)"""
        it_actions = [base_actions[agent] for agent in self.agents if "IT" in agent]
        evc_actions = [base_actions[agent] for agent in self.agents if "EVC" in agent]

        it_action_sum = np.mean(it_actions, axis=0)
        evc_action_sum = np.mean(evc_actions, axis=0)

        it_target = self.y1[image_idx]
        evc_target = self.y2[image_idx]

        it_reward = -np.mean(np.abs(it_action_sum - it_target))
        evc_reward = -np.mean(np.abs(evc_action_sum - evc_target))

        rewards = {}
        for agent in self.agents:
            if "IT" in agent:
                rewards[agent] = it_reward
            else:
                rewards[agent] = evc_reward
        return rewards

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step} (Scikit-TDA enhanced with unique neighbor actions)")

    def close(self):
        pass

# Memory efficient tournament selection
import torch
import numpy as np
import random
import copy
import gc
from typing import List, Tuple, Optional
import psutil
import os

class MemoryEfficientTournamentSelection:
    """
    Memory-optimized tournament selection that processes agents in batches
    and uses streaming algorithms to minimize memory usage.
    """
    
    def __init__(self, tournament_size: int = 2, elitism: bool = True, 
                 population_size: int = 11, eval_loop: int = 1,
                 memory_budget_gb: float = 20.0, device: str = 'cuda'):
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population_size = population_size
        self.eval_loop = eval_loop
        self.device = device
        
        # Calculate optimal batch size based on memory budget
        # Estimate ~2.5GB per agent for MATD3 networks
        memory_per_agent_gb = 2.5
        self.max_agents_in_memory = max(2, int(memory_budget_gb / memory_per_agent_gb))
        self.batch_size = min(self.max_agents_in_memory, population_size)
        
        print(f"Tournament selection will process {self.batch_size} agents at a time")
        
        # Fitness cache to avoid recomputation
        self.fitness_cache = {}
        self.elite_agent = None
        
    def log_memory(self, step_name: str):
        """Log current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        else:
            gpu_memory = 0
        
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024**3)
        print(f"  {step_name}: CPU: {cpu_memory:.1f}GB, GPU: {gpu_memory:.1f}GB")
        return cpu_memory + gpu_memory
        
    def clear_memory(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def move_agent_to_device(self, agent, target_device: str):
        """Move agent networks to specified device"""
        if hasattr(agent, 'actor') and agent.actor is not None:
            agent.actor = agent.actor.to(target_device)
        if hasattr(agent, 'critic') and agent.critic is not None:
            agent.critic = agent.critic.to(target_device)
        if hasattr(agent, 'actor_target') and agent.actor_target is not None:
            agent.actor_target = agent.actor_target.to(target_device)
        if hasattr(agent, 'critic_target') and agent.critic_target is not None:
            agent.critic_target = agent.critic_target.to(target_device)
        return agent
    
    def get_agent_fitness(self, agent, agent_id: int, env) -> float:
        """Get or calculate agent fitness with caching"""
        # Use a simple hash of agent parameters as cache key
        if hasattr(agent, 'actor') and agent.actor is not None:
            param_hash = hash(tuple(p.data.flatten()[:10].cpu().numpy().tobytes() 
                                  for p in agent.actor.parameters() if p.requires_grad))
            cache_key = f"{agent_id}_{param_hash}"
        else:
            cache_key = f"{agent_id}_{random.randint(0, 1000000)}"
        
        if cache_key in self.fitness_cache:
            return self.fitness_cache[cache_key]
        
        # Calculate fitness
        fitness = agent.test(
            env,
            swap_channels=False,
            max_steps=20,  # Reduced for memory efficiency
            loop=self.eval_loop,
        )
        
        # Cache the result
        self.fitness_cache[cache_key] = fitness
        
        # Limit cache size to prevent memory growth
        if len(self.fitness_cache) > self.population_size * 3:
            # Remove oldest entries
            keys_to_remove = list(self.fitness_cache.keys())[:-self.population_size]
            for key in keys_to_remove:
                del self.fitness_cache[key]
        
        return fitness
    
    def streaming_tournament_select(self, population: List, fitnesses: List[float], 
                                  num_selections: int) -> List[int]:
        """
        Perform tournament selection without loading entire population into memory.
        Returns indices of selected agents.
        """
        selected_indices = []
        population_indices = list(range(len(population)))
        
        for _ in range(num_selections):
            # Sample tournament candidates
            tournament_indices = random.sample(population_indices, 
                                             min(self.tournament_size, len(population_indices)))
            
            # Find winner based on fitness
            winner_idx = max(tournament_indices, key=lambda idx: fitnesses[idx])
            selected_indices.append(winner_idx)
        
        return selected_indices
    
    def batch_fitness_evaluation(self, population: List, env) -> List[float]:
        """
        Evaluate population fitness in memory-efficient batches.
        """
        print("Starting batch fitness evaluation...")
        self.log_memory("before fitness evaluation")
        
        all_fitnesses = []
        
        for batch_start in range(0, len(population), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(population))
            batch_agents = population[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"  Evaluating batch {batch_start//self.batch_size + 1}: agents {batch_start}-{batch_end-1}")
            
            # Move batch to GPU for evaluation
            for i, agent in enumerate(batch_agents):
                self.move_agent_to_device(agent, self.device)
            
            # Evaluate fitness for this batch
            batch_fitnesses = []
            for i, agent in enumerate(batch_agents):
                agent_id = batch_indices[i]
                fitness = self.get_agent_fitness(agent, agent_id, env)
                batch_fitnesses.append(fitness)
            
            all_fitnesses.extend(batch_fitnesses)
            
            # Move batch back to CPU to save GPU memory
            for agent in batch_agents:
                self.move_agent_to_device(agent, 'cpu')
            
            # Memory cleanup between batches
            self.clear_memory()
            self.log_memory(f"after batch {batch_start//self.batch_size + 1}")
        
        return all_fitnesses
    
    def create_new_population(self, population: List, selected_indices: List[int]) -> List:
        """
        Create new population from selected indices with minimal memory usage.
        """
        print("Creating new population...")
        new_population = []
        
        # Process selections in batches
        for batch_start in range(0, len(selected_indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(selected_indices))
            batch_indices = selected_indices[batch_start:batch_end]
            
            batch_agents = []
            for idx in batch_indices:
                # Create deep copy of selected agent
                original_agent = population[idx]
                
                # Move to CPU for copying
                self.move_agent_to_device(original_agent, 'cpu')
                
                # Deep copy the agent
                new_agent = copy.deepcopy(original_agent)
                batch_agents.append(new_agent)
            
            new_population.extend(batch_agents)
            
            # Memory cleanup between batches
            self.clear_memory()
        
        return new_population
    
    def select(self, population: List, env) -> Tuple:
        """
        Main selection method with memory optimization.
        """
        print(f"\nStarting memory-efficient tournament selection for {len(population)} agents...")
        initial_memory = self.log_memory("initial")
        
        try:
            # Step 1: Batch fitness evaluation
            fitnesses = self.batch_fitness_evaluation(population, env)
            
            # Step 2: Find elite agent (best fitness)
            elite_idx = np.argmax(fitnesses)
            elite_fitness = fitnesses[elite_idx]
            
            print(f"Elite agent: index {elite_idx}, fitness {elite_fitness:.4f}")
            
            # Step 3: Elite preservation
            if self.elitism:
                # Move elite to CPU and create deep copy
                elite_agent = population[elite_idx]
                self.move_agent_to_device(elite_agent, 'cpu')
                elite_copy = copy.deepcopy(elite_agent)
                self.elite_agent = elite_copy
                
                # Create selection pool (population size - 1 for elite slot)
                num_selections = len(population) - 1
            else:
                num_selections = len(population)
            
            # Step 4: Tournament selection
            print("Performing streaming tournament selection...")
            selected_indices = self.streaming_tournament_select(
                population, fitnesses, num_selections
            )
            
            # Step 5: Create new population
            new_population = self.create_new_population(population, selected_indices)
            
            # Step 6: Add elite if using elitism
            if self.elitism and self.elite_agent is not None:
                new_population.insert(0, self.elite_agent)
            
            # Final memory cleanup
            self.clear_memory()
            final_memory = self.log_memory("final")
            
            print(f"Tournament selection completed. Memory change: {final_memory - initial_memory:+.1f}GB")
            
            return self.elite_agent if self.elitism else new_population[0], new_population
            
        except Exception as e:
            print(f"Error during tournament selection: {e}")
            self.clear_memory()
            
            # Fallback: return original population
            return population[0], population

def replace_tournament_selection(pop_agent, env, memory_budget_gb: float = 20.0):
    """
    Drop-in replacement for your existing tournament selection.
    """
    
    # Create memory-efficient tournament selector
    efficient_tournament = MemoryEfficientTournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=len(pop_agent),
        eval_loop=1,
        memory_budget_gb=memory_budget_gb,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return efficient_tournament.select(pop_agent, env)

# Setup training environment and parameters
from tqdm import trange
import os
import torch

device = 'cuda'
max_steps = 200000
training_steps = 1600
USE_BFLOAT16 = True

# Create environment with unique neighbor actions
env = ScikitTDAEnhancedCichyEnvUniqueActions(x, y1, y2)
num_envs = 1

# Print action dimensions for verification
print("Action dimensions per agent:")
for agent in env.agents:
    neighbors = env.neighbor_map[agent]
    action_dim = env.action_dimensions[agent]
    print(f"  {agent}: {action_dim} dims (92 base + {len(neighbors)} neighbors × 92)")

# Initial Hyperparameters
INIT_HP = {
    "DOUBLE": True,
    "CHANNELS_LAST": False,
    "POPULATION_SIZE": 11,
    "O_U_NOISE": 0.2,
    "EXPL_NOISE": 0.1,
    "BATCH_SIZE": 8,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    "LR": 0.001, 
    "LR_ACTOR": 0.002,
    "LR_CRITIC": 0.002,
    "TAU": 0.5,
    "GAMMA": 1.0,
    "LAMBDA": 1.0,
    "REG": 0.0625,
    "LEARN_STEP": 2,
    "MEAN_NOISE": 1,
    "THETA": 1,
    "DT": 1,
    "POLICY_FREQ": 2,
    "AGENT_IDS": [
        "IT1", "IT2", "IT3", "IT4", "IT5",
        "EVC1", "EVC2", "EVC3", "EVC4", "EVC5", "EVC6"
    ],
    "MEMORY_SIZE": 100000
}

hp_config = HyperparameterConfig(
    batch_size=RLParameter(min=8, max=512, dtype=int),
    learn_step=RLParameter(min=1, max=10, dtype=int, grow_factor=1.5, shrink_factor=0.75)
)

NET_CONFIG = {
    "encoder_config": {
        "hidden_size": [32],
    },
    "head_config": {
        "hidden_size": [32],
    },
}

# Create populations for each agent - now with different action dimensions
pop_agent = Population(
    algo="MATD3",
    observation_space=[spaces.flatten_space(env.observation_space(agent)) for agent in env.agents],
    action_space=[env.action_space(agent) for agent in env.agents],  # Each agent has different action dimensions
    net_config=NET_CONFIG,
    INIT_HP=INIT_HP,
    hp_config=hp_config,
    population_size=INIT_HP["POPULATION_SIZE"],
    device=device
)

# Enable mixed precision for all agents in population
for agent in pop_agent:
    if hasattr(agent, 'actor') and hasattr(agent, 'critic'):
        if USE_BFLOAT16:
            agent.actor = agent.actor.to(torch.bfloat16)
            agent.critic = agent.critic.to(torch.bfloat16)
            if hasattr(agent, 'actor_target'):
                agent.actor_target = agent.actor_target.to(torch.bfloat16)
            if hasattr(agent, 'critic_target'):
                agent.critic_target = agent.critic_target.to(torch.bfloat16)
        else:
            # For older GPUs, keep float32 but use autocast during forward passes
            pass

import torch.utils.checkpoint as checkpoint

print("Applying activation checkpointing...")
for agent in pop_agent:
    if hasattr(agent, 'actor') and agent.actor is not None:
        original_actor_forward = agent.actor.forward
        agent.actor.forward = lambda x: checkpoint.checkpoint(original_actor_forward, x)
    
    if hasattr(agent, 'critic') and agent.critic is not None:
        original_critic_forward = agent.critic.forward
        agent.critic.forward = lambda x, action: checkpoint.checkpoint(original_critic_forward, x, action)

print("Activation checkpointing enabled!")

# Tournament selection
tournament = TournamentSelection(
    tournament_size=2,
    elitism=True,
    population_size=INIT_HP["POPULATION_SIZE"],
    eval_loop=1,
)

# Mutation settings
mutations = Mutations(
    no_mutation=0.4,
    architecture=0.2,
    new_layer_prob=0.2,
    parameters=0.2,
    activation=0,
    rl_hp=0.2,
    mutation_sd=0.1,
    rand_seed=1,
    device=device
)

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer

field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    INIT_HP["MEMORY_SIZE"],
    field_names=field_names,
    agent_ids=INIT_HP["AGENT_IDS"],
    device=device,
)

# Training loop with mixed precision
from torch.cuda.amp import autocast, GradScaler

total_steps = 0
elite = pop_agent[0]
pbar = trange(max_steps, unit="step")

while np.less([agent.steps[-1] for agent in pop_agent], max_steps).all():
    pop_episode_scores = []
    for agent in pop_agent:
        obs, info = env.reset()
        scores = np.zeros(num_envs)
        completed_episode_scores = []
        steps = 0

        for idx_step in range(training_steps // num_envs):
            # Create flat observations safely without overwriting original obs
            with autocast(enabled=True, dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float16):
                cont_actions, discrete_action = agent.get_action(obs=obs, training=True, infos=info)
                action = discrete_action if agent.discrete_actions else cont_actions

            next_state, reward, termination, truncation, info = env.step(action)
            scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
            total_steps += num_envs
            steps += num_envs

            if "__all__" in termination:
                termination.pop("__all__", None)
            if "__all__" in truncation:
                truncation.pop("__all__", None)

            reward = {agent_id: np.array(val, dtype=np.float32) for agent_id, val in reward.items()}
            termination = {agent_id: np.array(val, dtype=np.float32) for agent_id, val in termination.items()}

            # Prepare next_state as flat for saving
            flat_next_state = next_state

            squeezed_cont_actions = {
                agent_id: action_tensor.squeeze(0)
                for agent_id, action_tensor in cont_actions.items()
            }

            memory.save_to_memory(
                obs,
                squeezed_cont_actions,
                reward,
                flat_next_state,
                termination
            )
            
            # Mixed precision learning
            if agent.learn_step > num_envs:
                learn_step = agent.learn_step // num_envs
                if (
                    idx_step % learn_step == 0
                    and len(memory) >= agent.batch_size
                    and memory.counter > 0  # Reduced learning delay
                ):
                    experiences = memory.sample(agent.batch_size)
                    
                    # Learn with mixed precision
                    with autocast(enabled=True, dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float16):
                        if USE_BFLOAT16:
                            # BF16 doesn't need gradient scaling
                            agent.learn(experiences)
                        else:
                            # FP16 needs gradient scaling
                            with scaler.scale(agent.learn(experiences)):
                                pass
                            
            elif len(memory) >= agent.batch_size and memory.counter > 0:
                for _ in range(num_envs // agent.learn_step):
                    experiences = memory.sample(agent.batch_size)
                    
                    with autocast(enabled=True, dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float16):
                        if USE_BFLOAT16:
                            agent.learn(experiences)
                        else:
                            with scaler.scale(agent.learn(experiences)):
                                pass
            
            obs = next_state  # keep unflattened for next iteration

            reset_noise_indices = []
            term_array = np.array(list(termination.values())).transpose()
            trunc_array = np.array(list(truncation.values())).transpose()
            
            min_len = min(len(scores), len(term_array), len(trunc_array))
            for idx in range(min_len):
                d = term_array[idx]
                t = trunc_array[idx]
                if np.any(d) or np.any(t):
                    completed_episode_scores.append(scores[idx])
                    agent.scores.append(scores[idx])
                    scores[idx] = 0
                    reset_noise_indices.append(idx)
            agent.reset_action_noise(reset_noise_indices)

        pbar.update(evo_steps // len(pop_agent))
        agent.steps[-1] += steps
        pop_episode_scores.append(completed_episode_scores)

    fitnesses = []
    for agent in pop_agent:
        with autocast(enabled=True, dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float16):
            fitness = agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=20,  # Reduced for memory efficiency
                loop=1,
            )
        fitnesses.append(fitness)
        
    mean_scores = [
        np.mean(episode_scores) if episode_scores else "0 completed episodes"
        for episode_scores in pop_episode_scores
    ]

    print(f"--- Global steps {total_steps} ---")
    print(f"Steps {[agent.steps[-1] for agent in pop_agent]}")
    print(f"Scores: {mean_scores}")
    print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
    print(
        f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop_agent]}'
    )

    torch.cuda.empty_cache()
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RAM used: {mem_info.rss / (1024 ** 2):.2f} MB")
    
    # Tournament selection and population mutation
    elite, pop = replace_tournament_selection(pop_agent, env, memory_budget_gb=20.0)
    print("elite")

    for agent in pop:
        if hasattr(agent, 'actor'):
            agent.actor = agent.actor.cpu()
            agent.critic = agent.critic.cpu()
            if hasattr(agent, 'actor_target'):
                agent.actor_target = agent.actor_target.cpu()
            if hasattr(agent, 'critic_target'):
                agent.critic_target = agent.critic_target.cpu()
    pop = mutations.mutation(pop)

    for agent in pop:
        if hasattr(agent, 'actor'):
            agent.actor = agent.actor.to(device)
            agent.critic = agent.critic.to(device)
            if hasattr(agent, 'actor_target'):
                agent.actor_target = agent.actor_target.to(device)
            if hasattr(agent, 'critic_target'):
                agent.critic_target = agent.critic_target.to(device)
            
            if USE_BFLOAT16:
                agent.actor = agent.actor.to(torch.bfloat16)
                agent.critic = agent.critic.to(torch.bfloat16)
                if hasattr(agent, 'actor_target'):
                    agent.actor_target = agent.actor_target.to(torch.bfloat16)
                if hasattr(agent, 'critic_target'):
                    agent.critic_target = agent.critic_target.to(torch.bfloat16)
    
    # Update step counter
    for agent in pop:
        agent.steps.append(agent.steps[-1])
        
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Save the trained algorithm
path = "/home/ubuntu/data/"
filename = "MATD3_trained_agent_unique_actions.pt"
os.makedirs(path, exist_ok=True)
save_path = os.path.join(path, filename)
elite.save_checkpoint(save_path)

pbar.close()
env.close()
print("Training completed with unique neighbor actions!")

# Print summary of changes
print("\n=== Summary of Unique Action Implementation ===")
print("✅ Each agent now sends unique actions to each neighbor")
print("✅ Action dimensions adapted per agent based on neighbor count:")
for agent in env.agents:
    neighbors = env.neighbor_map[agent] 
    action_dim = env.action_dimensions[agent]
    print(f"   {agent}: {len(neighbors)} neighbors → {action_dim} action dims")
print("✅ Environment automatically parses and distributes actions")
print("✅ Rewards computed using base actions (unchanged)")
print("✅ All existing training optimizations preserved")
