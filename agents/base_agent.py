"""
Base Agent class for Network Traffic Management
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging

from config.config import Config


class BaseNetworkPolicy(nn.Module):
    """Base neural network policy for network routing"""
    
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Build the network
        layers = []
        prev_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Help with training stability
                nn.Dropout(0.1)  # Light regularization
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*layers)
        
        # Network-specific layers (to be overridden by subclasses)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        features = self.shared_network(observations)
        return self.output_layer(features)


class NetworkValueFunction(nn.Module):
    """Value function network for network optimization"""
    
    def __init__(self, observation_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Value function outputs a single value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state values"""
        return self.network(observations).squeeze(-1)


class BaseAgent(ABC):
    """Abstract base class for network traffic management agents"""
    
    def __init__(self, config: Config, observation_space, action_space):
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize networks (to be implemented by subclasses)
        self.policy_net = None
        self.value_net = None
        self.optimizer = None
        
        # Training statistics
        self.total_steps = 0
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'learning_rate': [],
            'explained_variance': []
        }
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation"""
        pass
    
    @abstractmethod
    def update(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Update the agent's policy given a batch of experience"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save the agent's model"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load the agent's model"""
        pass
    
    def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Preprocess observation for neural network input"""
        # Normalize observations if configured
        if self.config.environment.normalize_observations:
            observation = self._normalize_observation(observation)
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        # Add batch dimension if needed
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        return obs_tensor
    
    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize observation to improve training stability"""
        # Simple min-max normalization
        # In practice, you might want running statistics
        obs_min = 0.0
        obs_max = 1.0
        
        # Clip extreme values
        observation = np.clip(observation, obs_min, obs_max)
        
        # Normalize to [-1, 1] range
        normalized = 2.0 * (observation - obs_min) / (obs_max - obs_min) - 1.0
        
        return normalized
    
    def compute_explained_variance(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute explained variance for value function evaluation"""
        if len(y_true) < 2:
            return 0.0
        
        var_y = torch.var(y_true, unbiased=False)
        if var_y == 0:
            return 0.0
        
        return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)
    
    def get_training_stats(self) -> Dict[str, list]:
        """Get training statistics"""
        return self.training_stats.copy()
    
    def reset_training_stats(self):
        """Reset training statistics"""
        for key in self.training_stats:
            self.training_stats[key] = []
    
    def log_training_step(self, step_stats: Dict[str, float]):
        """Log training step statistics"""
        for key, value in step_stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        self.total_steps += 1
        
        # Log every N steps
        if self.total_steps % 1000 == 0:
            avg_stats = {
                key: np.mean(values[-100:]) if values else 0.0
                for key, values in self.training_stats.items()
            }
            
            log_msg = f"Step {self.total_steps:6d} - "
            log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in avg_stats.items()])
            self.logger.info(log_msg)


class NetworkPolicyNet(BaseNetworkPolicy):
    """Policy network specifically designed for network routing decisions"""
    
    def __init__(self, observation_dim: int, action_dim: int, 
                 max_flows: int = 50, num_paths: int = 5,
                 hidden_dims: list = [512, 512, 256]):
        
        super().__init__(observation_dim, action_dim, hidden_dims)
        
        self.max_flows = max_flows
        self.num_paths = num_paths
        
        # Network-aware architecture
        # The observation contains network state + flow information
        
        # Separate processing for network state and flow information
        network_state_dim = observation_dim - (max_flows * 6)  # 6 features per flow
        flow_info_dim = max_flows * 6
        
        # Network state processor
        self.network_processor = nn.Sequential(
            nn.Linear(network_state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        
        # Flow information processor
        self.flow_processor = nn.Sequential(
            nn.Linear(flow_info_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        
        # Combined processing
        self.combined_processor = nn.Sequential(
            nn.Linear(512, 512),  # 256 + 256
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Output layers for each flow's routing decision
        self.flow_routing_heads = nn.ModuleList([
            nn.Linear(256, num_paths) for _ in range(max_flows)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with network-aware processing"""
        batch_size = observations.shape[0]
        
        # Split observation into network state and flow information
        network_state_dim = observations.shape[1] - (self.max_flows * 6)
        
        network_state = observations[:, :network_state_dim]
        flow_info = observations[:, network_state_dim:]
        
        # Process network state and flow information separately
        network_features = self.network_processor(network_state)
        flow_features = self.flow_processor(flow_info)
        
        # Combine features
        combined_features = torch.cat([network_features, flow_features], dim=1)
        shared_features = self.combined_processor(combined_features)
        
        # Generate routing decisions for each flow
        flow_logits = []
        for i in range(self.max_flows):
            logits = self.flow_routing_heads[i](shared_features)
            flow_logits.append(logits)
        
        # Stack logits: (batch_size, max_flows, num_paths)
        output = torch.stack(flow_logits, dim=1)
        
        # Reshape to match action space: (batch_size, max_flows * num_paths)
        output = output.view(batch_size, -1)
        
        return output


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms"""
    
    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate arrays for efficiency
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
    
    def add(self, observation: np.ndarray, action: np.ndarray, reward: float,
            next_observation: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences"""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling"""
        return self.size >= batch_size


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms like PPO"""
    
    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        self.capacity = capacity
        self.position = 0
        self.full = False
        
        # Pre-allocate arrays
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Computed during rollout processing
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
    
    def add(self, observation: np.ndarray, action: np.ndarray, reward: float,
            value: float, log_prob: float, done: bool):
        """Add experience to buffer"""
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.dones[self.position] = done
        
        self.position += 1
        if self.position == self.capacity:
            self.full = True
            self.position = 0
    
    def compute_advantages_and_returns(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute advantages and returns using GAE"""
        last_gae_lambda = 0
        
        for step in reversed(range(self.get_size())):
            if step == self.get_size() - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            delta = (self.rewards[step] + gamma * next_value * next_non_terminal - 
                    self.values[step])
            last_gae_lambda = (delta + gamma * gae_lambda * next_non_terminal * 
                             last_gae_lambda)
            self.advantages[step] = last_gae_lambda
        
        self.returns = self.advantages + self.values[:self.get_size()]
    
    def get_size(self) -> int:
        """Get current buffer size"""
        return self.capacity if self.full else self.position
    
    def get_batch(self, batch_size: int):
        """Get batches for training"""
        size = self.get_size()
        indices = np.random.permutation(size)
        
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]
            
            yield {
                'observations': self.observations[batch_indices],
                'actions': self.actions[batch_indices],
                'old_log_probs': self.log_probs[batch_indices],
                'advantages': self.advantages[batch_indices],
                'returns': self.returns[batch_indices],
                'values': self.values[batch_indices]
            }
    
    def clear(self):
        """Clear the buffer"""
        self.position = 0
        self.full = False