"""
Soft Actor-Critic (SAC) Agent for Network Traffic Management
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Any, Tuple, Optional
import logging

from agents.base_agent import BaseAgent, BaseNetworkPolicy, NetworkValueFunction, ReplayBuffer
from config.config import Config


class SACPolicyNet(BaseNetworkPolicy):
    """SAC Policy Network with continuous actions"""
    
    def __init__(self, observation_dim: int, action_dim: int, 
                 max_flows: int = 50, num_paths: int = 5,
                 hidden_dims: list = [512, 512, 256]):
        
        super().__init__(observation_dim, action_dim, hidden_dims)
        
        self.max_flows = max_flows
        self.num_paths = num_paths
        self.log_std_min = -20
        self.log_std_max = 2
        
        # For SAC, we output mean and log_std for continuous actions
        # We'll use a continuous representation and then convert to discrete
        
        # Network architecture similar to PPO but with continuous outputs
        final_dim = hidden_dims[-1] if hidden_dims else 256
        
        # Mean and log_std outputs for each flow's path selection
        self.mean_layer = nn.Linear(final_dim, max_flows)
        self.log_std_layer = nn.Linear(final_dim, max_flows)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std"""
        features = self.shared_network(observations)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, observations: torch.Tensor, deterministic: bool = False):
        """Sample actions from the policy"""
        mean, log_std = self.forward(observations)
        std = log_std.exp()
        
        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            # Reparameterization trick
            action = normal.rsample()
        
        # Convert continuous actions to discrete path selections
        # Use softmax to get probabilities, then sample
        action_probs = F.softmax(action, dim=-1)
        
        # For discrete action selection, we can use Gumbel-Softmax
        if not deterministic:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(action_probs) + 1e-8) + 1e-8)
            action_logits = torch.log(action_probs + 1e-8) + gumbel_noise
            discrete_actions = F.softmax(action_logits, dim=-1)
        else:
            discrete_actions = action_probs
        
        # Convert to discrete indices
        discrete_indices = torch.multinomial(discrete_actions, 1).squeeze(-1)
        
        return discrete_indices, action_probs, mean, log_std
    
    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor):
        """Compute log probability of actions"""
        mean, log_std = self.forward(observations)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        log_prob = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        
        return log_prob


class SACCritic(nn.Module):
    """SAC Critic Network (Q-function)"""
    
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list = [512, 512, 256]):
        super().__init__()
        
        # Q1 network
        q1_layers = []
        prev_dim = observation_dim + action_dim
        
        for hidden_dim in hidden_dims:
            q1_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2 network (twin critics)
        q2_layers = []
        prev_dim = observation_dim + action_dim
        
        for hidden_dim in hidden_dims:
            q2_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        """Forward pass through both Q networks"""
        obs_act = torch.cat([observations, actions], dim=-1)
        
        q1_value = self.q1_network(obs_act)
        q2_value = self.q2_network(obs_act)
        
        return q1_value, q2_value
    
    def q1(self, observations: torch.Tensor, actions: torch.Tensor):
        """Get Q1 value only"""
        obs_act = torch.cat([observations, actions], dim=-1)
        return self.q1_network(obs_act)


class SACAgent(BaseAgent):
    """Soft Actor-Critic Agent for Network Traffic Management"""
    
    def __init__(self, config: Config, observation_space, action_space):
        super().__init__(config, observation_space, action_space)
        
        # Get dimensions
        self.observation_dim = observation_space.shape[0]
        
        # For MultiDiscrete action space, treat as continuous and discretize
        if hasattr(action_space, 'nvec'):
            self.action_dim = len(action_space.nvec)
            self.max_flows = len(action_space.nvec)
            self.num_paths = action_space.nvec[0]
        else:
            self.action_dim = action_space.n
            self.max_flows = 50
            self.num_paths = 5
        
        # SAC hyperparameters
        self.gamma = config.rl.gamma
        self.tau = 0.005  # Soft update coefficient
        self.alpha = 0.2   # Entropy regularization coefficient
        self.target_update_interval = 1
        self.automatic_entropy_tuning = True
        
        # Initialize networks
        self.policy_net = SACPolicyNet(
            observation_dim=self.observation_dim,
            action_dim=self.max_flows,
            max_flows=self.max_flows,
            num_paths=self.num_paths
        ).to(self.device)
        
        self.critic = SACCritic(
            observation_dim=self.observation_dim,
            action_dim=self.max_flows
        ).to(self.device)
        
        self.critic_target = SACCritic(
            observation_dim=self.observation_dim,
            action_dim=self.max_flows
        ).to(self.device)
        
        # Copy parameters to target network
        self.hard_update(self.critic_target, self.critic)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.rl.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.rl.learning_rate
        )
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([self.max_flows])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.rl.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100000,  # Large buffer for off-policy learning
            observation_dim=self.observation_dim,
            action_dim=self.max_flows
        )
        
        self.update_counter = 0
        
        self.logger.info(f"Initialized SAC agent with {self.observation_dim}D observations, "
                        f"{self.max_flows} flows")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy"""
        obs_tensor = self.preprocess_observation(observation)
        
        with torch.no_grad():
            discrete_actions, _, _, _ = self.policy_net.sample(obs_tensor, deterministic)
        
        return discrete_actions.cpu().numpy().flatten()
    
    def store_transition(self, observation: np.ndarray, action: np.ndarray,
                        reward: float, next_observation: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.add(observation, action, reward, next_observation, done)
    
    def update(self, batch_size: int = None) -> Dict[str, float]:
        """Update SAC networks"""
        if batch_size is None:
            batch_size = self.config.rl.batch_size
        
        if not self.replay_buffer.is_ready(batch_size):
            return {}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device).unsqueeze(1)
        next_observations = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device).unsqueeze(1)
        
        # Update critic
        critic_loss = self.update_critic(observations, actions, rewards, next_observations, dones)
        
        # Update policy
        policy_loss, alpha_loss = self.update_policy(observations)
        
        # Update target networks
        if self.update_counter % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)
        
        self.update_counter += 1
        
        # Return training statistics
        stats = {
            'critic_loss': critic_loss,
            'policy_loss': policy_loss,
            'alpha': self.alpha,
            'learning_rate': self.policy_optimizer.param_groups[0]['lr']
        }
        
        if self.automatic_entropy_tuning:
            stats['alpha_loss'] = alpha_loss
        
        self.log_training_step(stats)
        return stats
    
    def update_critic(self, observations: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_observations: torch.Tensor,
                     dones: torch.Tensor) -> float:
        """Update critic networks"""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_action_probs, _, _ = self.policy_net.sample(next_observations)
            
            # Convert discrete actions to continuous for critic
            next_actions_continuous = next_action_probs
            
            # Compute target Q values
            next_q1, next_q2 = self.critic_target(next_observations, next_actions_continuous)
            next_q = torch.min(next_q1, next_q2)
            
            # Add entropy term
            next_log_probs = torch.log(next_action_probs + 1e-8).sum(dim=-1, keepdim=True)
            next_q = next_q - self.alpha * next_log_probs
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q values
        current_actions_continuous = F.one_hot(actions.long(), num_classes=self.num_paths).float()
        if current_actions_continuous.dim() == 3:
            current_actions_continuous = current_actions_continuous.view(current_actions_continuous.shape[0], -1)
        
        current_q1, current_q2 = self.critic(observations, current_actions_continuous)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_policy(self, observations: torch.Tensor) -> Tuple[float, float]:
        """Update policy network"""
        # Sample actions from current policy
        sampled_actions, action_probs, mean, log_std = self.policy_net.sample(observations)
        
        # Convert to continuous actions for critic
        actions_continuous = action_probs
        
        # Compute Q values
        q1, q2 = self.critic(observations, actions_continuous)
        q = torch.min(q1, q2)
        
        # Policy loss (maximize Q - alpha * entropy)
        log_probs = torch.log(action_probs + 1e-8).sum(dim=-1, keepdim=True)
        policy_loss = (self.alpha * log_probs - q).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update alpha (entropy regularization)
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        
        return policy_loss.item(), alpha_loss
    
    def soft_update(self, target, source, tau):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target, source):
        """Hard update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save_model(self, filepath: str):
        """Save the SAC model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'config': self.config,
            'total_steps': self.total_steps
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the SAC model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.alpha = checkpoint['alpha']
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_network_parameters(self) -> Dict[str, int]:
        """Get information about network parameters"""
        policy_params = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        
        return {
            'policy_parameters': policy_params,
            'critic_parameters': critic_params,
            'total_parameters': policy_params + critic_params
        }