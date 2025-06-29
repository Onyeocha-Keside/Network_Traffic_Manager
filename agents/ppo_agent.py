"""
Proximal Policy Optimization (PPO) Agent for Network Traffic Management
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Any, Tuple, Optional
import logging

from agents.base_agent import BaseAgent, NetworkPolicyNet, NetworkValueFunction, RolloutBuffer
from config.config import Config


class PPOPolicyNet(NetworkPolicyNet):
    """PPO-specific policy network with proper action distribution"""
    
    def __init__(self, observation_dim: int, action_dim: int, 
                 max_flows: int = 50, num_paths: int = 5):
        super().__init__(observation_dim, action_dim, max_flows, num_paths)
        
        # For PPO, we need to output logits for categorical distributions
        # Each flow gets a categorical distribution over possible paths
        
    def get_action_distribution(self, observations: torch.Tensor):
        """Get action distribution for PPO"""
        logits = self.forward(observations)
        batch_size = logits.shape[0]
        
        # Reshape logits to (batch_size, max_flows, num_paths)
        logits = logits.view(batch_size, self.max_flows, self.num_paths)
        
        # Create categorical distributions for each flow
        distributions = []
        for flow_idx in range(self.max_flows):
            flow_logits = logits[:, flow_idx, :]
            dist = Categorical(logits=flow_logits)
            distributions.append(dist)
        
        return distributions
    
    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update"""
        distributions = self.get_action_distribution(observations)
        
        # Convert actions to individual flow actions
        # actions shape: (batch_size, max_flows)
        batch_size = actions.shape[0]
        actions = actions.view(batch_size, self.max_flows).long()
        
        log_probs = []
        entropy = []
        
        for flow_idx in range(self.max_flows):
            flow_actions = actions[:, flow_idx]
            dist = distributions[flow_idx]
            
            log_prob = dist.log_prob(flow_actions)
            flow_entropy = dist.entropy()
            
            log_probs.append(log_prob)
            entropy.append(flow_entropy)
        
        # Sum log probabilities and entropy across flows
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        total_entropy = torch.stack(entropy, dim=1).mean(dim=1)
        
        return total_log_prob, total_entropy


class PPOAgent(BaseAgent):
    """PPO Agent for Network Traffic Management"""
    
    def __init__(self, config: Config, observation_space, action_space):
        super().__init__(config, observation_space, action_space)
        
        # Get dimensions
        self.observation_dim = observation_space.shape[0]
        
        # For MultiDiscrete action space, get total dimension
        if hasattr(action_space, 'nvec'):
            self.action_dim = len(action_space.nvec)
            self.max_flows = len(action_space.nvec)
            self.num_paths = action_space.nvec[0]  # Assuming all flows have same number of paths
        else:
            self.action_dim = action_space.n
            self.max_flows = 50  # Default
            self.num_paths = 5   # Default
        
        # Initialize networks
        self.policy_net = PPOPolicyNet(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            max_flows=self.max_flows,
            num_paths=self.num_paths
        ).to(self.device)
        
        self.value_net = NetworkValueFunction(
            observation_dim=self.observation_dim,
            hidden_dims=config.rl.value_network
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=config.rl.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), 
            lr=config.rl.learning_rate
        )
        
        # PPO hyperparameters
        self.gamma = config.rl.gamma
        self.gae_lambda = config.rl.gae_lambda
        self.clip_epsilon = 0.2
        self.entropy_coeff = config.rl.ent_coef
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            capacity=config.rl.n_steps,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )
        
        self.logger.info(f"Initialized PPO agent with {self.observation_dim}D observations, "
                        f"{self.max_flows} flows, {self.num_paths} paths per flow")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy"""
        obs_tensor = self.preprocess_observation(observation)
        
        with torch.no_grad():
            # Get action distributions
            distributions = self.policy_net.get_action_distribution(obs_tensor)
            
            # Get value estimate
            value = self.value_net(obs_tensor)
            
            # Sample actions for each flow
            actions = []
            log_probs = []
            
            for dist in distributions:
                if deterministic:
                    action = torch.argmax(dist.probs, dim=-1)
                else:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action)
                
                actions.append(action)
                log_probs.append(log_prob)
            
            # Stack actions and log_probs
            actions = torch.stack(actions, dim=1)  # (batch_size, max_flows)
            log_probs = torch.stack(log_probs, dim=1).sum(dim=1)  # Sum across flows
        
        return (
            actions.cpu().numpy().flatten(),
            log_probs.cpu().item(),
            value.cpu().item()
        )
    
    def store_transition(self, observation: np.ndarray, action: np.ndarray, 
                        reward: float, value: float, log_prob: float, done: bool):
        """Store transition in rollout buffer"""
        self.buffer.add(observation, action, reward, value, log_prob, done)
    
    def update(self, last_observation: np.ndarray) -> Dict[str, float]:
        """Update PPO policy and value function"""
        # Get last value for GAE computation
        with torch.no_grad():
            last_obs_tensor = self.preprocess_observation(last_observation)
            last_value = self.value_net(last_obs_tensor).cpu().item()
        
        # Compute advantages and returns
        self.buffer.compute_advantages_and_returns(
            last_value, self.gamma, self.gae_lambda
        )
        
        # Training statistics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0
        
        # PPO update epochs
        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batch(self.config.rl.batch_size):
                # Convert to tensors
                obs = torch.FloatTensor(batch['observations']).to(self.device)
                actions = torch.LongTensor(batch['actions']).to(self.device)
                old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                returns = torch.FloatTensor(batch['returns']).to(self.device)
                old_values = torch.FloatTensor(batch['values']).to(self.device)
                
                # Normalize advantages with clipping
                if torch.std(advantages) > 0:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = torch.clamp(advantages, -10.0, 10.0)  # Clip extreme values
                
                # Evaluate current policy
                new_log_probs, entropy = self.policy_net.evaluate_actions(obs, actions)
                values = self.value_net(obs)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                values_clipped = old_values + torch.clamp(values - old_values, -0.2, 0.2)
                value_loss1 = F.mse_loss(values, returns)
                value_loss2 = F.mse_loss(values_clipped, returns)
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Clip value loss to prevent explosion
                value_loss = torch.clamp(value_loss, 0.0, 1000.0)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coeff * value_loss + 
                             self.entropy_coeff * entropy_loss)
                
                # Policy update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1
        
        # Clear buffer for next rollout
        self.buffer.clear()
        
        # Compute explained variance with safety checks
        explained_var = 0.0
        if len(self.buffer.returns) > 1:
            try:
                all_returns = torch.FloatTensor(self.buffer.returns).to(self.device)
                # Only compute if we have valid data
                if len(all_returns) == len(obs):
                    predicted_values = self.value_net(obs)
                    explained_var = self.compute_explained_variance(all_returns, predicted_values).item()
                else:
                    explained_var = 0.0
            except Exception as e:
                self.logger.warning(f"Error computing explained variance: {e}")
                explained_var = 0.0
        
        # Return training statistics
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / num_updates,
            'explained_variance': explained_var,
            'learning_rate': self.policy_optimizer.param_groups[0]['lr']
        }
        
        # Log statistics
        self.log_training_step(stats)
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the PPO model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the PPO model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities for analysis"""
        obs_tensor = self.preprocess_observation(observation)
        
        with torch.no_grad():
            distributions = self.policy_net.get_action_distribution(obs_tensor)
            
            # Get probabilities for each flow
            all_probs = []
            for dist in distributions:
                probs = dist.probs.cpu().numpy()
                all_probs.append(probs)
            
            return np.array(all_probs)
    
    def evaluate_policy(self, observation: np.ndarray) -> Dict[str, float]:
        """Evaluate current policy on given observation"""
        obs_tensor = self.preprocess_observation(observation)
        
        with torch.no_grad():
            # Get value estimate
            value = self.value_net(obs_tensor).cpu().item()
            
            # Get action distributions
            distributions = self.policy_net.get_action_distribution(obs_tensor)
            
            # Compute average entropy across flows
            total_entropy = 0.0
            for dist in distributions:
                total_entropy += dist.entropy().cpu().item()
            avg_entropy = total_entropy / len(distributions)
            
            # Get action probabilities statistics
            max_probs = []
            for dist in distributions:
                max_prob = torch.max(dist.probs).cpu().item()
                max_probs.append(max_prob)
            
            avg_max_prob = np.mean(max_probs)
        
        return {
            'value_estimate': value,
            'average_entropy': avg_entropy,
            'average_max_prob': avg_max_prob,
            'confidence': 1.0 - avg_entropy / np.log(self.num_paths)  # Normalized confidence
        }
    
    def set_learning_rate(self, lr: float):
        """Set learning rate for both optimizers"""
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = lr
        
        self.logger.info(f"Learning rate set to {lr}")
    
    def get_network_parameters(self) -> Dict[str, int]:
        """Get information about network parameters"""
        policy_params = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        value_params = sum(p.numel() for p in self.value_net.parameters() if p.requires_grad)
        
        return {
            'policy_parameters': policy_params,
            'value_parameters': value_params,
            'total_parameters': policy_params + value_params
        }