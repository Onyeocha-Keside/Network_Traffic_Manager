"""
Training script for Network Traffic Management RL agents
"""
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import time
from datetime import datetime
import json

from config.config import Config
from environments.network_env import NetworkEnvironment
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from utils.logger import TrainingLogger, performance_logger


def create_agent(config: Config, env: NetworkEnvironment):
    """Create RL agent based on configuration"""
    if config.rl.algorithm.upper() == "PPO":
        return PPOAgent(config, env.observation_space, env.action_space)
    elif config.rl.algorithm.upper() == "SAC":
        return SACAgent(config, env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unsupported algorithm: {config.rl.algorithm}")


def collect_rollout_ppo(agent: PPOAgent, env: NetworkEnvironment, n_steps: int) -> Dict[str, Any]:
    """Collect rollout data for PPO training"""
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    dones = []
    
    obs, info = env.reset()
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    for step in range(n_steps):
        # Select action
        action, log_prob, value = agent.select_action(obs, deterministic=False)
        
        # Store transition data
        agent.store_transition(obs, action, 0, value, log_prob, False)  # Reward will be updated
        
        # Take environment step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Update stored reward
        agent.buffer.rewards[agent.buffer.position - 1] = reward
        agent.buffer.dones[agent.buffer.position - 1] = done or truncated
        
        current_episode_reward += reward
        current_episode_length += 1
        
        # Episode finished
        if done or truncated:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0
            current_episode_length = 0
            obs, info = env.reset()
        else:
            obs = next_obs
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_observation': obs
    }


def collect_transitions_sac(agent: SACAgent, env: NetworkEnvironment, n_steps: int) -> Dict[str, Any]:
    """Collect transitions for SAC training"""
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    obs, info = env.reset()
    
    for step in range(n_steps):
        # Select action
        action = agent.select_action(obs, deterministic=False)
        
        # Take environment step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done or truncated)
        
        current_episode_reward += reward
        current_episode_length += 1
        
        # Episode finished
        if done or truncated:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0
            current_episode_length = 0
            obs, info = env.reset()
        else:
            obs = next_obs
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def evaluate_agent(agent, env: NetworkEnvironment, n_episodes: int = 10) -> Dict[str, float]:
    """Evaluate agent performance"""
    episode_rewards = []
    episode_lengths = []
    network_metrics = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < env.config.environment.max_episode_steps:
            # Select action deterministically
            if hasattr(agent, 'select_action'):
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.select_action(obs, deterministic=True)
                else:
                    action = agent.select_action(obs, deterministic=True)
            else:
                # For baseline agents
                action = agent.get_action(obs)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Get network performance metrics
        metrics = env.get_performance_metrics()
        network_metrics.append(metrics)
    
    # Aggregate metrics
    avg_metrics = {}
    if network_metrics:
        for key in network_metrics[0].keys():
            values = [m[key] for m in network_metrics if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        **avg_metrics
    }


def train_agent(config: Config) -> Tuple[Any, Dict[str, Any]]:
    """Main training function"""
    logger = logging.getLogger('experiments.train')
    training_logger = TrainingLogger(f"{config.rl.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    logger.info("Starting training...")
    performance_logger.start_timer('total_training')
    
    # Create environment
    logger.info("Creating environment...")
    env = NetworkEnvironment(config)
    logger.info(f"Environment created with {env.network.number_of_nodes()} nodes, "
               f"{env.network.number_of_edges()} edges")
    
    # Create agent
    logger.info(f"Creating {config.rl.algorithm} agent...")
    agent = create_agent(config, env)
    
    # Log network parameters
    if hasattr(agent, 'get_network_parameters'):
        params_info = agent.get_network_parameters()
        logger.info(f"Agent created with {params_info['total_parameters']:,} parameters")
    
    # Training statistics
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'evaluation_rewards': [],
        'evaluation_metrics': [],
        'training_losses': {},
        'timestamps': []
    }
    
    # Training loop
    logger.info("Starting training loop...")
    total_steps = 0
    episode_count = 0
    best_reward = float('-inf')
    
    while total_steps < config.rl.total_timesteps:
        performance_logger.start_timer('rollout_collection')
        
        if config.rl.algorithm.upper() == "PPO":
            # PPO: Collect rollout and update
            rollout_data = collect_rollout_ppo(agent, env, config.rl.n_steps)
            
            # Update agent
            performance_logger.start_timer('agent_update')
            update_stats = agent.update(rollout_data['final_observation'])
            performance_logger.end_timer('agent_update', log_result=False)
            
            # Log episode statistics
            if rollout_data['episode_rewards']:
                for reward, length in zip(rollout_data['episode_rewards'], rollout_data['episode_lengths']):
                    training_logger.log_episode(episode_count, reward, length)
                    training_stats['episode_rewards'].append(reward)
                    training_stats['episode_lengths'].append(length)
                    episode_count += 1
            
            total_steps += config.rl.n_steps
            
        elif config.rl.algorithm.upper() == "SAC":
            # SAC: Collect transitions and update frequently
            transitions_data = collect_transitions_sac(agent, env, 1000)  # Collect 1000 steps
            
            # Update agent multiple times
            performance_logger.start_timer('agent_update')
            update_stats = {}
            for _ in range(1000):  # Update once per step collected
                if len(agent.replay_buffer) >= config.rl.batch_size:
                    step_stats = agent.update()
                    for key, value in step_stats.items():
                        if key not in update_stats:
                            update_stats[key] = []
                        update_stats[key].append(value)
            
            # Average update statistics
            update_stats = {key: np.mean(values) for key, values in update_stats.items()}
            performance_logger.end_timer('agent_update', log_result=False)
            
            # Log episode statistics
            if transitions_data['episode_rewards']:
                for reward, length in zip(transitions_data['episode_rewards'], transitions_data['episode_lengths']):
                    training_logger.log_episode(episode_count, reward, length)
                    training_stats['episode_rewards'].append(reward)
                    training_stats['episode_lengths'].append(length)
                    episode_count += 1
            
            total_steps += 1000
        
        performance_logger.end_timer('rollout_collection', log_result=False)
        
        # Log training statistics
        for key, value in update_stats.items():
            if key not in training_stats['training_losses']:
                training_stats['training_losses'][key] = []
            training_stats['training_losses'][key].append(value)
            training_logger.log_metric(key, value, total_steps)
        
        # Evaluation
        if total_steps % config.rl.eval_freq == 0:
            logger.info(f"Evaluating at step {total_steps:,}...")
            performance_logger.start_timer('evaluation')
            
            eval_results = evaluate_agent(agent, env, config.rl.n_eval_episodes)
            performance_logger.end_timer('evaluation', log_result=False)
            
            # Log evaluation results
            mean_reward = eval_results['mean_reward']
            logger.info(f"Evaluation - Mean reward: {mean_reward:.4f}, "
                       f"Mean length: {eval_results['mean_length']:.1f}")
            
            training_stats['evaluation_rewards'].append(mean_reward)
            training_stats['evaluation_metrics'].append(eval_results)
            training_stats['timestamps'].append(total_steps)
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_model_path = Path(config.experiment.models_dir) / f"best_{config.rl.algorithm.lower()}_model.zip"
                agent.save_model(str(best_model_path))
                logger.info(f"New best model saved with reward {best_reward:.4f}")
        
        # Save checkpoint
        if total_steps % config.rl.save_freq == 0:
            checkpoint_path = Path(config.experiment.models_dir) / f"{config.rl.algorithm.lower()}_checkpoint_{total_steps}.zip"
            agent.save_model(str(checkpoint_path))
            logger.info(f"Checkpoint saved at step {total_steps:,}")
        
        # Progress update
        if total_steps % 10000 == 0:
            elapsed_time = performance_logger.end_timer('total_training', log_result=False)
            performance_logger.start_timer('total_training')
            
            progress = total_steps / config.rl.total_timesteps * 100
            recent_rewards = training_stats['episode_rewards'][-10:] if training_stats['episode_rewards'] else [0]
            recent_avg = np.mean(recent_rewards)
            
            logger.info(f"Progress: {progress:.1f}% ({total_steps:,}/{config.rl.total_timesteps:,}) - "
                       f"Recent avg reward: {recent_avg:.4f} - "
                       f"Episodes: {episode_count} - "
                       f"Time: {elapsed_time:.1f}s")
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_eval_results = evaluate_agent(agent, env, config.rl.n_eval_episodes * 2)
    
    # Save final model
    final_model_path = Path(config.experiment.models_dir) / f"final_{config.rl.algorithm.lower()}_model.zip"
    agent.save_model(str(final_model_path))
    
    # Save training statistics
    stats_path = Path(config.experiment.results_dir) / f"training_stats_{config.rl.algorithm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    training_stats['final_evaluation'] = final_eval_results
    training_stats['config'] = config.save_to_dict()
    training_stats['total_training_time'] = performance_logger.end_timer('total_training')
    
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2, default=str)
    
    logger.info(f"Training completed! Final evaluation reward: {final_eval_results['mean_reward']:.4f}")
    logger.info(f"Best model: {best_model_path}")
    logger.info(f"Training statistics saved: {stats_path}")
    
    return agent, training_stats


def resume_training(checkpoint_path: str, config: Config, additional_steps: int = None):
    """Resume training from checkpoint"""
    logger = logging.getLogger('experiments.train')
    
    # Create environment and agent
    env = NetworkEnvironment(config)
    agent = create_agent(config, env)
    
    # Load checkpoint
    agent.load_model(checkpoint_path)
    logger.info(f"Resumed training from {checkpoint_path}")
    
    # Update training steps if specified
    if additional_steps:
        config.rl.total_timesteps = agent.total_steps + additional_steps
    
    # Continue training
    return train_agent(config)


def train_with_curriculum(config: Config, curriculum_stages: list):
    """Train agent with curriculum learning"""
    logger = logging.getLogger('experiments.train')
    
    logger.info("Starting curriculum training...")
    
    # Create environment and agent
    env = NetworkEnvironment(config)
    agent = create_agent(config, env)
    
    total_training_stats = {
        'stage_results': [],
        'cumulative_steps': 0
    }
    
    for stage_idx, stage_config in enumerate(curriculum_stages):
        logger.info(f"Starting curriculum stage {stage_idx + 1}/{len(curriculum_stages)}")
        
        # Update configuration for this stage
        stage_config_obj = Config()
        stage_config_obj.update(**stage_config)
        
        # Update environment with new config
        env = NetworkEnvironment(stage_config_obj)
        
        # Train for this stage
        stage_agent, stage_stats = train_agent(stage_config_obj)
        
        # Transfer learned policy to next stage
        if stage_idx < len(curriculum_stages) - 1:
            # Save intermediate model
            stage_model_path = Path(config.experiment.models_dir) / f"curriculum_stage_{stage_idx + 1}_model.zip"
            stage_agent.save_model(str(stage_model_path))
            
            # Load into next agent (this would be implemented based on specific needs)
            agent = stage_agent
        
        total_training_stats['stage_results'].append(stage_stats)
        total_training_stats['cumulative_steps'] += stage_config.get('rl', {}).get('total_timesteps', 100000)
    
    logger.info("Curriculum training completed!")
    return agent, total_training_stats


def hyperparameter_search(base_config: Config, param_grid: Dict[str, list], n_trials: int = 10):
    """Perform hyperparameter search"""
    logger = logging.getLogger('experiments.train')
    
    logger.info(f"Starting hyperparameter search with {n_trials} trials...")
    
    results = []
    
    for trial in range(n_trials):
        logger.info(f"Trial {trial + 1}/{n_trials}")
        
        # Sample hyperparameters
        trial_config = Config()
        trial_config.update(**base_config.save_to_dict())
        
        for param_path, values in param_grid.items():
            # Parse parameter path (e.g., "rl.learning_rate")
            parts = param_path.split('.')
            current = trial_config
            for part in parts[:-1]:
                current = getattr(current, part)
            
            # Sample value
            sampled_value = np.random.choice(values)
            setattr(current, parts[-1], sampled_value)
            
            logger.info(f"  {param_path}: {sampled_value}")
        
        # Train with sampled parameters
        try:
            agent, stats = train_agent(trial_config)
            
            # Get final performance
            final_reward = stats['final_evaluation']['mean_reward']
            
            # Store results
            trial_result = {
                'trial': trial,
                'config': trial_config.save_to_dict(),
                'final_reward': final_reward,
                'training_stats': stats
            }
            results.append(trial_result)
            
            logger.info(f"Trial {trial + 1} completed with final reward: {final_reward:.4f}")
            
        except Exception as e:
            logger.error(f"Trial {trial + 1} failed: {e}")
            continue
    
    # Find best configuration
    if results:
        best_trial = max(results, key=lambda x: x['final_reward'])
        logger.info(f"Best configuration found with reward: {best_trial['final_reward']:.4f}")
        
        # Save results
        search_results_path = Path(base_config.experiment.results_dir) / f"hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(search_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Hyperparameter search results saved: {search_results_path}")
        
        return best_trial, results
    else:
        logger.error("No successful trials in hyperparameter search")
        return None, []


if __name__ == "__main__":
    # Example usage
    from config.config import get_config
    from utils.logger import setup_logging
    
    setup_logging()
    config = get_config()
    
    # Train agent
    trained_agent, training_statistics = train_agent(config)
    
    print("Training completed successfully!")
    print(f"Final reward: {training_statistics['final_evaluation']['mean_reward']:.4f}")