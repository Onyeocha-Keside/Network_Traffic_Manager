"""
Evaluation script for trained Network Traffic Management agents
"""
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import time
from datetime import datetime

from config.config import Config
from environments.network_env import NetworkEnvironment
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from utils.logger import performance_logger
from utils.visualization import create_evaluation_plots, save_network_visualization


def load_trained_agent(model_path: str, config: Config, env: NetworkEnvironment):
    """Load a trained agent from file"""
    logger = logging.getLogger('experiments.evaluate')
    
    # Determine agent type from model path or config
    if 'ppo' in model_path.lower():
        agent = PPOAgent(config, env.observation_space, env.action_space)
    elif 'sac' in model_path.lower():
        agent = SACAgent(config, env.observation_space, env.action_space)
    else:
        # Try to determine from config
        if config.rl.algorithm.upper() == "PPO":
            agent = PPOAgent(config, env.observation_space, env.action_space)
        elif config.rl.algorithm.upper() == "SAC":
            agent = SACAgent(config, env.observation_space, env.action_space)
        else:
            raise ValueError(f"Cannot determine agent type from path: {model_path}")
    
    # Load the model
    agent.load_model(model_path)
    logger.info(f"Loaded {agent.__class__.__name__} from {model_path}")
    
    return agent


def detailed_episode_evaluation(agent, env: NetworkEnvironment, episode_length: int = 1000) -> Dict[str, Any]:
    """Run detailed evaluation of a single episode"""
    # Reset environment
    obs, info = env.reset()
    
    # Episode data collection
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'network_states': [],
        'flow_info': [],
        'routing_decisions': [],
        'performance_metrics': []
    }
    
    total_reward = 0
    step = 0
    
    for step in range(episode_length):
        # Store current state
        episode_data['observations'].append(obs.copy())
        network_state = env.get_network_state()
        episode_data['network_states'].append(network_state)
        
        # Get action from agent
        if isinstance(agent, PPOAgent):
            action, log_prob, value = agent.select_action(obs, deterministic=True)
        else:
            action = agent.select_action(obs, deterministic=True)
        
        episode_data['actions'].append(action.copy())
        
        # Take environment step
        next_obs, reward, done, truncated, info = env.step(action)
        
        episode_data['rewards'].append(reward)
        episode_data['routing_decisions'].append(env.get_routing_table())
        episode_data['performance_metrics'].append(env.get_performance_metrics())
        
        # Store flow information
        flow_info = {
            'active_flows': len(env.active_flows),
            'completed_flows': env.metrics['completed_flows'],
            'dropped_packets': env.metrics['dropped_packets']
        }
        episode_data['flow_info'].append(flow_info)
        
        total_reward += reward
        obs = next_obs
        
        if done or truncated:
            break
    
    episode_data['total_reward'] = total_reward
    episode_data['episode_length'] = step + 1
    
    return episode_data


def evaluate_agent(model_path: str, config: Config, num_episodes: int = 10, 
                  detailed: bool = False) -> Dict[str, Any]:
    """Comprehensive agent evaluation"""
    logger = logging.getLogger('experiments.evaluate')
    
    logger.info(f"Starting evaluation with {num_episodes} episodes...")
    performance_logger.start_timer('evaluation')
    
    # Create environment
    env = NetworkEnvironment(config)
    
    # Load agent
    agent = load_trained_agent(model_path, config, env)
    
    # Evaluation results
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'network_metrics': [],
        'detailed_episodes': [] if detailed else None,
        'agent_info': {
            'model_path': model_path,
            'agent_type': agent.__class__.__name__,
            'total_parameters': agent.get_network_parameters()['total_parameters'] if hasattr(agent, 'get_network_parameters') else 0
        },
        'environment_info': {
            'num_nodes': env.network.number_of_nodes(),
            'num_edges': env.network.number_of_edges(),
            'topology_type': config.network.topology_type
        }
    }
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        
        if detailed and episode < 3:  # Only detailed evaluation for first 3 episodes
            episode_data = detailed_episode_evaluation(agent, env, config.environment.max_episode_steps)
            results['detailed_episodes'].append(episode_data)
            
            episode_reward = episode_data['total_reward']
            episode_length = episode_data['episode_length']
            
        else:
            # Standard evaluation
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < config.environment.max_episode_steps:
                # Select action
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.select_action(obs, deterministic=True)
                else:
                    action = agent.select_action(obs, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
        
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        
        # Get final network metrics for this episode
        final_metrics = env.get_performance_metrics()
        results['network_metrics'].append(final_metrics)
        
        logger.info(f"Episode {episode + 1} - Reward: {episode_reward:.4f}, "
                   f"Length: {episode_length}, "
                   f"Avg Latency: {final_metrics['avg_latency']:.2f}ms")
    
    # Aggregate results
    results['summary'] = {
        'mean_reward': np.mean(results['episode_rewards']),
        'std_reward': np.std(results['episode_rewards']),
        'min_reward': np.min(results['episode_rewards']),
        'max_reward': np.max(results['episode_rewards']),
        'mean_length': np.mean(results['episode_lengths']),
        'std_length': np.std(results['episode_lengths']),
    }
    
    # Aggregate network metrics
    metric_keys = results['network_metrics'][0].keys()
    for key in metric_keys:
        values = [metrics[key] for metrics in results['network_metrics']]
        results['summary'][f'mean_{key}'] = np.mean(values)
        results['summary'][f'std_{key}'] = np.std(values)
    
    total_time = performance_logger.end_timer('evaluation')
    results['evaluation_time'] = total_time
    
    logger.info(f"Evaluation completed in {total_time:.2f}s")
    logger.info(f"Mean reward: {results['summary']['mean_reward']:.4f} ± {results['summary']['std_reward']:.4f}")
    logger.info(f"Mean latency: {results['summary']['mean_avg_latency']:.2f}ms")
    logger.info(f"Mean packet loss: {results['summary']['mean_packet_loss_rate']:.4f}")
    
    return results


def stress_test_agent(model_path: str, config: Config, stress_scenarios: List[Dict]) -> Dict[str, Any]:
    """Test agent under various stress conditions"""
    logger = logging.getLogger('experiments.evaluate')
    
    logger.info("Starting stress test evaluation...")
    
    # Create environment
    env = NetworkEnvironment(config)
    agent = load_trained_agent(model_path, config, env)
    
    stress_results = {}
    
    for scenario_name, scenario_config in stress_scenarios:
        logger.info(f"Testing scenario: {scenario_name}")
        
        # Apply scenario modifications to environment
        if 'failures' in scenario_config:
            env.set_network_failures(
                scenario_config['failures'].get('links', []),
                scenario_config['failures'].get('nodes', [])
            )
        
        if 'traffic_multiplier' in scenario_config:
            # Temporarily increase traffic generation rate
            original_rate = env.traffic_gen.config.flow_arrival_rate
            env.traffic_gen.config.flow_arrival_rate *= scenario_config['traffic_multiplier']
        
        # Run evaluation under stress
        scenario_results = []
        for episode in range(5):  # Fewer episodes for stress test
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < config.environment.max_episode_steps:
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.select_action(obs, deterministic=True)
                else:
                    action = agent.select_action(obs, deterministic=True)
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            scenario_results.append({
                'reward': episode_reward,
                'length': episode_length,
                'final_metrics': env.get_performance_metrics()
            })
        
        # Aggregate scenario results
        stress_results[scenario_name] = {
            'mean_reward': np.mean([r['reward'] for r in scenario_results]),
            'mean_latency': np.mean([r['final_metrics']['avg_latency'] for r in scenario_results]),
            'mean_packet_loss': np.mean([r['final_metrics']['packet_loss_rate'] for r in scenario_results]),
            'episodes': scenario_results
        }
        
        # Restore original configuration
        if 'traffic_multiplier' in scenario_config:
            env.traffic_gen.config.flow_arrival_rate = original_rate
        
        # Reset failures
        env.set_network_failures([], [])
        
        logger.info(f"Scenario {scenario_name} - Mean reward: {stress_results[scenario_name]['mean_reward']:.4f}")
    
    return stress_results


def compare_agent_policies(model_paths: List[str], config: Config, num_episodes: int = 5) -> Dict[str, Any]:
    """Compare multiple trained agents"""
    logger = logging.getLogger('experiments.evaluate')
    
    logger.info(f"Comparing {len(model_paths)} agents...")
    
    env = NetworkEnvironment(config)
    comparison_results = {}
    
    for model_path in model_paths:
        agent_name = Path(model_path).stem
        logger.info(f"Evaluating {agent_name}...")
        
        try:
            results = evaluate_agent(model_path, config, num_episodes, detailed=False)
            comparison_results[agent_name] = results['summary']
        except Exception as e:
            logger.error(f"Failed to evaluate {agent_name}: {e}")
            comparison_results[agent_name] = None
    
    return comparison_results


def analyze_routing_decisions(model_path: str, config: Config, analysis_steps: int = 1000) -> Dict[str, Any]:
    """Analyze agent's routing decision patterns"""
    logger = logging.getLogger('experiments.evaluate')
    
    logger.info("Analyzing routing decision patterns...")
    
    env = NetworkEnvironment(config)
    agent = load_trained_agent(model_path, config, env)
    
    # Collect routing decisions
    routing_analysis = {
        'path_usage_frequency': {},
        'flow_type_preferences': {},
        'adaptation_to_congestion': [],
        'decision_consistency': []
    }
    
    obs, info = env.reset()
    
    for step in range(analysis_steps):
        # Get current network state
        network_state = env.get_network_state()
        
        # Get agent's action
        if isinstance(agent, PPOAgent):
            action, _, _ = agent.select_action(obs, deterministic=True)
            action_probs = agent.get_action_probabilities(obs)
        else:
            action = agent.select_action(obs, deterministic=True)
            action_probs = None
        
        # Analyze routing decisions
        routing_table = env.get_routing_table()
        
        # Track path usage
        for flow_id, path in routing_table.items():
            if path:
                path_key = tuple(path)
                routing_analysis['path_usage_frequency'][path_key] = \
                    routing_analysis['path_usage_frequency'].get(path_key, 0) + 1
        
        # Track decision consistency (how often agent chooses same action in similar states)
        if action_probs is not None:
            max_prob = np.max(action_probs)
            routing_analysis['decision_consistency'].append(max_prob)
        
        # Take environment step
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, info = env.reset()
    
    # Process analysis results
    routing_analysis['most_used_paths'] = sorted(
        routing_analysis['path_usage_frequency'].items(),
        key=lambda x: x[1], reverse=True
    )[:10]
    
    if routing_analysis['decision_consistency']:
        routing_analysis['avg_decision_confidence'] = np.mean(routing_analysis['decision_consistency'])
    
    return routing_analysis


def save_evaluation_results(results: Dict[str, Any], output_dir: str, filename_prefix: str = "evaluation"):
    """Save evaluation results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = output_path / f"{filename_prefix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary report
    report_path = output_path / f"{filename_prefix}_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("NETWORK TRAFFIC MANAGER - EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {results['agent_info']['model_path']}\n")
        f.write(f"Agent: {results['agent_info']['agent_type']}\n")
        f.write(f"Parameters: {results['agent_info']['total_parameters']:,}\n")
        f.write(f"Network: {results['environment_info']['num_nodes']} nodes, {results['environment_info']['num_edges']} edges\n")
        f.write(f"Topology: {results['environment_info']['topology_type']}\n\n")
        
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 20 + "\n")
        summary = results['summary']
        f.write(f"Mean Reward: {summary['mean_reward']:.4f} ± {summary['std_reward']:.4f}\n")
        f.write(f"Mean Episode Length: {summary['mean_length']:.1f} ± {summary['std_length']:.1f}\n")
        f.write(f"Mean Latency: {summary['mean_avg_latency']:.2f}ms ± {summary['std_avg_latency']:.2f}ms\n")
        f.write(f"Mean Packet Loss: {summary['mean_packet_loss_rate']:.4f} ± {summary['std_packet_loss_rate']:.4f}\n")
        f.write(f"Mean Throughput: {summary['mean_throughput']:.4f} ± {summary['std_throughput']:.4f}\n")
        f.write(f"Mean Link Utilization: {summary['mean_avg_link_utilization']:.4f} ± {summary['std_avg_link_utilization']:.4f}\n")
        f.write(f"Fairness Index: {summary['mean_fairness_index']:.4f} ± {summary['std_fairness_index']:.4f}\n")
    
    return json_path, report_path


if __name__ == "__main__":
    # Example usage
    from config.config import get_config
    from utils.logger import setup_logging
    
    setup_logging()
    config = get_config()
    
    # Evaluate a trained model
    model_path = "data/models/best_ppo_model.zip"
    if Path(model_path).exists():
        results = evaluate_agent(model_path, config, num_episodes=10, detailed=True)
        
        # Save results
        json_path, report_path = save_evaluation_results(
            results, config.experiment.results_dir, "evaluation"
        )
        
        print(f"Evaluation completed!")
        print(f"Results saved to: {json_path}")
        print(f"Report saved to: {report_path}")
    else:
        print(f"Model not found: {model_path}")
        print("Please train a model first using: python main.py train")