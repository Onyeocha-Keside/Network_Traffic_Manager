"""
Baseline comparison experiments for Network Traffic Management
"""
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from config.config import Config
from environments.network_env import NetworkEnvironment
from baselines.ospf_routing import BaselineEvaluator, create_baseline_router
from experiments.evaluate import load_trained_agent
from utils.logger import performance_logger


def compare_all_baselines(config: Config, num_episodes: int = 10, 
                         include_rl_agent: bool = True, 
                         rl_model_path: Optional[str] = None) -> Dict[str, Dict]:
    """Compare all baseline algorithms and optionally RL agent"""
    logger = logging.getLogger('experiments.compare')
    
    logger.info("Starting comprehensive baseline comparison...")
    performance_logger.start_timer('baseline_comparison')
    
    # Create environment
    env = NetworkEnvironment(config)
    evaluator = BaselineEvaluator(config)
    
    # Define baseline algorithms to compare
    baseline_algorithms = ["OSPF", "ECMP", "Random", "Adaptive_OSPF"]
    
    # Evaluate baseline algorithms
    logger.info(f"Evaluating {len(baseline_algorithms)} baseline algorithms...")
    baseline_results = evaluator.compare_baselines(baseline_algorithms, env, num_episodes)
    
    all_results = baseline_results.copy()
    
    # Include RL agent if requested
    if include_rl_agent:
        # Try to find RL model
        if rl_model_path is None:
            # Look for best model in models directory
            models_dir = Path(config.experiment.models_dir)
            possible_models = [
                models_dir / f"best_{config.rl.algorithm.lower()}_model.zip",
                models_dir / f"final_{config.rl.algorithm.lower()}_model.zip"
            ]
            
            for model_path in possible_models:
                if model_path.exists():
                    rl_model_path = str(model_path)
                    break
        
        if rl_model_path and Path(rl_model_path).exists():
            logger.info(f"Evaluating RL agent: {rl_model_path}")
            try:
                rl_results = evaluate_rl_agent(rl_model_path, config, env, num_episodes)
                all_results[f"RL_{config.rl.algorithm}"] = rl_results
            except Exception as e:
                logger.error(f"Failed to evaluate RL agent: {e}")
        else:
            logger.warning("No RL model found for comparison")
    
    total_time = performance_logger.end_timer('baseline_comparison')
    
    # Create comparison summary
    comparison_summary = create_comparison_summary(all_results)
    
    logger.info(f"Baseline comparison completed in {total_time:.2f}s")
    log_comparison_results(all_results, logger)
    
    return all_results


def evaluate_rl_agent(model_path: str, config: Config, env: NetworkEnvironment, 
                     num_episodes: int) -> Dict[str, Any]:
    """Evaluate RL agent for comparison with baselines"""
    from experiments.evaluate import evaluate_agent
    
    # Use the existing evaluation function but extract only what we need for comparison
    full_results = evaluate_agent(model_path, config, num_episodes, detailed=False)
    
    # Extract summary metrics for comparison
    summary = full_results['summary']
    
    # Format to match baseline results structure
    comparison_results = {
        'algorithm': f"RL_{config.rl.algorithm}",
        'mean_reward': summary['mean_reward'],
        'std_reward': summary['std_reward'],
        'mean_length': summary['mean_length'],
        'std_length': summary['std_length'],
        'episode_rewards': full_results['episode_rewards'],
        'episode_lengths': full_results['episode_lengths']
    }
    
    # Add network metrics
    for key, value in summary.items():
        if key.startswith('mean_') and key not in comparison_results:
            comparison_results[key] = value
        elif key.startswith('std_') and key not in comparison_results:
            comparison_results[key] = value
    
    return comparison_results


def create_comparison_summary(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Create a summary comparison of all algorithms"""
    summary = {
        'best_algorithm': None,
        'best_reward': float('-inf'),
        'algorithm_rankings': {},
        'metric_winners': {},
        'statistical_significance': {}
    }
    
    # Find best algorithm by mean reward
    for algorithm, result in results.items():
        if result and result['mean_reward'] > summary['best_reward']:
            summary['best_reward'] = result['mean_reward']
            summary['best_algorithm'] = algorithm
    
    # Rank algorithms by different metrics
    metrics = ['mean_reward', 'mean_avg_latency', 'mean_packet_loss_rate', 'mean_throughput']
    
    for metric in metrics:
        if metric in ['mean_avg_latency', 'mean_packet_loss_rate']:
            # Lower is better
            ranking = sorted(
                [(alg, res[metric]) for alg, res in results.items() if res and metric in res],
                key=lambda x: x[1]
            )
        else:
            # Higher is better
            ranking = sorted(
                [(alg, res[metric]) for alg, res in results.items() if res and metric in res],
                key=lambda x: x[1], reverse=True
            )
        
        summary['algorithm_rankings'][metric] = ranking
        if ranking:
            summary['metric_winners'][metric] = ranking[0][0]
    
    return summary


def compare_under_stress(config: Config, stress_scenarios: List[Dict], 
                        num_episodes: int = 5) -> Dict[str, Dict]:
    """Compare algorithms under various stress conditions"""
    logger = logging.getLogger('experiments.compare')
    
    logger.info("Starting stress test comparison...")
    
    env = NetworkEnvironment(config)
    evaluator = BaselineEvaluator(config)
    
    algorithms = ["OSPF", "ECMP", "Adaptive_OSPF"]
    stress_results = {}
    
    for scenario_name, scenario_config in stress_scenarios:
        logger.info(f"Testing scenario: {scenario_name}")
        
        scenario_results = {}
        
        for algorithm in algorithms:
            try:
                # Apply stress conditions
                if 'failures' in scenario_config:
                    env.set_network_failures(
                        scenario_config['failures'].get('links', []),
                        scenario_config['failures'].get('nodes', [])
                    )
                
                if 'traffic_multiplier' in scenario_config:
                    original_rate = env.traffic_gen.config.flow_arrival_rate
                    env.traffic_gen.config.flow_arrival_rate *= scenario_config['traffic_multiplier']
                
                # Evaluate algorithm under stress
                result = evaluator.evaluate_baseline(algorithm, env, num_episodes)
                scenario_results[algorithm] = result
                
                # Restore original conditions
                if 'traffic_multiplier' in scenario_config:
                    env.traffic_gen.config.flow_arrival_rate = original_rate
                env.set_network_failures([], [])
                
            except Exception as e:
                logger.error(f"Failed to test {algorithm} under {scenario_name}: {e}")
                scenario_results[algorithm] = None
        
        stress_results[scenario_name] = scenario_results
    
    return stress_results


def compare_scalability(config: Config, node_counts: List[int], 
                       num_episodes: int = 5) -> Dict[str, Dict]:
    """Compare algorithm performance across different network sizes"""
    logger = logging.getLogger('experiments.compare')
    
    logger.info("Starting scalability comparison...")
    
    scalability_results = {}
    
    for num_nodes in node_counts:
        logger.info(f"Testing with {num_nodes} nodes...")
        
        # Create config for this network size
        scale_config = Config()
        scale_config.update(**config.save_to_dict())
        scale_config.network.num_nodes = num_nodes
        
        # Evaluate algorithms
        try:
            results = compare_all_baselines(scale_config, num_episodes, include_rl_agent=False)
            scalability_results[f"{num_nodes}_nodes"] = results
        except Exception as e:
            logger.error(f"Failed scalability test for {num_nodes} nodes: {e}")
            scalability_results[f"{num_nodes}_nodes"] = None
    
    return scalability_results


def compare_with_statistical_tests(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Perform statistical significance tests between algorithms"""
    from scipy import stats
    
    statistical_results = {}
    
    algorithms = list(results.keys())
    
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms[i+1:], i+1):
            if results[alg1] and results[alg2]:
                rewards1 = results[alg1]['episode_rewards']
                rewards2 = results[alg2]['episode_rewards']
                
                # Perform t-test
                try:
                    t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
                    
                    statistical_results[f"{alg1}_vs_{alg2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'better_algorithm': alg1 if np.mean(rewards1) > np.mean(rewards2) else alg2
                    }
                except Exception as e:
                    statistical_results[f"{alg1}_vs_{alg2}"] = {
                        'error': str(e)
                    }
    
    return statistical_results


def log_comparison_results(results: Dict[str, Dict], logger):
    """Log comparison results in a formatted way"""
    logger.info("=" * 60)
    logger.info("BASELINE COMPARISON RESULTS")
    logger.info("=" * 60)
    
    # Sort by mean reward
    sorted_results = sorted(
        [(alg, res) for alg, res in results.items() if res],
        key=lambda x: x[1]['mean_reward'], reverse=True
    )
    
    logger.info(f"{'Algorithm':<15} {'Mean Reward':<12} {'Std':<8} {'Latency(ms)':<12} {'Packet Loss':<12}")
    logger.info("-" * 70)
    
    for algorithm, result in sorted_results:
        logger.info(
            f"{algorithm:<15} "
            f"{result['mean_reward']:<12.4f} "
            f"{result['std_reward']:<8.4f} "
            f"{result.get('mean_avg_latency', 0):<12.2f} "
            f"{result.get('mean_packet_loss_rate', 0):<12.4f}"
        )
    
    if sorted_results:
        best_algorithm, best_result = sorted_results[0]
        logger.info("=" * 60)
        logger.info(f"BEST ALGORITHM: {best_algorithm}")
        logger.info(f"Best Reward: {best_result['mean_reward']:.4f} ± {best_result['std_reward']:.4f}")
        logger.info("=" * 60)


def save_comparison_results(results: Dict[str, Any], config: Config, 
                           filename_prefix: str = "baseline_comparison"):
    """Save comparison results to files"""
    output_dir = Path(config.experiment.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = output_dir / f"{filename_prefix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create comparison report
    report_path = output_dir / f"{filename_prefix}_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("NETWORK TRAFFIC MANAGER - BASELINE COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Algorithm comparison table
        f.write("ALGORITHM PERFORMANCE COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Algorithm':<15} {'Mean Reward':<12} {'Std Dev':<10} {'Latency':<10} {'Loss Rate':<10}\n")
        f.write("-" * 65 + "\n")
        
        # Sort by mean reward
        sorted_results = sorted(
            [(alg, res) for alg, res in results.items() if res and 'mean_reward' in res],
            key=lambda x: x[1]['mean_reward'], reverse=True
        )
        
        for algorithm, result in sorted_results:
            f.write(
                f"{algorithm:<15} "
                f"{result['mean_reward']:<12.4f} "
                f"{result['std_reward']:<10.4f} "
                f"{result.get('mean_avg_latency', 0):<10.2f} "
                f"{result.get('mean_packet_loss_rate', 0):<10.4f}\n"
            )
        
        if sorted_results:
            f.write(f"\nBEST ALGORITHM: {sorted_results[0][0]}\n")
            f.write(f"Best Performance: {sorted_results[0][1]['mean_reward']:.4f}\n")
    
    return json_path, report_path


if __name__ == "__main__":
    # Example usage
    from config.config import get_config
    from utils.logger import setup_logging
    
    setup_logging()
    config = get_config()
    
    # Compare all baselines
    results = compare_all_baselines(config, num_episodes=10)
    
    # Save results
    json_path, report_path = save_comparison_results(results, config)
    
    print("Baseline comparison completed!")
    print(f"Results saved to: {json_path}")
    print(f"Report saved to: {report_path}")
    
    # Print summary
    if results:
        best_algorithm = max(results.items(), key=lambda x: x[1]['mean_reward'] if x[1] else float('-inf'))
        print(f"\nBest Algorithm: {best_algorithm[0]}")
        print(f"Best Reward: {best_algorithm[1]['mean_reward']:.4f}")