"""
Main entry point for Network Traffic Manager
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config import Config, get_config, TOPOLOGY_CONFIGS, TRAFFIC_CONFIGS, TRAINING_CONFIGS
from utils.logger import setup_logging
from experiments.train import train_agent
from experiments.evaluate import evaluate_agent
from experiments.compare_baselines import compare_all_baselines
from utils.visualization import create_training_dashboard


def setup_experiment(args):
    """Setup experiment configuration based on arguments"""
    config = get_config()
    
    # Apply predefined configurations if specified
    if args.topology_config and args.topology_config in TOPOLOGY_CONFIGS:
        config.update(**TOPOLOGY_CONFIGS[args.topology_config])
        print(f"Applied topology config: {args.topology_config}")
    
    if args.traffic_config and args.traffic_config in TRAFFIC_CONFIGS:
        config.update(**TRAFFIC_CONFIGS[args.traffic_config])
        print(f"Applied traffic config: {args.traffic_config}")
    
    if args.training_config and args.training_config in TRAINING_CONFIGS:
        config.update(**TRAINING_CONFIGS[args.training_config])
        print(f"Applied training config: {args.training_config}")
    
    # Override with command line arguments
    if args.algorithm:
        config.rl.algorithm = args.algorithm
    
    if args.timesteps:
        config.rl.total_timesteps = args.timesteps
    
    if args.nodes:
        config.network.num_nodes = args.nodes
    
    if args.topology:
        config.network.topology_type = args.topology
    
    return config


def train_command(args):
    """Execute training command"""
    config = setup_experiment(args)
    
    print("=" * 60)
    print("NETWORK TRAFFIC MANAGER - TRAINING")
    print("=" * 60)
    print(f"Algorithm: {config.rl.algorithm}")
    print(f"Topology: {config.network.topology_type} ({config.network.num_nodes} nodes)")
    print(f"Training steps: {config.rl.total_timesteps:,}")
    print(f"Output directory: {config.experiment.models_dir}")
    print("=" * 60)
    
    # Train the agent
    trained_model, training_stats = train_agent(config)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {config.experiment.models_dir}")
    
    # Create training visualization if requested
    if config.experiment.create_plots:
        try:
            from utils.visualization import create_training_dashboard
            dashboard_path = create_training_dashboard(training_stats, config)
            print(f"Training dashboard saved to: {dashboard_path}")
        except Exception as e:
            print(f"Warning: Could not create training dashboard: {e}")
    
    return trained_model


def evaluate_command(args):
    """Execute evaluation command"""
    config = setup_experiment(args)
    
    print("=" * 60)
    print("NETWORK TRAFFIC MANAGER - EVALUATION")
    print("=" * 60)
    
    if args.model_path:
        model_path = args.model_path
    else:
        # Use latest model
        model_path = Path(config.experiment.models_dir) / f"best_{config.rl.algorithm.lower()}_model.zip"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first or specify a valid model path.")
        return
    
    print(f"Evaluating model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)
    
    # Evaluate the agent
    results = evaluate_agent(model_path, config, num_episodes=args.episodes)
    
    print("\nEvaluation Results:")
    print("=" * 40)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    return results


def compare_command(args):
    """Execute baseline comparison command"""
    config = setup_experiment(args)
    
    print("=" * 60)
    print("NETWORK TRAFFIC MANAGER - BASELINE COMPARISON")
    print("=" * 60)
    print(f"Baselines: {', '.join(config.experiment.baselines)}")
    print(f"Episodes per baseline: {args.episodes}")
    print("=" * 60)
    
    # Compare all baselines
    comparison_results = compare_all_baselines(config, num_episodes=args.episodes)
    
    print("\nComparison Results:")
    print("=" * 50)
    
    # Print results in a nice table format
    metrics = list(comparison_results[list(comparison_results.keys())[0]].keys())
    
    # Header
    print(f"{'Method':<15}", end="")
    for metric in metrics:
        print(f"{metric:<12}", end="")
    print()
    print("-" * (15 + 12 * len(metrics)))
    
    # Results
    for method, results in comparison_results.items():
        print(f"{method:<15}", end="")
        for metric in metrics:
            value = results[metric]
            if isinstance(value, float):
                print(f"{value:<12.4f}", end="")
            else:
                print(f"{value:<12}", end="")
        print()
    
    return comparison_results


def demo_command(args):
    """Execute demo command"""
    config = setup_experiment(args)
    
    print("=" * 60)
    print("NETWORK TRAFFIC MANAGER - DEMO")
    print("=" * 60)
    print("Running quick demonstration...")
    
    # Quick training
    config.rl.total_timesteps = 50000
    config.rl.eval_freq = 10000
    config.environment.max_episode_steps = 200
    
    print("Training agent (quick demo)...")
    trained_model, _ = train_agent(config)
    
    print("Evaluating agent...")
    results = evaluate_agent(trained_model, config, num_episodes=5)
    
    print("Comparing with baselines...")
    comparison_results = compare_all_baselines(config, num_episodes=3)
    
    print("\nDemo completed! Check the results above.")
    return results, comparison_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Network Traffic Manager - Deep Reinforcement Learning for Network Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a PPO agent on a mesh topology
  python main.py train --algorithm PPO --topology mesh --nodes 10

  # Quick demo
  python main.py demo

  # Evaluate trained model
  python main.py evaluate --episodes 20

  # Compare all baselines
  python main.py compare --episodes 10

  # Use predefined configurations
  python main.py train --topology-config large_fat_tree --traffic-config heavy_load
        """
    )
    
    # Common arguments
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Configuration override arguments
    parser.add_argument('--algorithm', choices=['PPO', 'SAC', 'A2C'],
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int,
                       help='Number of training timesteps')
    parser.add_argument('--nodes', type=int,
                       help='Number of network nodes')
    parser.add_argument('--topology', 
                       choices=['mesh', 'ring', 'tree', 'fat_tree', 'datacenter'],
                       help='Network topology type')
    
    # Predefined configuration arguments
    parser.add_argument('--topology-config', 
                       choices=list(TOPOLOGY_CONFIGS.keys()),
                       help='Use predefined topology configuration')
    parser.add_argument('--traffic-config',
                       choices=list(TRAFFIC_CONFIGS.keys()),
                       help='Use predefined traffic configuration')
    parser.add_argument('--training-config',
                       choices=list(TRAINING_CONFIGS.keys()),
                       help='Use predefined training configuration')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new agent')
    train_parser.add_argument('--resume', type=str,
                             help='Resume training from checkpoint')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained agent')
    eval_parser.add_argument('--model-path', type=str,
                            help='Path to trained model (uses latest if not specified)')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare with baselines')
    compare_parser.add_argument('--episodes', type=int, default=5,
                               help='Number of episodes per baseline')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demonstration')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Set random seed
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    try:
        # Execute command
        if args.command == 'train':
            train_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'compare':
            compare_command(args)
        elif args.command == 'demo':
            demo_command(args)
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())