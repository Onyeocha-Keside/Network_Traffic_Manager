"""
Configuration management for Network Traffic Manager
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


@dataclass
class NetworkConfig:
    """Network topology configuration"""
    # Topology parameters
    num_nodes: int = 10
    topology_type: str = "mesh"  # mesh, ring, tree, fat_tree, custom
    connectivity: float = 0.6  # For random topologies
    
    # Link parameters
    min_bandwidth: float = 10.0  # Mbps
    max_bandwidth: float = 1000.0  # Mbps
    min_latency: float = 1.0  # ms
    max_latency: float = 50.0  # ms
    
    # Failure parameters
    link_failure_prob: float = 0.001  # Per time step
    node_failure_prob: float = 0.0001
    recovery_time_range: Tuple[int, int] = (10, 100)  # Time steps


@dataclass
class TrafficConfig:
    """Traffic generation configuration"""
    # Flow parameters
    flow_arrival_rate: float = 5.0  # flows per time step
    flow_size_mean: float = 1.0  # MB
    flow_size_alpha: float = 1.16  # Pareto shape parameter
    flow_duration_mean: float = 30.0  # time steps
    
    # Traffic patterns
    diurnal_pattern: bool = True
    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 16, 19, 20, 21])
    peak_multiplier: float = 3.0
    
    # Traffic types
    traffic_types: Dict[str, float] = field(default_factory=lambda: {
        "web": 0.4,
        "video": 0.3,
        "file_transfer": 0.2,
        "real_time": 0.1
    })
    
    # QoS requirements per traffic type
    qos_requirements: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "web": {"max_latency": 100, "min_bandwidth": 1, "priority": 2},
        "video": {"max_latency": 50, "min_bandwidth": 5, "priority": 3},
        "file_transfer": {"max_latency": 1000, "min_bandwidth": 10, "priority": 1},
        "real_time": {"max_latency": 20, "min_bandwidth": 0.5, "priority": 4}
    })


@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    # Algorithm selection
    algorithm: str = "PPO"  # PPO, SAC, A2C
    
    # Training parameters
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048  # For PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Network architecture
    policy_network: List[int] = field(default_factory=lambda: [256, 256])
    value_network: List[int] = field(default_factory=lambda: [256, 256])
    
    # Exploration
    ent_coef: float = 0.01
    
    # Evaluation
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    n_eval_episodes: int = 10
    
    # Checkpointing
    save_freq: int = 50000
    
    # Multi-agent settings
    multi_agent: bool = False
    shared_policy: bool = True


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    # Time parameters
    time_step_duration: float = 1.0  # seconds per step
    max_episode_steps: int = 1000
    max_concurrent_flows: int = 50  # Add this field
    
    # State space
    observation_window: int = 10  # Historical observations
    normalize_observations: bool = True
    
    # Action space
    action_type: str = "discrete"  # discrete, continuous, multi_discrete
    num_paths: int = 5  # Max alternative paths to consider
    
    # Reward function weights (adjusted for better scaling)
    latency_weight: float = -0.1      # Reduced from -1.0
    throughput_weight: float = 1.0    # Increased from 0.5
    utilization_weight: float = 0.5   # Increased from 0.3
    fairness_weight: float = 0.2
    drop_weight: float = -1.0         # Reduced from -10.0
    
    # Environment dynamics
    dynamic_topology: bool = True
    traffic_variation: bool = True


@dataclass
class ExperimentConfig:
    """Experiment and evaluation configuration"""
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_tensorboard: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Comparison baselines
    baselines: List[str] = field(default_factory=lambda: ["OSPF", "ECMP", "Random"])
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        "avg_latency", "throughput", "packet_loss", "link_utilization", 
        "fairness_index", "convergence_time"
    ])
    
    # Visualization
    create_plots: bool = True
    save_animations: bool = False
    plot_format: str = "png"  # png, pdf, svg
    
    # Output directories
    results_dir: str = "data/results"
    models_dir: str = "data/models"
    logs_dir: str = "data/logs"


@dataclass
class Config:
    """Main configuration class"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Create necessary directories"""
        for dir_path in [
            self.experiment.results_dir,
            self.experiment.models_dir,
            self.experiment.logs_dir,
            "data/topologies",
            "data/traffic_patterns"
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_from_dict(cls, config_dict: Dict) -> 'Config':
        """Load configuration from dictionary"""
        return cls(
            network=NetworkConfig(**config_dict.get('network', {})),
            traffic=TrafficConfig(**config_dict.get('traffic', {})),
            rl=RLConfig(**config_dict.get('rl', {})),
            environment=EnvironmentConfig(**config_dict.get('environment', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def save_to_dict(self) -> Dict:
        """Save configuration to dictionary"""
        return {
            'network': self.network.__dict__,
            'traffic': self.traffic.__dict__,
            'rl': self.rl.__dict__,
            'environment': self.environment.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    config_section = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(config_section, sub_key):
                            setattr(config_section, sub_key, sub_value)
                else:
                    setattr(self, key, value)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs):
    """Update the global configuration"""
    global config
    config.update(**kwargs)


# Environment-specific configurations
TOPOLOGY_CONFIGS = {
    "small_mesh": {
        "network": {"num_nodes": 6, "topology_type": "mesh", "connectivity": 0.8}
    },
    "medium_tree": {
        "network": {"num_nodes": 15, "topology_type": "tree"}
    },
    "large_fat_tree": {
        "network": {"num_nodes": 32, "topology_type": "fat_tree"}
    }
}

TRAFFIC_CONFIGS = {
    "light_load": {
        "traffic": {"flow_arrival_rate": 2.0, "peak_multiplier": 2.0}
    },
    "heavy_load": {
        "traffic": {"flow_arrival_rate": 10.0, "peak_multiplier": 5.0}
    },
    "bursty": {
        "traffic": {"flow_arrival_rate": 3.0, "peak_multiplier": 8.0}
    }
}

TRAINING_CONFIGS = {
    "quick_test": {
        "rl": {"total_timesteps": 100000, "eval_freq": 5000}
    },
    "full_training": {
        "rl": {"total_timesteps": 2000000, "eval_freq": 20000}
    },
    "hyperparameter_search": {
        "rl": {"total_timesteps": 500000, "eval_freq": 10000}
    }
}