"""
Logging utilities for Network Traffic Manager
"""
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

from config.config import get_config


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    # Create logs directory
    config = get_config()
    logs_dir = Path(config.experiment.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"network_traffic_manager_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Setup specific loggers
    setup_module_loggers(level)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")


def setup_module_loggers(level: int):
    """Setup module-specific loggers"""
    
    # Environment logger
    env_logger = logging.getLogger('environments')
    env_logger.setLevel(level)
    
    # Agent logger
    agent_logger = logging.getLogger('agents')
    agent_logger.setLevel(level)
    
    # Training logger
    training_logger = logging.getLogger('experiments')
    training_logger.setLevel(level)
    
    # Visualization logger
    viz_logger = logging.getLogger('utils.visualization')
    viz_logger.setLevel(level)
    
    # Suppress some verbose third-party loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


class TrainingLogger:
    """Logger specifically for training metrics"""
    
    def __init__(self, experiment_name: str):
        self.logger = logging.getLogger(f'experiments.{experiment_name}')
        self.metrics = {}
        
    def log_metric(self, name: str, value: float, step: int):
        """Log a training metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append((step, value))
        self.logger.info(f"Step {step:6d} - {name}: {value:.6f}")
    
    def log_episode(self, episode: int, reward: float, length: int, **kwargs):
        """Log episode information"""
        msg = f"Episode {episode:4d} - Reward: {reward:8.2f}, Length: {length:4d}"
        
        for key, value in kwargs.items():
            if isinstance(value, float):
                msg += f", {key}: {value:.4f}"
            else:
                msg += f", {key}: {value}"
        
        self.logger.info(msg)
    
    def get_metrics(self):
        """Get all logged metrics"""
        return self.metrics.copy()
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        import json
        
        # Convert to JSON-serializable format
        serializable_metrics = {}
        for name, values in self.metrics.items():
            serializable_metrics[name] = {
                'steps': [step for step, _ in values],
                'values': [value for _, value in values]
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {filepath}")


class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
        self.timers = {}
        
    def start_timer(self, name: str):
        """Start a performance timer"""
        import time
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_result: bool = True):
        """End a performance timer and optionally log the result"""
        import time
        
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return None
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        if log_result:
            self.logger.info(f"{name}: {elapsed:.4f} seconds")
        
        return elapsed
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory usage: {memory_mb:.1f} MB")
            return memory_mb
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return None


def log_config(config):
    """Log the current configuration"""
    logger = logging.getLogger('config')
    
    logger.info("=" * 50)
    logger.info("CONFIGURATION")
    logger.info("=" * 50)
    
    # Network configuration
    logger.info("Network Configuration:")
    logger.info(f"  Nodes: {config.network.num_nodes}")
    logger.info(f"  Topology: {config.network.topology_type}")
    logger.info(f"  Connectivity: {config.network.connectivity}")
    logger.info(f"  Bandwidth range: {config.network.min_bandwidth}-{config.network.max_bandwidth} Mbps")
    
    # Traffic configuration
    logger.info("Traffic Configuration:")
    logger.info(f"  Arrival rate: {config.traffic.flow_arrival_rate}")
    logger.info(f"  Peak multiplier: {config.traffic.peak_multiplier}")
    logger.info(f"  Traffic types: {config.traffic.traffic_types}")
    
    # RL configuration
    logger.info("RL Configuration:")
    logger.info(f"  Algorithm: {config.rl.algorithm}")
    logger.info(f"  Total timesteps: {config.rl.total_timesteps:,}")
    logger.info(f"  Learning rate: {config.rl.learning_rate}")
    logger.info(f"  Batch size: {config.rl.batch_size}")
    
    # Environment configuration
    logger.info("Environment Configuration:")
    logger.info(f"  Max episode steps: {config.environment.max_episode_steps}")
    logger.info(f"  Action type: {config.environment.action_type}")
    logger.info(f"  Reward weights: latency={config.environment.latency_weight}, "
               f"throughput={config.environment.throughput_weight}")
    
    logger.info("=" * 50)


class ExperimentLogger:
    """High-level experiment logger"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(f'experiment.{experiment_name}')
        self.start_time = None
        self.results = {}
        
    def start_experiment(self):
        """Mark the start of an experiment"""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting experiment: {self.experiment_name}")
    
    def end_experiment(self):
        """Mark the end of an experiment"""
        import time
        
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.info(f"Experiment completed in {duration:.2f} seconds")
        else:
            self.logger.info("Experiment completed")
    
    def log_result(self, key: str, value):
        """Log an experiment result"""
        self.results[key] = value
        self.logger.info(f"Result - {key}: {value}")
    
    def log_comparison(self, results_dict: dict):
        """Log comparison results"""
        self.logger.info("Comparison Results:")
        self.logger.info("-" * 40)
        
        for method, metrics in results_dict.items():
            self.logger.info(f"{method}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {metric}: {value:.4f}")
                else:
                    self.logger.info(f"  {metric}: {value}")
    
    def save_experiment_summary(self, filepath: str):
        """Save experiment summary to file"""
        import json
        from datetime import datetime
        
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'duration': time.time() - self.start_time if self.start_time else None,
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Experiment summary saved to {filepath}")


# Global performance logger instance
performance_logger = PerformanceLogger()