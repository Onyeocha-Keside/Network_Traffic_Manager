"""
OSPF (Open Shortest Path First) Baseline Routing Algorithm
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging

from config.config import Config


class OSPFRouter:
    """OSPF routing algorithm implementation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # OSPF state
        self.network: Optional[nx.Graph] = None
        self.routing_table: Dict[Tuple[int, int], List[int]] = {}
        self.link_costs: Dict[Tuple[int, int], float] = {}
        
        # Performance tracking
        self.decisions_made = 0
        self.total_reward = 0.0
        
    def initialize(self, network: nx.Graph):
        """Initialize OSPF with network topology"""
        self.network = network.copy()
        self._compute_link_costs()
        self._compute_routing_table()
        
        self.logger.info(f"OSPF initialized for network with {self.network.number_of_nodes()} nodes")
    
    def _compute_link_costs(self):
        """Compute link costs based on latency and capacity"""
        self.link_costs = {}
        
        for u, v, data in self.network.edges(data=True):
            # OSPF typically uses inverse bandwidth as cost
            # We'll use a combination of latency and inverse bandwidth
            bandwidth = data.get('bandwidth', 100.0)  # Mbps
            latency = data.get('latency', 10.0)  # ms
            
            # Cost formula: latency + (reference_bandwidth / bandwidth)
            reference_bandwidth = 1000.0  # 1 Gbps reference
            cost = latency + (reference_bandwidth / bandwidth)
            
            self.link_costs[(u, v)] = cost
            self.link_costs[(v, u)] = cost  # Symmetric links
    
    def _compute_routing_table(self):
        """Compute shortest paths for all node pairs"""
        self.routing_table = {}
        
        # Update edge weights in network
        for (u, v), cost in self.link_costs.items():
            if self.network.has_edge(u, v):
                self.network[u][v]['weight'] = cost
        
        # Compute shortest paths for all pairs
        for source in self.network.nodes():
            for destination in self.network.nodes():
                if source != destination:
                    try:
                        path = nx.shortest_path(
                            self.network, source, destination, weight='weight'
                        )
                        self.routing_table[(source, destination)] = path
                    except nx.NetworkXNoPath:
                        self.routing_table[(source, destination)] = []
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get routing action based on OSPF algorithm"""
        # OSPF doesn't use observations directly - it uses pre-computed paths
        # For RL environment compatibility, we need to return actions for active flows
        
        # This is a simplified approach - in practice, we'd need to extract
        # flow information from the observation
        max_flows = getattr(self.config.environment, 'max_concurrent_flows', 50)
        num_paths = getattr(self.config.environment, 'num_paths', 5)
        
        # Default action: select path 0 (shortest path) for all flows
        actions = np.zeros(max_flows, dtype=np.int32)
        
        self.decisions_made += 1
        return actions
    
    def update_link_costs(self, link_utilizations: Dict[Tuple[int, int], float]):
        """Update link costs based on current utilization (adaptive OSPF)"""
        for (u, v), utilization in link_utilizations.items():
            if (u, v) in self.link_costs:
                # Increase cost based on utilization
                base_cost = self.link_costs[(u, v)]
                
                # Exponential penalty for high utilization
                utilization_penalty = np.exp(5 * utilization) - 1
                new_cost = base_cost * (1 + utilization_penalty)
                
                if self.network.has_edge(u, v):
                    self.network[u][v]['weight'] = new_cost
        
        # Recompute routing table with updated costs
        self._compute_routing_table()
    
    def handle_link_failure(self, failed_links: List[Tuple[int, int]]):
        """Handle link failures by recomputing routes"""
        # Remove failed links from network
        for u, v in failed_links:
            if self.network.has_edge(u, v):
                self.network.remove_edge(u, v)
        
        # Recompute routing table
        self._compute_routing_table()
        
        self.logger.info(f"OSPF updated routing due to {len(failed_links)} link failures")
    
    def get_path_for_flow(self, source: int, destination: int, 
                         path_index: int = 0) -> List[int]:
        """Get specific path for a flow"""
        if (source, destination) in self.routing_table:
            return self.routing_table[(source, destination)]
        else:
            return []
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing algorithm statistics"""
        return {
            'algorithm': 'OSPF',
            'decisions_made': self.decisions_made,
            'total_reward': self.total_reward,
            'routing_table_size': len(self.routing_table),
            'avg_path_length': np.mean([
                len(path) for path in self.routing_table.values() if path
            ]) if self.routing_table else 0
        }


class ECMPRouter:
    """Equal-Cost Multi-Path (ECMP) routing algorithm"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.network: Optional[nx.Graph] = None
        self.routing_table: Dict[Tuple[int, int], List[List[int]]] = {}
        self.flow_assignments: Dict[int, int] = {}  # flow_id -> path_index
        self.decisions_made = 0
        self.total_reward = 0.0
    
    def initialize(self, network: nx.Graph):
        """Initialize ECMP with network topology"""
        self.network = network.copy()
        self._compute_equal_cost_paths()
        
        self.logger.info(f"ECMP initialized for network with {self.network.number_of_nodes()} nodes")
    
    def _compute_equal_cost_paths(self):
        """Compute equal-cost paths for all node pairs"""
        self.routing_table = {}
        
        # Set edge weights based on latency
        for u, v, data in self.network.edges(data=True):
            latency = data.get('latency', 10.0)
            self.network[u][v]['weight'] = latency
        
        # Find equal-cost paths
        for source in self.network.nodes():
            for destination in self.network.nodes():
                if source != destination:
                    try:
                        # Get all shortest paths
                        paths = list(nx.all_shortest_paths(
                            self.network, source, destination, weight='weight'
                        ))
                        # Limit number of paths
                        max_paths = getattr(self.config.environment, 'num_paths', 5)
                        self.routing_table[(source, destination)] = paths[:max_paths]
                    except nx.NetworkXNoPath:
                        self.routing_table[(source, destination)] = []
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get routing action using ECMP load balancing"""
        max_flows = getattr(self.config.environment, 'max_concurrent_flows', 50)
        num_paths = getattr(self.config.environment, 'num_paths', 5)
        
        actions = np.zeros(max_flows, dtype=np.int32)
        
        # Simple round-robin assignment for load balancing
        for flow_idx in range(max_flows):
            # Assign paths in round-robin fashion
            path_index = flow_idx % num_paths
            actions[flow_idx] = path_index
        
        self.decisions_made += 1
        return actions
    
    def get_path_for_flow(self, source: int, destination: int, 
                         path_index: int = 0) -> List[int]:
        """Get specific path for a flow"""
        if (source, destination) in self.routing_table:
            paths = self.routing_table[(source, destination)]
            if paths and path_index < len(paths):
                return paths[path_index]
        return []
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing algorithm statistics"""
        avg_paths_per_destination = np.mean([
            len(paths) for paths in self.routing_table.values()
        ]) if self.routing_table else 0
        
        return {
            'algorithm': 'ECMP',
            'decisions_made': self.decisions_made,
            'total_reward': self.total_reward,
            'routing_table_size': len(self.routing_table),
            'avg_paths_per_destination': avg_paths_per_destination
        }


class RandomRouter:
    """Random routing baseline for comparison"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.network: Optional[nx.Graph] = None
        self.decisions_made = 0
        self.total_reward = 0.0
        np.random.seed(42)  # For reproducibility
    
    def initialize(self, network: nx.Graph):
        """Initialize random router"""
        self.network = network.copy()
        self.logger.info(f"Random router initialized for network with {self.network.number_of_nodes()} nodes")
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get random routing actions"""
        max_flows = getattr(self.config.environment, 'max_concurrent_flows', 50)
        num_paths = getattr(self.config.environment, 'num_paths', 5)
        
        # Random path selection for each flow
        actions = np.random.randint(0, num_paths, size=max_flows)
        
        self.decisions_made += 1
        return actions
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing algorithm statistics"""
        return {
            'algorithm': 'Random',
            'decisions_made': self.decisions_made,
            'total_reward': self.total_reward
        }


class AdaptiveOSPF(OSPFRouter):
    """Adaptive OSPF that updates costs based on congestion"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.update_frequency = 100  # Update costs every N decisions
        self.last_update = 0
        
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action with adaptive cost updates"""
        # Extract link utilizations from observation (simplified)
        # In practice, this would require proper observation parsing
        
        # Update costs periodically
        if self.decisions_made - self.last_update >= self.update_frequency:
            # Simulate link utilizations (in real implementation, extract from observation)
            simulated_utilizations = {}
            for u, v in self.network.edges():
                utilization = np.random.beta(2, 5)  # Typical network utilization distribution
                simulated_utilizations[(u, v)] = utilization
                simulated_utilizations[(v, u)] = utilization
            
            self.update_link_costs(simulated_utilizations)
            self.last_update = self.decisions_made
        
        return super().get_action(observation)


def create_baseline_router(algorithm: str, config: Config):
    """Factory function to create baseline routing algorithms"""
    algorithm = algorithm.upper()
    
    if algorithm == "OSPF":
        return OSPFRouter(config)
    elif algorithm == "ECMP":
        return ECMPRouter(config)
    elif algorithm == "RANDOM":
        return RandomRouter(config)
    elif algorithm == "ADAPTIVE_OSPF":
        return AdaptiveOSPF(config)
    else:
        raise ValueError(f"Unknown baseline algorithm: {algorithm}")


class BaselineEvaluator:
    """Evaluator for baseline routing algorithms"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_baseline(self, algorithm: str, env, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate a baseline algorithm"""
        router = create_baseline_router(algorithm, self.config)
        router.initialize(env.network)
        
        episode_rewards = []
        episode_lengths = []
        network_metrics = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < self.config.environment.max_episode_steps:
                # Get action from baseline algorithm
                action = router.get_action(obs)
                
                # Take environment step
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Get network metrics
            metrics = env.get_performance_metrics()
            network_metrics.append(metrics)
            
            router.total_reward += episode_reward
        
        # Aggregate results
        results = {
            'algorithm': algorithm,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        # Add network metrics
        metric_keys = network_metrics[0].keys()
        for key in metric_keys:
            values = [metrics[key] for metrics in network_metrics]
            results[f'mean_{key}'] = np.mean(values)
            results[f'std_{key}'] = np.std(values)
        
        # Add algorithm-specific statistics
        results['algorithm_stats'] = router.get_routing_statistics()
        
        self.logger.info(f"{algorithm} evaluation - Mean reward: {results['mean_reward']:.4f}")
        
        return results
    
    def compare_baselines(self, algorithms: List[str], env, num_episodes: int = 10) -> Dict[str, Dict]:
        """Compare multiple baseline algorithms"""
        self.logger.info(f"Comparing {len(algorithms)} baseline algorithms...")
        
        results = {}
        for algorithm in algorithms:
            try:
                results[algorithm] = self.evaluate_baseline(algorithm, env, num_episodes)
            except Exception as e:
                self.logger.error(f"Failed to evaluate {algorithm}: {e}")
                results[algorithm] = None
        
        return results


if __name__ == "__main__":
    # Example usage
    from config.config import get_config
    from environments.network_env import NetworkEnvironment
    
    config = get_config()
    env = NetworkEnvironment(config)
    evaluator = BaselineEvaluator(config)
    
    # Test baseline algorithms
    algorithms = ["OSPF", "ECMP", "Random"]
    results = evaluator.compare_baselines(algorithms, env, num_episodes=5)
    
    print("Baseline Comparison Results:")
    for algorithm, result in results.items():
        if result:
            print(f"{algorithm}: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
        else:
            print(f"{algorithm}: Failed")