"""
Network Traffic Management Environment for Reinforcement Learning
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import logging

from config.config import Config
from environments.topology_generator import TopologyGenerator
from environments.traffic_generator import TrafficGenerator


@dataclass
class NetworkFlow:
    """Represents a network flow"""
    flow_id: int
    source: int
    destination: int
    size: float  # MB
    remaining_size: float  # MB
    bandwidth_requirement: float  # Mbps
    max_latency: float  # ms
    priority: int
    traffic_type: str
    start_time: int
    current_path: Optional[List[int]] = None
    packets_sent: int = 0
    packets_dropped: int = 0
    total_latency: float = 0.0


@dataclass
class LinkState:
    """Represents the state of a network link"""
    bandwidth: float  # Mbps
    latency: float  # ms
    utilization: float  # 0-1
    queue_length: int
    max_queue_length: int = 100
    failed: bool = False
    failure_time: Optional[int] = None
    recovery_time: Optional[int] = None


class NetworkEnvironment(gym.Env):
    """
    Network Traffic Management Environment
    
    State Space:
        - Link utilizations (num_links,)
        - Queue lengths at nodes (num_nodes,)
        - Active flow information (num_flows, flow_features)
        - Network topology status
        - Historical performance metrics
    
    Action Space:
        - Route selection for each active flow
        - Load balancing decisions
        - Priority adjustments
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.topology_gen = TopologyGenerator(config.network)
        self.traffic_gen = TrafficGenerator(config.traffic)
        
        # Network state
        self.network: Optional[nx.Graph] = None
        self.links: Dict[Tuple[int, int], LinkState] = {}
        self.nodes: Dict[int, Dict] = {}
        self.active_flows: Dict[int, NetworkFlow] = {}
        
        # Environment state
        self.current_step = 0
        self.episode_step = 0
        self.total_flows_generated = 0
        
        # Performance tracking
        self.metrics = {
            'total_latency': 0.0,
            'total_packets': 0,
            'dropped_packets': 0,
            'completed_flows': 0,
            'link_utilizations': [],
            'queue_lengths': []
        }
        
        # Initialize environment
        self._initialize_network()
        self._setup_spaces()
        
    def _initialize_network(self):
        """Initialize the network topology and states"""
        # Generate topology
        self.network = self.topology_gen.generate_topology()
        
        # Initialize link states
        self.links = {}
        for u, v, data in self.network.edges(data=True):
            self.links[(u, v)] = LinkState(
                bandwidth=data['bandwidth'],
                latency=data['latency'],
                utilization=0.0,
                queue_length=0
            )
            # Add reverse link for undirected graph
            self.links[(v, u)] = LinkState(
                bandwidth=data['bandwidth'],
                latency=data['latency'],
                utilization=0.0,
                queue_length=0
            )
        
        # Initialize node states
        self.nodes = {
            node: {
                'queue_length': 0,
                'processing_delay': np.random.uniform(0.1, 2.0),
                'failed': False
            }
            for node in self.network.nodes()
        }
        
        self.logger.info(f"Initialized network with {self.network.number_of_nodes()} nodes "
                        f"and {self.network.number_of_edges()} edges")
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Calculate dimensions
        num_nodes = self.network.number_of_nodes()
        num_links = len(self.links) // 2  # Undirected links
        max_flows = 50  # Maximum concurrent flows to track
        
        # Observation space components
        obs_dim = (
            num_links +  # Link utilizations
            num_nodes +  # Node queue lengths
            max_flows * 6 +  # Flow features (src, dst, size, remaining, priority, age)
            num_nodes +  # Node failure status  
            num_links +  # Link failure status
            10  # Historical metrics (moving averages)
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: route selection for each flow
        # For discrete actions: select one of k-shortest paths for each flow
        self.max_paths_per_flow = self.config.environment.num_paths
        self.action_space = spaces.MultiDiscrete([
            self.max_paths_per_flow for _ in range(max_flows)
        ])
        
        self.max_concurrent_flows = max_flows
        
        self.logger.info(f"Observation space: {self.observation_space.shape}")
        self.logger.info(f"Action space: {self.action_space}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset network state
        self.current_step = 0
        self.episode_step = 0
        self.active_flows = {}
        self.total_flows_generated = 0
        
        # Reset metrics
        self.metrics = {
            'total_latency': 0.0,
            'total_packets': 0,
            'dropped_packets': 0,
            'completed_flows': 0,
            'link_utilizations': [],
            'queue_lengths': []
        }
        
        # Reset link and node states
        for link_state in self.links.values():
            link_state.utilization = 0.0
            link_state.queue_length = 0
            link_state.failed = False
            link_state.failure_time = None
            link_state.recovery_time = None
        
        for node_state in self.nodes.values():
            node_state['queue_length'] = 0
            node_state['failed'] = False
        
        # Generate initial traffic if needed
        if self.config.environment.traffic_variation:
            self._generate_new_flows()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        self.episode_step += 1
        
        # Apply actions (routing decisions)
        self._apply_actions(action)
        
        # Update network dynamics
        self._update_network_dynamics()
        
        # Generate new flows
        self._generate_new_flows()
        
        # Process existing flows
        self._process_flows()
        
        # Handle failures and recovery
        self._handle_failures()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.episode_step >= self.config.environment.max_episode_steps
        truncated = False
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _apply_actions(self, actions):
        """Apply routing actions to active flows"""
        flow_ids = list(self.active_flows.keys())
        flows_to_remove = []
        
        for i, action in enumerate(actions[:len(flow_ids)]):
            if i >= len(flow_ids):
                break
                
            flow_id = flow_ids[i]
            if flow_id not in self.active_flows:  # Flow might have been removed
                continue
                
            flow = self.active_flows[flow_id]
            
            # Validate that source and destination exist in network
            if (flow.source not in self.network.nodes() or 
                flow.destination not in self.network.nodes()):
                # Invalid flow, mark for removal
                flows_to_remove.append(flow_id)
                continue
            
            # Get k-shortest paths for this flow
            try:
                paths = list(nx.shortest_simple_paths(
                    self.network, flow.source, flow.destination, weight='latency'
                ))[:self.max_paths_per_flow]
                
                if paths and action < len(paths):
                    flow.current_path = paths[action]
                else:
                    # Fallback to shortest path
                    flow.current_path = nx.shortest_path(
                        self.network, flow.source, flow.destination, weight='latency'
                    )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # No path available or invalid nodes, mark for dropping
                flow.current_path = None
        
        # Remove invalid flows
        for flow_id in flows_to_remove:
            if flow_id in self.active_flows:
                del self.active_flows[flow_id]
    
    def _update_network_dynamics(self):
        """Update network state based on current traffic"""
        # Reset utilizations
        for link_state in self.links.values():
            link_state.utilization = 0.0
            link_state.queue_length = max(0, link_state.queue_length - 1)  # Process queue
        
        # Calculate link utilizations based on active flows
        for flow in self.active_flows.values():
            if flow.current_path and len(flow.current_path) > 1:
                bandwidth_per_hop = flow.bandwidth_requirement / len(flow.current_path)
                
                for i in range(len(flow.current_path) - 1):
                    u, v = flow.current_path[i], flow.current_path[i + 1]
                    if (u, v) in self.links:
                        link = self.links[(u, v)]
                        if not link.failed:
                            link.utilization += bandwidth_per_hop / link.bandwidth
                            link.utilization = min(1.0, link.utilization)  # Cap at 100%
    
    def _generate_new_flows(self):
        """Generate new traffic flows"""
        # Get available nodes from the actual network
        available_nodes = list(self.network.nodes())
        new_flows = self.traffic_gen.generate_flows(self.current_step, available_nodes)
        
        for flow_data in new_flows:
            if len(self.active_flows) < self.max_concurrent_flows:
                # Validate that source and destination exist in network
                if (flow_data['source'] in available_nodes and 
                    flow_data['destination'] in available_nodes and
                    flow_data['source'] != flow_data['destination']):
                    
                    flow = NetworkFlow(
                        flow_id=self.total_flows_generated,
                        source=flow_data['source'],
                        destination=flow_data['destination'],
                        size=flow_data['size'],
                        remaining_size=flow_data['size'],
                        bandwidth_requirement=flow_data['bandwidth_requirement'],
                        max_latency=flow_data['max_latency'],
                        priority=flow_data['priority'],
                        traffic_type=flow_data['traffic_type'],
                        start_time=self.current_step
                    )
                    
                    self.active_flows[flow.flow_id] = flow
                    self.total_flows_generated += 1
    
    def _process_flows(self):
        """Process active flows and update their state"""
        completed_flows = []
        
        for flow_id, flow in list(self.active_flows.items()):  # Create a copy of items
            if flow.current_path is None:
                # No path available, drop the flow
                flow.packets_dropped += 1
                self.metrics['dropped_packets'] += 1
                completed_flows.append(flow_id)
                continue
            
            # Calculate transmission progress
            if len(flow.current_path) > 1:
                # Simplified: assume we can transmit some data each step
                transmission_rate = self._calculate_transmission_rate(flow)
                transmitted = min(flow.remaining_size, transmission_rate)
                
                flow.remaining_size -= transmitted
                flow.packets_sent += 1
                
                # Calculate latency for this packet
                path_latency = self._calculate_path_latency(flow.current_path)
                flow.total_latency += path_latency
                
                self.metrics['total_latency'] += path_latency
                self.metrics['total_packets'] += 1
                
                # Check if flow is completed
                if flow.remaining_size <= 0:
                    completed_flows.append(flow_id)
                    self.metrics['completed_flows'] += 1
                
                # Check if flow exceeded max latency (simplified)
                avg_latency = flow.total_latency / max(1, flow.packets_sent)
                if avg_latency > flow.max_latency:
                    # Consider as dropped due to QoS violation
                    flow.packets_dropped += 1
                    self.metrics['dropped_packets'] += 1
                    completed_flows.append(flow_id)
        
        # Remove completed flows
        for flow_id in completed_flows:
            if flow_id in self.active_flows:  # Check if still exists
                del self.active_flows[flow_id]
    
    def _calculate_transmission_rate(self, flow: NetworkFlow) -> float:
        """Calculate how much data can be transmitted for a flow"""
        if not flow.current_path or len(flow.current_path) < 2:
            return 0.0
        
        # Find bottleneck link in the path
        min_available_bandwidth = float('inf')
        
        for i in range(len(flow.current_path) - 1):
            u, v = flow.current_path[i], flow.current_path[i + 1]
            if (u, v) in self.links:
                link = self.links[(u, v)]
                if link.failed:
                    return 0.0  # Path is broken
                
                available_bw = link.bandwidth * (1 - link.utilization)
                min_available_bandwidth = min(min_available_bandwidth, available_bw)
        
        # Convert from Mbps to MB per time step
        transmission_rate = min_available_bandwidth * self.config.environment.time_step_duration / 8
        return max(0.0, transmission_rate)
    
    def _calculate_path_latency(self, path: List[int]) -> float:
        """Calculate total latency for a path"""
        if len(path) < 2:
            return 0.0
        
        total_latency = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) in self.links:
                link = self.links[(u, v)]
                if not link.failed:
                    # Link latency + queuing delay
                    queuing_delay = link.queue_length * 0.1  # Simplified
                    total_latency += link.latency + queuing_delay
                    
                    # Add processing delay at node
                    total_latency += self.nodes[v]['processing_delay']
        
        return total_latency
    
    def _handle_failures(self):
        """Handle network failures and recovery"""
        if not self.config.network.link_failure_prob:
            return
        
        # Introduce new failures
        for link_key, link_state in self.links.items():
            if not link_state.failed and np.random.random() < self.config.network.link_failure_prob:
                link_state.failed = True
                link_state.failure_time = self.current_step
                recovery_time = np.random.randint(*self.config.network.recovery_time_range)
                link_state.recovery_time = self.current_step + recovery_time
                
                self.logger.info(f"Link {link_key} failed at step {self.current_step}")
        
        # Handle recovery
        for link_key, link_state in self.links.items():
            if link_state.failed and link_state.recovery_time <= self.current_step:
                link_state.failed = False
                link_state.failure_time = None
                link_state.recovery_time = None
                
                self.logger.info(f"Link {link_key} recovered at step {self.current_step}")
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step"""
        reward = 0.0
        
        # Latency penalty (normalized)
        if self.metrics['total_packets'] > 0:
            avg_latency = self.metrics['total_latency'] / self.metrics['total_packets']
            # Normalize latency to reasonable range
            normalized_latency = min(avg_latency / 100.0, 10.0)  # Cap at 10x normal
            reward += self.config.environment.latency_weight * normalized_latency
        
        # Throughput reward (normalized)
        if self.metrics['completed_flows'] > 0:
            # Reward based on flow completion rate
            completion_rate = self.metrics['completed_flows'] / max(1, self.episode_step)
            reward += self.config.environment.throughput_weight * completion_rate
        
        # Utilization efficiency reward (already 0-1)
        if self.links:
            avg_utilization = np.mean([link.utilization for link in self.links.values()])
            # Reward moderate utilization (not too low, not too high)
            utilization_reward = avg_utilization * (1 - avg_utilization)  # Peaks at 0.5 utilization
            reward += self.config.environment.utilization_weight * utilization_reward
        
        # Packet drop penalty (normalized)
        if self.metrics['total_packets'] > 0:
            drop_rate = self.metrics['dropped_packets'] / self.metrics['total_packets']
            reward += self.config.environment.drop_weight * drop_rate
        
        # Fairness reward (already 0-1)
        if len(self.active_flows) > 1:
            flow_latencies = [
                flow.total_latency / max(1, flow.packets_sent) 
                for flow in self.active_flows.values() 
                if flow.packets_sent > 0
            ]
            if len(flow_latencies) > 1:
                fairness_index = self._calculate_jains_fairness(flow_latencies)
                reward += self.config.environment.fairness_weight * fairness_index
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -100.0, 100.0)
        
        return float(reward)
    
    def _calculate_jains_fairness(self, values: List[float]) -> float:
        """Calculate Jain's fairness index"""
        if not values:
            return 0.0
        
        n = len(values)
        sum_values = sum(values)
        sum_squares = sum(x**2 for x in values)
        
        if sum_squares == 0:
            return 1.0
        
        return (sum_values**2) / (n * sum_squares)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Link utilizations (only unique links, not both directions)
        unique_links = set()
        for (u, v) in self.links.keys():
            if (v, u) not in unique_links:
                unique_links.add((u, v))
        
        for (u, v) in sorted(unique_links):
            obs.append(self.links[(u, v)].utilization)
        
        # Node queue lengths
        for node_id in sorted(self.nodes.keys()):
            obs.append(self.nodes[node_id]['queue_length'] / 100.0)  # Normalize
        
        # Active flow features (pad or truncate to max_concurrent_flows)
        flow_features = []
        for i in range(self.max_concurrent_flows):
            if i < len(self.active_flows):
                flow = list(self.active_flows.values())[i]
                # Normalize features
                flow_features.extend([
                    flow.source / len(self.nodes),
                    flow.destination / len(self.nodes),
                    min(1.0, flow.size / 100.0),  # Normalize size
                    min(1.0, flow.remaining_size / flow.size),
                    flow.priority / 4.0,  # Normalize priority
                    min(1.0, (self.current_step - flow.start_time) / 100.0)  # Age
                ])
            else:
                flow_features.extend([0.0] * 6)  # Padding
        
        obs.extend(flow_features)
        
        # Node failure status
        for node_id in sorted(self.nodes.keys()):
            obs.append(1.0 if self.nodes[node_id]['failed'] else 0.0)
        
        # Link failure status
        for (u, v) in sorted(unique_links):
            obs.append(1.0 if self.links[(u, v)].failed else 0.0)
        
        # Historical metrics (moving averages)
        recent_steps = 10
        if len(self.metrics['link_utilizations']) >= recent_steps:
            avg_util = np.mean(self.metrics['link_utilizations'][-recent_steps:])
        else:
            avg_util = 0.0
        
        if len(self.metrics['queue_lengths']) >= recent_steps:
            avg_queue = np.mean(self.metrics['queue_lengths'][-recent_steps:])
        else:
            avg_queue = 0.0
        
        # Add historical metrics
        historical_metrics = [
            avg_util,
            avg_queue,
            min(1.0, len(self.active_flows) / self.max_concurrent_flows),
            min(1.0, self.metrics['completed_flows'] / max(1, self.episode_step)),
            min(1.0, self.metrics['dropped_packets'] / max(1, self.metrics['total_packets'])),
            0.0, 0.0, 0.0, 0.0, 0.0  # Reserved for future metrics
        ]
        
        obs.extend(historical_metrics)
        
        # Update metrics history
        current_avg_util = np.mean([link.utilization for link in self.links.values()])
        current_avg_queue = np.mean([node['queue_length'] for node in self.nodes.values()])
        
        self.metrics['link_utilizations'].append(current_avg_util)
        self.metrics['queue_lengths'].append(current_avg_queue)
        
        # Keep only recent history
        max_history = 100
        if len(self.metrics['link_utilizations']) > max_history:
            self.metrics['link_utilizations'] = self.metrics['link_utilizations'][-max_history:]
        if len(self.metrics['queue_lengths']) > max_history:
            self.metrics['queue_lengths'] = self.metrics['queue_lengths'][-max_history:]
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state"""
        return {
            'episode_step': self.episode_step,
            'active_flows': len(self.active_flows),
            'completed_flows': self.metrics['completed_flows'],
            'dropped_packets': self.metrics['dropped_packets'],
            'total_packets': self.metrics['total_packets'],
            'avg_latency': (
                self.metrics['total_latency'] / max(1, self.metrics['total_packets'])
            ),
            'avg_link_utilization': np.mean([link.utilization for link in self.links.values()]),
            'failed_links': sum(1 for link in self.links.values() if link.failed),
            'network_nodes': self.network.number_of_nodes(),
            'network_edges': self.network.number_of_edges()
        }
    
    def render(self, mode='human'):
        """Render the environment (placeholder)"""
        if mode == 'human':
            print(f"Step: {self.episode_step}")
            print(f"Active flows: {len(self.active_flows)}")
            print(f"Completed flows: {self.metrics['completed_flows']}")
            print(f"Dropped packets: {self.metrics['dropped_packets']}")
            print(f"Average latency: {self.metrics['total_latency'] / max(1, self.metrics['total_packets']):.2f}")
            print("---")
    
    def close(self):
        """Clean up environment resources"""
        pass
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get detailed network state for visualization"""
        return {
            'network': self.network,
            'links': self.links,
            'nodes': self.nodes,
            'active_flows': self.active_flows,
            'metrics': self.metrics,
            'current_step': self.current_step
        }
    
    def set_network_failures(self, failed_links: List[Tuple[int, int]], 
                           failed_nodes: List[int]):
        """Manually set network failures (for testing)"""
        # Reset all failures first
        for link_state in self.links.values():
            link_state.failed = False
        
        for node_state in self.nodes.values():
            node_state['failed'] = False
        
        # Set specified failures
        for (u, v) in failed_links:
            if (u, v) in self.links:
                self.links[(u, v)].failed = True
                self.links[(u, v)].failure_time = self.current_step
            if (v, u) in self.links:
                self.links[(v, u)].failed = True
                self.links[(v, u)].failure_time = self.current_step
        
        for node_id in failed_nodes:
            if node_id in self.nodes:
                self.nodes[node_id]['failed'] = True
    
    def get_routing_table(self) -> Dict[int, List[int]]:
        """Get current routing decisions for all active flows"""
        routing_table = {}
        for flow_id, flow in self.active_flows.items():
            routing_table[flow_id] = flow.current_path or []
        return routing_table
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if self.metrics['total_packets'] > 0:
            avg_latency = self.metrics['total_latency'] / self.metrics['total_packets']
            packet_loss_rate = self.metrics['dropped_packets'] / self.metrics['total_packets']
        else:
            avg_latency = 0.0
            packet_loss_rate = 0.0
        
        if self.links:
            avg_utilization = np.mean([link.utilization for link in self.links.values()])
            utilization_variance = np.var([link.utilization for link in self.links.values()])
        else:
            avg_utilization = 0.0
            utilization_variance = 0.0
        
        # Calculate throughput (flows completed per time step)
        throughput = self.metrics['completed_flows'] / max(1, self.episode_step)
        
        # Calculate fairness index for active flows
        active_flow_latencies = [
            flow.total_latency / max(1, flow.packets_sent)
            for flow in self.active_flows.values()
            if flow.packets_sent > 0
        ]
        fairness_index = self._calculate_jains_fairness(active_flow_latencies)
        
        return {
            'avg_latency': avg_latency,
            'packet_loss_rate': packet_loss_rate,
            'avg_link_utilization': avg_utilization,
            'link_utilization_variance': utilization_variance,
            'throughput': throughput,
            'fairness_index': fairness_index,
            'completed_flows': self.metrics['completed_flows'],
            'active_flows': len(self.active_flows),
            'failed_links': sum(1 for link in self.links.values() if link.failed)
        }