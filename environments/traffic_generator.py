"""
Realistic Network Traffic Generator
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from config.config import TrafficConfig


@dataclass
class TrafficFlow:
    """Represents a generated traffic flow"""
    source: int
    destination: int
    size: float  # MB
    bandwidth_requirement: float  # Mbps
    max_latency: float  # ms
    priority: int  # 1-4 (4 = highest)
    traffic_type: str
    start_time: int
    duration: int  # time steps


class TrafficGenerator:
    """Generates realistic network traffic patterns"""
    
    def __init__(self, config: TrafficConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Traffic generation state
        self.current_hour = 0  # 0-23
        self.flows_generated = 0
        self.daily_cycle_position = 0
        
        # Cache for efficiency
        self._flow_size_cache = []
        self._duration_cache = []
        
        # Pre-generate some distributions for performance
        self._pregenerate_distributions()
        
    def _pregenerate_distributions(self):
        """Pre-generate heavy-tailed distributions for performance"""
        # Pre-generate Pareto-distributed flow sizes
        self._flow_size_cache = np.random.pareto(
            self.config.flow_size_alpha, 10000
        ) * self.config.flow_size_mean
        
        # Pre-generate exponential durations
        self._duration_cache = np.random.exponential(
            self.config.flow_duration_mean, 10000
        ).astype(int)
        
        self._cache_index = 0
        
        self.logger.info("Pre-generated traffic distributions")
    
    def generate_flows(self, current_step: int, 
                      available_nodes: Optional[List[int]] = None) -> List[Dict]:
        """Generate new traffic flows for the current time step"""
        
        # Update time tracking
        self._update_time_tracking(current_step)
        
        # Calculate arrival rate based on time of day
        base_rate = self.config.flow_arrival_rate
        if self.config.diurnal_pattern:
            rate_multiplier = self._get_diurnal_multiplier()
            arrival_rate = base_rate * rate_multiplier
        else:
            arrival_rate = base_rate
        
        # Poisson arrivals
        num_new_flows = np.random.poisson(arrival_rate)
        
        # Generate flows
        new_flows = []
        for _ in range(num_new_flows):
            flow_data = self._generate_single_flow(current_step, available_nodes)
            if flow_data:
                new_flows.append(flow_data)
        
        self.flows_generated += len(new_flows)
        
        if new_flows:
            self.logger.debug(f"Generated {len(new_flows)} new flows at step {current_step}")
        
        return new_flows
    
    def _update_time_tracking(self, current_step: int):
        """Update internal time tracking"""
        # Assume each step is 1 second by default
        seconds_per_day = 24 * 60 * 60
        seconds_elapsed = current_step % seconds_per_day
        
        self.current_hour = (seconds_elapsed // 3600) % 24
        self.daily_cycle_position = seconds_elapsed / seconds_per_day
    
    def _get_diurnal_multiplier(self) -> float:
        """Get traffic multiplier based on time of day"""
        if self.current_hour in self.config.peak_hours:
            return self.config.peak_multiplier
        elif self.current_hour in [2, 3, 4, 5]:  # Late night
            return 0.3
        elif self.current_hour in [6, 7, 8]:  # Early morning
            return 0.7
        elif self.current_hour in [22, 23, 0, 1]:  # Evening/night
            return 1.5
        else:
            return 1.0  # Normal hours
    
    def _generate_single_flow(self, current_step: int, 
                            available_nodes: Optional[List[int]] = None) -> Optional[Dict]:
        """Generate a single traffic flow"""
        
        # Default node range if not provided
        if available_nodes is None:
            available_nodes = list(range(10))  # Assume 10 nodes by default
        
        if len(available_nodes) < 2:
            return None
        
        # Select source and destination
        source, destination = self._select_endpoints(available_nodes)
        if source == destination or source not in available_nodes or destination not in available_nodes:
            return None
        
        # Select traffic type
        traffic_type = self._select_traffic_type()
        
        # Generate flow characteristics based on type
        flow_size = self._generate_flow_size(traffic_type)
        bandwidth_req = self._generate_bandwidth_requirement(traffic_type, flow_size)
        max_latency = self._generate_latency_requirement(traffic_type)
        priority = self._get_priority(traffic_type)
        duration = self._generate_flow_duration(traffic_type)
        
        return {
            'source': source,
            'destination': destination,
            'size': flow_size,
            'bandwidth_requirement': bandwidth_req,
            'max_latency': max_latency,
            'priority': priority,
            'traffic_type': traffic_type,
            'start_time': current_step,
            'duration': duration
        }
    
    def _select_endpoints(self, available_nodes: List[int]) -> Tuple[int, int]:
        """Select source and destination nodes with realistic patterns"""
        
        if len(available_nodes) < 2:
            # Fallback: return first two nodes if available
            if len(available_nodes) == 2:
                return available_nodes[0], available_nodes[1]
            else:
                return 0, 0  # This should be handled by caller
        
        # Implement gravity model: larger nodes attract more traffic
        # For simplicity, assume node ID correlates with "size"
        weights = np.array([1.0 + 0.1 * i for i in range(len(available_nodes))])
        weights = weights / weights.sum()
        
        # Select source
        source_idx = np.random.choice(len(available_nodes), p=weights)
        source = available_nodes[source_idx]
        
        # Select destination (biased against very distant nodes)
        dest_weights = weights.copy()
        dest_weights[source_idx] = 0  # Can't send to self
        
        # Add distance bias (closer nodes more likely)
        for i, node in enumerate(available_nodes):
            if node != source:
                distance_factor = 1.0 / (1.0 + 0.1 * abs(i - source_idx))
                dest_weights[i] *= distance_factor
        
        if dest_weights.sum() > 0:
            dest_weights = dest_weights / dest_weights.sum()
            dest_idx = np.random.choice(len(available_nodes), p=dest_weights)
            destination = available_nodes[dest_idx]
        else:
            # Fallback to random selection
            dest_candidates = [n for n in available_nodes if n != source]
            destination = np.random.choice(dest_candidates) if dest_candidates else source
        
        return source, destination
    
    def _select_traffic_type(self) -> str:
        """Select traffic type based on configured probabilities"""
        traffic_types = list(self.config.traffic_types.keys())
        probabilities = list(self.config.traffic_types.values())
        
        return np.random.choice(traffic_types, p=probabilities)
    
    def _generate_flow_size(self, traffic_type: str) -> float:
        """Generate flow size based on traffic type"""
        
        # Use cached Pareto distributions for performance
        if self._cache_index >= len(self._flow_size_cache):
            self._cache_index = 0
        
        base_size = self._flow_size_cache[self._cache_index]
        self._cache_index += 1
        
        # Adjust based on traffic type
        if traffic_type == "web":
            # Web traffic: mostly small with occasional large files
            if np.random.random() < 0.9:
                return max(0.001, base_size * 0.1)  # Small web requests
            else:
                return base_size * 2.0  # Large downloads
        
        elif traffic_type == "video":
            # Video traffic: consistently large
            return max(1.0, base_size * 5.0)
        
        elif traffic_type == "file_transfer":
            # File transfer: very large
            return max(10.0, base_size * 20.0)
        
        elif traffic_type == "real_time":
            # Real-time traffic: small, constant
            return max(0.0001, base_size * 0.01)
        
        else:
            return max(0.001, base_size)
    
    def _generate_bandwidth_requirement(self, traffic_type: str, flow_size: float) -> float:
        """Generate bandwidth requirement based on traffic type and size"""
        
        if traffic_type == "web":
            # Web traffic: moderate bandwidth
            return np.random.uniform(0.5, 5.0)
        
        elif traffic_type == "video":
            # Video streaming: high bandwidth, depends on quality
            quality_factor = np.random.choice([1, 2, 4, 8], p=[0.3, 0.4, 0.2, 0.1])
            return np.random.uniform(2.0, 10.0) * quality_factor
        
        elif traffic_type == "file_transfer":
            # File transfer: wants maximum available bandwidth
            return np.random.uniform(10.0, 100.0)
        
        elif traffic_type == "real_time":
            # Real-time: low bandwidth but consistent
            return np.random.uniform(0.1, 1.0)
        
        else:
            return np.random.uniform(1.0, 10.0)
    
    def _generate_latency_requirement(self, traffic_type: str) -> float:
        """Generate maximum acceptable latency based on traffic type"""
        
        qos_reqs = self.config.qos_requirements.get(traffic_type, {})
        base_latency = qos_reqs.get('max_latency', 100.0)
        
        # Add some variation
        variation = np.random.uniform(0.8, 1.2)
        return base_latency * variation
    
    def _get_priority(self, traffic_type: str) -> int:
        """Get priority level for traffic type"""
        qos_reqs = self.config.qos_requirements.get(traffic_type, {})
        return qos_reqs.get('priority', 2)
    
    def _generate_flow_duration(self, traffic_type: str) -> int:
        """Generate flow duration based on traffic type"""
        
        # Use cached exponential distributions
        if self._cache_index >= len(self._duration_cache):
            self._cache_index = 0
        
        base_duration = self._duration_cache[self._cache_index]
        self._cache_index += 1
        
        # Adjust based on traffic type
        if traffic_type == "web":
            # Web traffic: short bursts
            return max(1, int(base_duration * 0.1))
        
        elif traffic_type == "video":
            # Video streaming: long duration
            return max(30, int(base_duration * 3.0))
        
        elif traffic_type == "file_transfer":
            # File transfer: variable, depends on size
            return max(5, int(base_duration * 0.5))
        
        elif traffic_type == "real_time":
            # Real-time: very long sessions
            return max(60, int(base_duration * 5.0))
        
        else:
            return max(1, base_duration)
    
    def generate_traffic_matrix(self, num_nodes: int, 
                              time_steps: int) -> np.ndarray:
        """Generate a traffic matrix for analysis"""
        
        traffic_matrix = np.zeros((num_nodes, num_nodes, time_steps))
        
        for step in range(time_steps):
            flows = self.generate_flows(step, list(range(num_nodes)))
            
            for flow in flows:
                source = flow['source']
                dest = flow['destination']
                duration = flow['duration']
                
                # Add traffic to matrix for the duration of the flow
                end_step = min(step + duration, time_steps)
                for t in range(step, end_step):
                    traffic_matrix[source, dest, t] += flow['bandwidth_requirement']
        
        return traffic_matrix
    
    def get_traffic_statistics(self, flows: List[Dict]) -> Dict:
        """Get statistics about generated traffic"""
        
        if not flows:
            return {
                'total_flows': 0,
                'total_bandwidth': 0.0,
                'avg_flow_size': 0.0,
                'traffic_type_distribution': {}
            }
        
        total_bandwidth = sum(flow['bandwidth_requirement'] for flow in flows)
        total_size = sum(flow['size'] for flow in flows)
        
        # Traffic type distribution
        type_counts = {}
        for flow in flows:
            traffic_type = flow['traffic_type']
            type_counts[traffic_type] = type_counts.get(traffic_type, 0) + 1
        
        # Priority distribution
        priority_counts = {}
        for flow in flows:
            priority = flow['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            'total_flows': len(flows),
            'total_bandwidth': total_bandwidth,
            'avg_bandwidth': total_bandwidth / len(flows),
            'total_size': total_size,
            'avg_flow_size': total_size / len(flows),
            'traffic_type_distribution': type_counts,
            'priority_distribution': priority_counts,
            'avg_latency_requirement': np.mean([f['max_latency'] for f in flows]),
            'avg_duration': np.mean([f['duration'] for f in flows])
        }
    
    def create_traffic_scenario(self, scenario_type: str) -> Dict:
        """Create predefined traffic scenarios for testing"""
        
        scenarios = {
            'normal': {
                'flow_arrival_rate': self.config.flow_arrival_rate,
                'peak_multiplier': 2.0,
                'traffic_types': self.config.traffic_types
            },
            
            'peak_load': {
                'flow_arrival_rate': self.config.flow_arrival_rate * 3,
                'peak_multiplier': 5.0,
                'traffic_types': self.config.traffic_types
            },
            
            'video_heavy': {
                'flow_arrival_rate': self.config.flow_arrival_rate,
                'peak_multiplier': 2.0,
                'traffic_types': {
                    'web': 0.2, 'video': 0.6, 'file_transfer': 0.1, 'real_time': 0.1
                }
            },
            
            'file_transfer_heavy': {
                'flow_arrival_rate': self.config.flow_arrival_rate * 0.5,
                'peak_multiplier': 2.0,
                'traffic_types': {
                    'web': 0.2, 'video': 0.1, 'file_transfer': 0.6, 'real_time': 0.1
                }
            },
            
            'real_time_heavy': {
                'flow_arrival_rate': self.config.flow_arrival_rate * 2,
                'peak_multiplier': 2.0,
                'traffic_types': {
                    'web': 0.2, 'video': 0.2, 'file_transfer': 0.1, 'real_time': 0.5
                }
            },
            
            'flash_crowd': {
                'flow_arrival_rate': self.config.flow_arrival_rate * 10,
                'peak_multiplier': 1.0,  # No diurnal pattern
                'traffic_types': {
                    'web': 0.8, 'video': 0.1, 'file_transfer': 0.05, 'real_time': 0.05
                }
            }
        }
        
        return scenarios.get(scenario_type, scenarios['normal'])
    
    def apply_scenario(self, scenario_type: str):
        """Apply a traffic scenario to the generator"""
        scenario = self.create_traffic_scenario(scenario_type)
        
        # Temporarily modify configuration
        self.original_config = {
            'flow_arrival_rate': self.config.flow_arrival_rate,
            'peak_multiplier': self.config.peak_multiplier,
            'traffic_types': self.config.traffic_types.copy()
        }
        
        self.config.flow_arrival_rate = scenario['flow_arrival_rate']
        self.config.peak_multiplier = scenario['peak_multiplier']
        self.config.traffic_types = scenario['traffic_types']
        
        self.logger.info(f"Applied traffic scenario: {scenario_type}")
    
    def reset_scenario(self):
        """Reset to original configuration"""
        if hasattr(self, 'original_config'):
            self.config.flow_arrival_rate = self.original_config['flow_arrival_rate']
            self.config.peak_multiplier = self.original_config['peak_multiplier']
            self.config.traffic_types = self.original_config['traffic_types']
            
            delattr(self, 'original_config')
            self.logger.info("Reset to original traffic configuration")
    
    def save_traffic_trace(self, flows: List[Dict], filename: str):
        """Save generated traffic trace to file"""
        import json
        from pathlib import Path
        
        filepath = Path("data/traffic_patterns") / f"{filename}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to JSON serializable
        serializable_flows = []
        for flow in flows:
            serializable_flow = {}
            for key, value in flow.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_flow[key] = value.item()
                else:
                    serializable_flow[key] = value
            serializable_flows.append(serializable_flow)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_flows, f, indent=2)
        
        self.logger.info(f"Saved traffic trace to {filepath}")
    
    def load_traffic_trace(self, filename: str) -> List[Dict]:
        """Load traffic trace from file"""
        import json
        from pathlib import Path
        
        filepath = Path("data/traffic_patterns") / f"{filename}.json"
        
        with open(filepath, 'r') as f:
            flows = json.load(f)
        
        self.logger.info(f"Loaded traffic trace from {filepath}")
        return flows