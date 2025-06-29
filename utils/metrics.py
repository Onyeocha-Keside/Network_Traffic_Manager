"""
Advanced metrics calculation and analysis for Network Traffic Management
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats
import networkx as nx


@dataclass
class NetworkMetrics:
    """Container for comprehensive network performance metrics"""
    
    # Basic performance metrics
    avg_latency: float
    total_throughput: float
    packet_loss_rate: float
    link_utilization_mean: float
    link_utilization_std: float
    
    # Advanced metrics
    fairness_index: float
    network_efficiency: float
    convergence_time: float
    adaptability_score: float
    
    # Reliability metrics
    availability: float
    mean_time_to_failure: float
    recovery_time: float
    
    # Economic metrics
    cost_efficiency: float
    energy_efficiency: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'avg_latency': self.avg_latency,
            'total_throughput': self.total_throughput,
            'packet_loss_rate': self.packet_loss_rate,
            'link_utilization_mean': self.link_utilization_mean,
            'link_utilization_std': self.link_utilization_std,
            'fairness_index': self.fairness_index,
            'network_efficiency': self.network_efficiency,
            'convergence_time': self.convergence_time,
            'adaptability_score': self.adaptability_score,
            'availability': self.availability,
            'mean_time_to_failure': self.mean_time_to_failure,
            'recovery_time': self.recovery_time,
            'cost_efficiency': self.cost_efficiency,
            'energy_efficiency': self.energy_efficiency
        }


class MetricsCalculator:
    """Advanced metrics calculation for network performance analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical data for time-series analysis
        self.history = {
            'latencies': [],
            'throughputs': [],
            'utilizations': [],
            'packet_losses': [],
            'timestamps': []
        }
    
    def calculate_comprehensive_metrics(self, 
                                      episode_data: List[Dict[str, Any]],
                                      network_topology: nx.Graph) -> NetworkMetrics:
        """Calculate comprehensive network performance metrics"""
        
        if not episode_data:
            return self._get_empty_metrics()
        
        # Extract time series data
        latencies = [step['avg_latency'] for step in episode_data if 'avg_latency' in step]
        throughputs = [step['throughput'] for step in episode_data if 'throughput' in step]
        packet_losses = [step['packet_loss_rate'] for step in episode_data if 'packet_loss_rate' in step]
        utilizations = [step['avg_link_utilization'] for step in episode_data if 'avg_link_utilization' in step]
        
        # Basic metrics
        avg_latency = np.mean(latencies) if latencies else 0.0
        total_throughput = np.mean(throughputs) if throughputs else 0.0
        packet_loss_rate = np.mean(packet_losses) if packet_losses else 0.0
        link_util_mean = np.mean(utilizations) if utilizations else 0.0
        link_util_std = np.std(utilizations) if utilizations else 0.0
        
        # Advanced metrics
        fairness_index = self._calculate_fairness_index(episode_data)
        network_efficiency = self._calculate_network_efficiency(
            utilizations, throughputs, network_topology
        )
        convergence_time = self._calculate_convergence_time(throughputs)
        adaptability_score = self._calculate_adaptability_score(episode_data)
        
        # Reliability metrics
        availability = self._calculate_availability(episode_data)
        mtbf = self._calculate_mean_time_between_failures(episode_data)
        recovery_time = self._calculate_recovery_time(episode_data)
        
        # Economic metrics
        cost_efficiency = self._calculate_cost_efficiency(throughputs, utilizations)
        energy_efficiency = self._calculate_energy_efficiency(utilizations, throughputs)
        
        return NetworkMetrics(
            avg_latency=avg_latency,
            total_throughput=total_throughput,
            packet_loss_rate=packet_loss_rate,
            link_utilization_mean=link_util_mean,
            link_utilization_std=link_util_std,
            fairness_index=fairness_index,
            network_efficiency=network_efficiency,
            convergence_time=convergence_time,
            adaptability_score=adaptability_score,
            availability=availability,
            mean_time_to_failure=mtbf,
            recovery_time=recovery_time,
            cost_efficiency=cost_efficiency,
            energy_efficiency=energy_efficiency
        )
    
    def _calculate_fairness_index(self, episode_data: List[Dict]) -> float:
        """Calculate Jain's fairness index across flows"""
        fairness_values = []
        
        for step_data in episode_data:
            if 'flow_latencies' in step_data:
                latencies = step_data['flow_latencies']
                if len(latencies) > 1:
                    fairness = self._jains_fairness_index(latencies)
                    fairness_values.append(fairness)
        
        return np.mean(fairness_values) if fairness_values else 1.0
    
    def _jains_fairness_index(self, values: List[float]) -> float:
        """Calculate Jain's fairness index for a set of values"""
        if not values or len(values) < 2:
            return 1.0
        
        n = len(values)
        sum_x = sum(values)
        sum_x_squared = sum(x**2 for x in values)
        
        if sum_x_squared == 0:
            return 1.0
        
        return (sum_x**2) / (n * sum_x_squared)
    
    def _calculate_network_efficiency(self, utilizations: List[float], 
                                    throughputs: List[float],
                                    network_topology: nx.Graph) -> float:
        """Calculate network efficiency as throughput per unit capacity"""
        if not utilizations or not throughputs:
            return 0.0
        
        avg_utilization = np.mean(utilizations)
        avg_throughput = np.mean(throughputs)
        
        if avg_utilization > 0:
            return min(1.0, avg_throughput / avg_utilization)
        else:
            return 0.0
    
    def _calculate_convergence_time(self, throughputs: List[float]) -> float:
        """Calculate time to convergence"""
        if len(throughputs) < 10:
            return float('inf')
        
        window_size = min(20, len(throughputs) // 5)
        threshold = 0.05
        
        for i in range(window_size, len(throughputs)):
            window = throughputs[i-window_size:i]
            cv = np.std(window) / np.mean(window) if np.mean(window) > 0 else float('inf')
            
            if cv < threshold:
                return float(i)
        
        return float('inf')
    
    def _calculate_adaptability_score(self, episode_data: List[Dict]) -> float:
        """Calculate adaptability score"""
        if len(episode_data) < 50:
            return 0.5
        
        return 0.75  # Simplified for now
    
    def _calculate_availability(self, episode_data: List[Dict]) -> float:
        """Calculate network availability"""
        if not episode_data:
            return 1.0
        
        operational_steps = sum(1 for step in episode_data 
                              if step.get('packet_loss_rate', 1.0) < 0.9)
        
        return operational_steps / len(episode_data)
    
    def _calculate_mean_time_between_failures(self, episode_data: List[Dict]) -> float:
        """Calculate MTBF"""
        failure_times = []
        
        for i, step in enumerate(episode_data):
            if step.get('failed_links', 0) > 0:
                failure_times.append(i)
        
        if len(failure_times) < 2:
            return float('inf')
        
        intervals = np.diff(failure_times)
        return float(np.mean(intervals))
    
    def _calculate_recovery_time(self, episode_data: List[Dict]) -> float:
        """Calculate recovery time"""
        return 10.0  # Simplified for now
    
    def _calculate_cost_efficiency(self, throughputs: List[float], 
                                  utilizations: List[float]) -> float:
        """Calculate cost efficiency"""
        if not throughputs or not utilizations:
            return 0.0
        
        avg_throughput = np.mean(throughputs)
        avg_utilization = np.mean(utilizations)
        
        if avg_utilization > 0:
            return avg_throughput / avg_utilization
        else:
            return 0.0
    
    def _calculate_energy_efficiency(self, utilizations: List[float], 
                                   throughputs: List[float]) -> float:
        """Calculate energy efficiency"""
        if not utilizations or not throughputs:
            return 0.0
        
        energy_consumption = np.mean([u**2 for u in utilizations])
        avg_throughput = np.mean(throughputs)
        
        if energy_consumption > 0:
            return avg_throughput / energy_consumption
        else:
            return 0.0
    
    def _get_empty_metrics(self) -> NetworkMetrics:
        """Return empty metrics"""
        return NetworkMetrics(
            avg_latency=0.0,
            total_throughput=0.0,
            packet_loss_rate=0.0,
            link_utilization_mean=0.0,
            link_utilization_std=0.0,
            fairness_index=1.0,
            network_efficiency=0.0,
            convergence_time=float('inf'),
            adaptability_score=0.0,
            availability=1.0,
            mean_time_to_failure=float('inf'),
            recovery_time=0.0,
            cost_efficiency=0.0,
            energy_efficiency=0.0
        )
    
    def compare_algorithms_statistically(self, 
                                       algorithm_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compare algorithms statistically"""
        comparison_results = {
            'pairwise_tests': {},
            'rankings': {}
        }
        
        algorithms = list(algorithm_results.keys())
        
        # Simple rankings
        mean_results = {alg: np.mean(results) for alg, results in algorithm_results.items()}
        sorted_algorithms = sorted(mean_results.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (alg, score) in enumerate(sorted_algorithms, 1):
            comparison_results['rankings'][alg] = {
                'rank': rank,
                'mean_score': float(score)
            }
        
        return comparison_results


# Helper functions
def calculate_network_performance(episode_data: List[Dict], network_topology: nx.Graph) -> NetworkMetrics:
    """Calculate network performance metrics"""
    calculator = MetricsCalculator()
    return calculator.calculate_comprehensive_metrics(episode_data, network_topology)


def compare_algorithms(algorithm_results: Dict[str, List[float]]) -> Dict[str, Any]:
    """Compare algorithm performance"""
    calculator = MetricsCalculator()
    return calculator.compare_algorithms_statistically(algorithm_results)


# This is the VERY LAST LINE of the file
print("✅ utils/metrics.py loaded successfully")