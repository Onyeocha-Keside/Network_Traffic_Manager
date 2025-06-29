"""
Unit tests for network environment components
"""
import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch

from config.config import Config, NetworkConfig, TrafficConfig
from environments.network_env import NetworkEnvironment
from environments.topology_generator import TopologyGenerator
from environments.traffic_generator import TrafficGenerator


class TestTopologyGenerator:
    """Test topology generation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = NetworkConfig(num_nodes=6, topology_type="mesh", connectivity=0.5)
        self.generator = TopologyGenerator(self.config)
    
    def test_mesh_topology_generation(self):
        """Test mesh topology generation"""
        network = self.generator._generate_mesh_topology()
        
        assert network.number_of_nodes() == 6
        assert nx.is_connected(network)
        assert all('bandwidth' in data for _, _, data in network.edges(data=True))
        assert all('latency' in data for _, _, data in network.edges(data=True))
    
    def test_ring_topology_generation(self):
        """Test ring topology generation"""
        network = self.generator._generate_ring_topology()
        
        assert network.number_of_nodes() == 6
        assert nx.is_connected(network)
        # Ring should have at least n edges (basic ring)
        assert network.number_of_edges() >= 6
    
    def test_tree_topology_generation(self):
        """Test tree topology generation"""
        network = self.generator._generate_tree_topology()
        
        assert network.number_of_nodes() == 6
        assert nx.is_connected(network)
        assert network.number_of_edges() == 5  # Tree has n-1 edges
        assert nx.is_tree(network)
    
    def test_fat_tree_topology_generation(self):
        """Test fat-tree topology generation"""
        self.config.num_nodes = 16
        network = self.generator._generate_fat_tree_topology()
        
        assert network.number_of_nodes() <= 16
        assert nx.is_connected(network)
        
        # Check hierarchical structure
        node_types = set(network.nodes[n].get('node_type', 'router') for n in network.nodes())
        assert len(node_types) > 1  # Should have multiple node types
    
    def test_link_properties_generation(self):
        """Test link properties are properly generated"""
        link_props = self.generator._generate_link_properties()
        
        required_props = ['bandwidth', 'latency', 'cost', 'reliability']
        for prop in required_props:
            assert prop in link_props
            assert isinstance(link_props[prop], (int, float))
            assert link_props[prop] > 0
        
        # Test tier-specific properties
        core_props = self.generator._generate_link_properties(tier="core")
        edge_props = self.generator._generate_link_properties(tier="edge")
        
        assert core_props['bandwidth'] > edge_props['bandwidth']
        assert core_props['latency'] < edge_props['latency']
    
    def test_connectivity_enforcement(self):
        """Test that generated graphs are always connected"""
        # Test with very low connectivity
        self.config.connectivity = 0.1
        network = self.generator._generate_mesh_topology()
        
        assert nx.is_connected(network)
    
    def test_topology_statistics(self):
        """Test topology statistics calculation"""
        network = self.generator.generate_topology()
        stats = self.generator.get_topology_stats(network)
        
        required_stats = ['num_nodes', 'num_edges', 'density', 'is_connected', 
                         'avg_degree', 'node_type_counts']
        
        for stat in required_stats:
            assert stat in stats
        
        assert stats['num_nodes'] == self.config.num_nodes
        assert stats['is_connected'] is True


class TestTrafficGenerator:
    """Test traffic generation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = TrafficConfig(flow_arrival_rate=3.0, diurnal_pattern=True)
        self.generator = TrafficGenerator(self.config)
    
    def test_flow_generation(self):
        """Test basic flow generation"""
        flows = self.generator.generate_flows(current_step=0, available_nodes=list(range(10)))
        
        assert isinstance(flows, list)
        
        if flows:  # Flows may be empty due to Poisson process
            flow = flows[0]
            required_fields = ['source', 'destination', 'size', 'bandwidth_requirement', 
                             'max_latency', 'priority', 'traffic_type', 'start_time', 'duration']
            
            for field in required_fields:
                assert field in flow
            
            assert flow['source'] != flow['destination']
            assert flow['size'] > 0
            assert flow['bandwidth_requirement'] > 0
            assert 1 <= flow['priority'] <= 4
    
    def test_diurnal_patterns(self):
        """Test diurnal traffic patterns"""
        # Test peak hours
        self.generator.current_hour = 10  # Peak hour
        peak_multiplier = self.generator._get_diurnal_multiplier()
        
        # Test off-peak hours
        self.generator.current_hour = 3  # Late night
        off_peak_multiplier = self.generator._get_diurnal_multiplier()
        
        assert peak_multiplier > off_peak_multiplier
    
    def test_traffic_types(self):
        """Test different traffic type generation"""
        traffic_types = set()
        
        # Generate multiple flows to test variety
        for _ in range(50):
            traffic_type = self.generator._select_traffic_type()
            traffic_types.add(traffic_type)
        
        # Should generate different traffic types
        assert len(traffic_types) > 1
        assert all(t_type in self.config.traffic_types for t_type in traffic_types)
    
    def test_flow_size_distribution(self):
        """Test flow size follows expected distribution"""
        sizes = []
        
        for _ in range(100):
            size = self.generator._generate_flow_size("web")
            sizes.append(size)
        
        # Should have variety in sizes
        assert len(set(sizes)) > 10
        assert all(size > 0 for size in sizes)
        
        # Video flows should generally be larger than web flows
        video_sizes = [self.generator._generate_flow_size("video") for _ in range(50)]
        web_sizes = [self.generator._generate_flow_size("web") for _ in range(50)]
        
        assert np.mean(video_sizes) > np.mean(web_sizes)
    
    def test_bandwidth_requirements(self):
        """Test bandwidth requirement generation"""
        web_bw = self.generator._generate_bandwidth_requirement("web", 1.0)
        video_bw = self.generator._generate_bandwidth_requirement("video", 10.0)
        file_bw = self.generator._generate_bandwidth_requirement("file_transfer", 100.0)
        
        assert video_bw > web_bw
        assert file_bw > video_bw
    
    def test_traffic_scenarios(self):
        """Test predefined traffic scenarios"""
        scenarios = ["normal", "peak_load", "video_heavy", "flash_crowd"]
        
        for scenario in scenarios:
            scenario_config = self.generator.create_traffic_scenario(scenario)
            
            assert 'flow_arrival_rate' in scenario_config
            assert 'traffic_types' in scenario_config
            assert sum(scenario_config['traffic_types'].values()) == pytest.approx(1.0)
    
    def test_traffic_statistics(self):
        """Test traffic statistics calculation"""
        flows = []
        for _ in range(20):
            flow_batch = self.generator.generate_flows(0, list(range(10)))
            flows.extend(flow_batch)
        
        if flows:
            stats = self.generator.get_traffic_statistics(flows)
            
            required_stats = ['total_flows', 'total_bandwidth', 'avg_bandwidth', 
                            'total_size', 'avg_flow_size', 'traffic_type_distribution']
            
            for stat in required_stats:
                assert stat in stats
            
            assert stats['total_flows'] == len(flows)
            assert stats['avg_bandwidth'] > 0
            assert stats['avg_flow_size'] > 0


class TestNetworkEnvironment:
    """Test network environment functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = Config()
        self.config.network.num_nodes = 6
        self.config.environment.max_episode_steps = 100
        self.env = NetworkEnvironment(self.config)
    
    def test_environment_initialization(self):
        """Test environment initialization"""
        assert self.env.network is not None
        assert self.env.network.number_of_nodes() == 6
        assert len(self.env.links) > 0
        assert len(self.env.nodes) == 6
        
        # Test observation and action spaces
        assert self.env.observation_space is not None
        assert self.env.action_space is not None
    
    def test_environment_reset(self):
        """Test environment reset functionality"""
        obs, info = self.env.reset()
        
        assert obs is not None
        assert obs.shape == self.env.observation_space.shape
        assert isinstance(info, dict)
        
        # Check that environment state is reset
        assert self.env.current_step == 0
        assert self.env.episode_step == 0
        assert len(self.env.active_flows) == 0
    
    def test_environment_step(self):
        """Test environment step functionality"""
        obs, info = self.env.reset()
        
        # Take a random action
        action = self.env.action_space.sample()
        next_obs, reward, done, truncated, info = self.env.step(action)
        
        assert next_obs.shape == obs.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check environment state progression
        assert self.env.current_step == 1
        assert self.env.episode_step == 1
    
    def test_observation_structure(self):
        """Test observation structure and content"""
        obs, _ = self.env.reset()
        
        # Observation should be properly normalized
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)
        
        # Take several steps and check observation consistency
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, _, _, _, _ = self.env.step(action)
            
            assert obs.shape == self.env.observation_space.shape
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))
    
    def test_action_application(self):
        """Test action application and routing decisions"""
        obs, _ = self.env.reset()
        
        # Generate some flows first
        self.env._generate_new_flows()
        
        if self.env.active_flows:
            action = self.env.action_space.sample()
            self.env._apply_actions(action)
            
            # Check that flows have paths assigned
            for flow in self.env.active_flows.values():
                if flow.current_path:
                    assert isinstance(flow.current_path, list)
                    assert len(flow.current_path) >= 2  # At least source and destination
    
    def test_reward_calculation(self):
        """Test reward calculation"""
        obs, _ = self.env.reset()
        
        # Take several steps and check rewards
        rewards = []
        for _ in range(20):
            action = self.env.action_space.sample()
            _, reward, _, _, _ = self.env.step(action)
            rewards.append(reward)
        
        # Rewards should be finite numbers
        assert all(np.isfinite(r) for r in rewards)
        
        # Should have some variation in rewards
        if len(set(rewards)) > 1:
            assert np.std(rewards) > 0
    
    def test_network_failures(self):
        """Test network failure handling"""
        obs, _ = self.env.reset()
        
        # Set some link failures
        failed_links = [(0, 1), (1, 2)]
        self.env.set_network_failures(failed_links, [])
        
        # Check that links are marked as failed
        for u, v in failed_links:
            if (u, v) in self.env.links:
                assert self.env.links[(u, v)].failed
            if (v, u) in self.env.links:
                assert self.env.links[(v, u)].failed
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        obs, _ = self.env.reset()
        
        # Take several steps to generate some metrics
        for _ in range(10):
            action = self.env.action_space.sample()
            self.env.step(action)
        
        metrics = self.env.get_performance_metrics()
        
        required_metrics = ['avg_latency', 'packet_loss_rate', 'avg_link_utilization', 
                          'throughput', 'fairness_index', 'completed_flows', 'active_flows']
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert np.isfinite(metrics[metric])
    
    def test_network_state_export(self):
        """Test network state export functionality"""
        obs, _ = self.env.reset()
        
        # Take some steps
        for _ in range(5):
            action = self.env.action_space.sample()
            self.env.step(action)
        
        network_state = self.env.get_network_state()
        
        required_keys = ['network', 'links', 'nodes', 'active_flows', 'metrics', 'current_step']
        
        for key in required_keys:
            assert key in network_state
        
        # Check network state integrity
        assert network_state['network'] == self.env.network
        assert network_state['current_step'] == self.env.current_step


class TestEnvironmentIntegration:
    """Test integration between environment components"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = Config()
        self.config.network.num_nodes = 8
        self.config.traffic.flow_arrival_rate = 2.0
        
    def test_end_to_end_episode(self):
        """Test complete episode execution"""
        env = NetworkEnvironment(self.config)
        
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        done = False
        while not done and steps < 50:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Verify environment consistency
            assert obs.shape == env.observation_space.shape
            assert np.isfinite(reward)
        
        # Check final state
        final_metrics = env.get_performance_metrics()
        assert isinstance(final_metrics, dict)
        assert steps > 0
    
    def test_multiple_episodes(self):
        """Test multiple episode execution"""
        env = NetworkEnvironment(self.config)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(5):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done and episode_length < 30:
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Check episode consistency
        assert len(episode_rewards) == 5
        assert len(episode_lengths) == 5
        assert all(length > 0 for length in episode_lengths)
    
    def test_traffic_flow_lifecycle(self):
        """Test complete traffic flow lifecycle"""
        env = NetworkEnvironment(self.config)
        obs, info = env.reset()
        
        initial_flows = len(env.active_flows)
        
        # Run environment to generate and process flows
        for _ in range(30):
            action = env.action_space.sample()
            env.step(action)
        
        # Should have processed some flows
        assert env.metrics['total_packets'] >= 0
        
        # If flows were completed, metrics should reflect that
        if env.metrics['completed_flows'] > 0:
            assert env.metrics['total_latency'] > 0


if __name__ == "__main__":
    pytest.main([__file__])