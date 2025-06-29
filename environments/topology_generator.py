"""
Network Topology Generator for different network architectures
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from config.config import NetworkConfig


@dataclass
class LinkProperties:
    """Properties of a network link"""
    bandwidth: float  # Mbps
    latency: float   # ms
    cost: float      # Routing cost
    reliability: float  # 0-1


class TopologyGenerator:
    """Generates various network topologies with realistic characteristics"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def generate_topology(self) -> nx.Graph:
        """Generate network topology based on configuration"""
        topology_type = self.config.topology_type.lower()
        
        if topology_type == "mesh":
            return self._generate_mesh_topology()
        elif topology_type == "ring":
            return self._generate_ring_topology()
        elif topology_type == "tree":
            return self._generate_tree_topology()
        elif topology_type == "fat_tree":
            return self._generate_fat_tree_topology()
        elif topology_type == "small_world":
            return self._generate_small_world_topology()
        elif topology_type == "scale_free":
            return self._generate_scale_free_topology()
        elif topology_type == "datacenter":
            return self._generate_datacenter_topology()
        else:
            self.logger.warning(f"Unknown topology type: {topology_type}, using mesh")
            return self._generate_mesh_topology()
    
    def _generate_mesh_topology(self) -> nx.Graph:
        """Generate a mesh topology with random connections"""
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.config.num_nodes):
            G.add_node(i, node_type="router")
        
        # Add edges based on connectivity parameter
        for i in range(self.config.num_nodes):
            for j in range(i + 1, self.config.num_nodes):
                if np.random.random() < self.config.connectivity:
                    link_props = self._generate_link_properties()
                    G.add_edge(i, j, **link_props)
        
        # Ensure connectivity
        G = self._ensure_connectivity(G)
        
        self.logger.info(f"Generated mesh topology with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges")
        return G
    
    def _generate_ring_topology(self) -> nx.Graph:
        """Generate a ring topology"""
        G = nx.cycle_graph(self.config.num_nodes)
        
        # Add link properties
        for u, v in G.edges():
            link_props = self._generate_link_properties()
            G.edges[u, v].update(link_props)
        
        # Add node properties
        for node in G.nodes():
            G.nodes[node]['node_type'] = "router"
        
        # Optionally add some cross-connections for redundancy
        if self.config.connectivity > 0.5:
            num_cross_links = int(self.config.num_nodes * 0.2)
            for _ in range(num_cross_links):
                u = np.random.randint(0, self.config.num_nodes)
                v = np.random.randint(0, self.config.num_nodes)
                if u != v and not G.has_edge(u, v):
                    link_props = self._generate_link_properties()
                    G.add_edge(u, v, **link_props)
        
        self.logger.info(f"Generated ring topology with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges")
        return G
    
    def _generate_tree_topology(self) -> nx.Graph:
        """Generate a tree topology"""
        # Create a random tree
        G = nx.random_tree(self.config.num_nodes, seed=42)
        
        # Add link properties
        for u, v in G.edges():
            link_props = self._generate_link_properties()
            G.edges[u, v].update(link_props)
        
        # Add node properties with hierarchy
        for node in G.nodes():
            degree = G.degree[node]
            if degree == 1:
                node_type = "edge"
            elif degree <= 3:
                node_type = "aggregation"
            else:
                node_type = "core"
            G.nodes[node]['node_type'] = node_type
        
        self.logger.info(f"Generated tree topology with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges")
        return G
    
    def _generate_fat_tree_topology(self) -> nx.Graph:
        """Generate a fat-tree topology (common in data centers)"""
        # For simplicity, create a 3-level fat-tree
        k = int(np.sqrt(self.config.num_nodes // 2))  # Parameter k
        if k < 2:
            k = 2
        
        G = nx.Graph()
        node_id = 0
        
        # Core switches
        core_switches = []
        for i in range((k // 2) ** 2):
            G.add_node(node_id, node_type="core", level=2)
            core_switches.append(node_id)
            node_id += 1
        
        # Aggregation switches (k pods, k/2 switches per pod)
        agg_switches = []
        for pod in range(k):
            pod_agg = []
            for i in range(k // 2):
                G.add_node(node_id, node_type="aggregation", level=1, pod=pod)
                pod_agg.append(node_id)
                agg_switches.append(node_id)
                node_id += 1
        
        # Edge switches (k pods, k/2 switches per pod)
        edge_switches = []
        for pod in range(k):
            pod_edge = []
            for i in range(k // 2):
                G.add_node(node_id, node_type="edge", level=0, pod=pod)
                pod_edge.append(node_id)
                edge_switches.append(node_id)
                node_id += 1
        
        # Add remaining nodes as servers if needed
        while node_id < self.config.num_nodes:
            pod = np.random.randint(0, k)
            G.add_node(node_id, node_type="server", level=-1, pod=pod)
            node_id += 1
        
        # Add edges between levels
        self._add_fat_tree_edges(G, core_switches, agg_switches, edge_switches, k)
        
        self.logger.info(f"Generated fat-tree topology with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges")
        return G
    
    def _add_fat_tree_edges(self, G: nx.Graph, core_switches: List[int], 
                           agg_switches: List[int], edge_switches: List[int], k: int):
        """Add edges for fat-tree topology"""
        # Core to aggregation connections
        for i, core in enumerate(core_switches):
            for pod in range(k):
                agg_in_pod = [n for n in agg_switches if G.nodes[n].get('pod') == pod]
                if agg_in_pod:
                    agg_node = agg_in_pod[i % len(agg_in_pod)]
                    link_props = self._generate_link_properties(tier="core")
                    G.add_edge(core, agg_node, **link_props)
        
        # Aggregation to edge connections
        for pod in range(k):
            agg_in_pod = [n for n in agg_switches if G.nodes[n].get('pod') == pod]
            edge_in_pod = [n for n in edge_switches if G.nodes[n].get('pod') == pod]
            
            for agg in agg_in_pod:
                for edge in edge_in_pod:
                    link_props = self._generate_link_properties(tier="aggregation")
                    G.add_edge(agg, edge, **link_props)
        
        # Edge to server connections
        servers = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'server']
        for server in servers:
            pod = G.nodes[server].get('pod', 0)
            edge_in_pod = [n for n in edge_switches if G.nodes[n].get('pod') == pod]
            if edge_in_pod:
                edge_node = np.random.choice(edge_in_pod)
                link_props = self._generate_link_properties(tier="edge")
                G.add_edge(edge_node, server, **link_props)
    
    def _generate_small_world_topology(self) -> nx.Graph:
        """Generate a small-world topology using Watts-Strogatz model"""
        # Start with a ring where each node connects to k nearest neighbors
        k = max(2, int(self.config.connectivity * 6))
        p = 0.3  # Rewiring probability
        
        G = nx.watts_strogatz_graph(self.config.num_nodes, k, p, seed=42)
        
        # Add properties
        for u, v in G.edges():
            link_props = self._generate_link_properties()
            G.edges[u, v].update(link_props)
        
        for node in G.nodes():
            G.nodes[node]['node_type'] = "router"
        
        self.logger.info(f"Generated small-world topology with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges")
        return G
    
    def _generate_scale_free_topology(self) -> nx.Graph:
        """Generate a scale-free topology using Barabási-Albert model"""
        m = max(1, int(self.config.connectivity * 3))  # Number of edges to attach
        
        G = nx.barabasi_albert_graph(self.config.num_nodes, m, seed=42)
        
        # Add properties
        for u, v in G.edges():
            link_props = self._generate_link_properties()
            G.edges[u, v].update(link_props)
        
        # Assign node types based on degree
        degrees = dict(G.degree())
        max_degree = max(degrees.values())
        
        for node in G.nodes():
            degree = degrees[node]
            if degree >= max_degree * 0.7:
                node_type = "core"
            elif degree >= max_degree * 0.3:
                node_type = "aggregation"
            else:
                node_type = "edge"
            G.nodes[node]['node_type'] = node_type
        
        self.logger.info(f"Generated scale-free topology with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges")
        return G
    
    def _generate_datacenter_topology(self) -> nx.Graph:
        """Generate a realistic data center topology"""
        # Create a hierarchical structure: core -> aggregation -> edge -> servers
        G = nx.Graph()
        node_id = 0
        
        # Core layer (2-4 nodes)
        num_core = max(2, self.config.num_nodes // 8)
        core_nodes = []
        for i in range(num_core):
            G.add_node(node_id, node_type="core", layer="core")
            core_nodes.append(node_id)
            node_id += 1
        
        # Aggregation layer
        num_agg = max(4, self.config.num_nodes // 4)
        agg_nodes = []
        for i in range(num_agg):
            G.add_node(node_id, node_type="aggregation", layer="aggregation")
            agg_nodes.append(node_id)
            node_id += 1
        
        # Edge layer
        num_edge = max(6, self.config.num_nodes // 2)
        edge_nodes = []
        for i in range(min(num_edge, self.config.num_nodes - node_id)):
            G.add_node(node_id, node_type="edge", layer="edge")
            edge_nodes.append(node_id)
            node_id += 1
        
        # Add remaining nodes as servers
        while node_id < self.config.num_nodes:
            G.add_node(node_id, node_type="server", layer="server")
            node_id += 1
        
        # Add connections
        self._add_datacenter_connections(G, core_nodes, agg_nodes, edge_nodes)
        
        self.logger.info(f"Generated datacenter topology with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges")
        return G
    
    def _add_datacenter_connections(self, G: nx.Graph, core_nodes: List[int], 
                                  agg_nodes: List[int], edge_nodes: List[int]):
        """Add connections for datacenter topology"""
        # Full mesh in core layer
        for i in range(len(core_nodes)):
            for j in range(i + 1, len(core_nodes)):
                link_props = self._generate_link_properties(tier="core")
                G.add_edge(core_nodes[i], core_nodes[j], **link_props)
        
        # Core to aggregation (each agg connects to all core)
        for agg in agg_nodes:
            for core in core_nodes:
                link_props = self._generate_link_properties(tier="core")
                G.add_edge(core, agg, **link_props)
        
        # Aggregation to edge (each edge connects to 2 agg for redundancy)
        for edge in edge_nodes:
            connected_agg = np.random.choice(agg_nodes, size=min(2, len(agg_nodes)), replace=False)
            for agg in connected_agg:
                link_props = self._generate_link_properties(tier="aggregation")
                G.add_edge(agg, edge, **link_props)
        
        # Edge to servers
        servers = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'server']
        for server in servers:
            edge = np.random.choice(edge_nodes)
            link_props = self._generate_link_properties(tier="edge")
            G.add_edge(edge, server, **link_props)
    
    def _generate_link_properties(self, tier: str = "default") -> Dict:
        """Generate realistic link properties based on network tier"""
        if tier == "core":
            # High-capacity core links
            bandwidth = np.random.uniform(1000, 10000)  # 1-10 Gbps
            latency = np.random.uniform(0.1, 2.0)  # Very low latency
            cost = 1
        elif tier == "aggregation":
            # Medium-capacity aggregation links
            bandwidth = np.random.uniform(100, 1000)  # 100 Mbps - 1 Gbps
            latency = np.random.uniform(1.0, 5.0)
            cost = 2
        elif tier == "edge":
            # Lower-capacity edge links
            bandwidth = np.random.uniform(10, 100)  # 10-100 Mbps
            latency = np.random.uniform(2.0, 10.0)
            cost = 5
        else:
            # Default random links
            bandwidth = np.random.uniform(
                self.config.min_bandwidth, 
                self.config.max_bandwidth
            )
            latency = np.random.uniform(
                self.config.min_latency, 
                self.config.max_latency
            )
            cost = np.random.randint(1, 10)
        
        reliability = np.random.uniform(0.95, 0.999)
        
        return {
            'bandwidth': bandwidth,
            'latency': latency,
            'cost': cost,
            'reliability': reliability,
            'weight': latency  # Use latency as default weight for shortest path
        }
    
    def _ensure_connectivity(self, G: nx.Graph) -> nx.Graph:
        """Ensure the graph is connected by adding minimum edges"""
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            
            # Connect components
            for i in range(len(components) - 1):
                # Connect largest nodes from adjacent components
                comp1 = list(components[i])
                comp2 = list(components[i + 1])
                
                node1 = max(comp1, key=lambda n: G.degree[n]) if comp1 else comp1[0]
                node2 = max(comp2, key=lambda n: G.degree[n]) if comp2 else comp2[0]
                
                link_props = self._generate_link_properties()
                G.add_edge(node1, node2, **link_props)
                
                self.logger.info(f"Added connectivity edge between {node1} and {node2}")
        
        return G
    
    def save_topology(self, G: nx.Graph, filename: str):
        """Save topology to file"""
        import pickle
        from pathlib import Path
        
        filepath = Path("data/topologies") / f"{filename}.pkl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(G, f)
        
        self.logger.info(f"Saved topology to {filepath}")
    
    def load_topology(self, filename: str) -> nx.Graph:
        """Load topology from file"""
        import pickle
        from pathlib import Path
        
        filepath = Path("data/topologies") / f"{filename}.pkl"
        
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
        
        self.logger.info(f"Loaded topology from {filepath}")
        return G
    
    def get_topology_stats(self, G: nx.Graph) -> Dict:
        """Get statistics about the topology"""
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
            'avg_clustering': nx.average_clustering(G),
            'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
        }
        
        # Degree statistics
        degrees = [G.degree[n] for n in G.nodes()]
        stats.update({
            'avg_degree': np.mean(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'degree_std': np.std(degrees)
        })
        
        # Node type distribution
        node_types = [G.nodes[n].get('node_type', 'unknown') for n in G.nodes()]
        stats['node_type_counts'] = {
            node_type: node_types.count(node_type) 
            for node_type in set(node_types)
        }
        
        return stats