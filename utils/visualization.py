"""
Visualization utilities for Network Traffic Manager
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from config.config import Config

# Optional imports with fallbacks
try:
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    SEABORN_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    SEABORN_AVAILABLE = False

try:
    import matplotlib.animation as animation
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False


class NetworkVisualizer:
    """Interactive network visualization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style if seaborn is available
        if SEABORN_AVAILABLE:
            try:
                plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'seaborn')
                sns.set_palette("husl")
            except:
                pass  # Use default style if seaborn style fails
    
    def plot_network_topology(self, network: nx.Graph, 
                             link_utilizations: Optional[Dict] = None,
                             active_flows: Optional[Dict] = None,
                             save_path: Optional[str] = None):
        """Create network topology visualization"""
        
        if PLOTLY_AVAILABLE:
            return self._plot_with_plotly(network, link_utilizations, active_flows, save_path)
        else:
            return self._plot_with_matplotlib(network, link_utilizations, active_flows, save_path)
    
    def _plot_with_plotly(self, network: nx.Graph, link_utilizations, active_flows, save_path):
        """Create interactive network topology with Plotly"""
        # Create layout
        pos = nx.spring_layout(network, k=3, iterations=50, seed=42)
        
        # Extract node and edge information
        node_trace, edge_traces = self._create_network_traces(
            network, pos, link_utilizations, active_flows
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        fig.update_layout(
            title="Network Topology",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ 
                dict(
                    text="Node size = degree, Edge color = utilization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Network topology saved to {save_path}")
        
        return fig
    
    def _plot_with_matplotlib(self, network: nx.Graph, link_utilizations, active_flows, save_path):
        """Create network topology with matplotlib (fallback)"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(network, k=3, iterations=50, seed=42)
        
        # Draw network
        nx.draw_networkx_nodes(network, pos, node_color='lightblue', 
                              node_size=500, alpha=0.7)
        nx.draw_networkx_edges(network, pos, edge_color='gray', width=1, alpha=0.5)
        nx.draw_networkx_labels(network, pos, font_size=10)
        
        plt.title("Network Topology")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path.replace('.html', '.png'))
            self.logger.info(f"Network topology saved to {save_path}")
        
        return plt.gcf()
    
    def _create_network_traces(self, network: nx.Graph, pos: Dict,
                              link_utilizations: Optional[Dict] = None,
                              active_flows: Optional[Dict] = None):
        """Create Plotly traces for network visualization"""
        
        # Edge traces
        edge_traces = []
        
        for edge in network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Get edge utilization
            utilization = 0.0
            if link_utilizations:
                utilization = link_utilizations.get(edge, 0.0)
            
            # Color based on utilization
            color = self._get_utilization_color(utilization)
            width = 1 + utilization * 4  # Line width based on utilization
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=width, color=color),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in network.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node size based on degree
            degree = network.degree[node]
            node_size.append(10 + degree * 2)
            
            # Node color based on type
            node_type = network.nodes[node].get('node_type', 'router')
            node_color.append(self._get_node_color(node_type))
            
            # Node info
            node_info = f"Node {node}<br>Type: {node_type}<br>Degree: {degree}"
            if active_flows:
                flows_through = sum(1 for flow in active_flows.values() 
                                  if flow.get('current_path') and node in flow['current_path'])
                node_info += f"<br>Active flows: {flows_through}"
            
            node_text.append(node_info)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(i) for i in range(len(node_x))],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=2, color='black')
            )
        )
        
        return node_trace, edge_traces
    
    def _get_utilization_color(self, utilization: float) -> str:
        """Get color based on link utilization"""
        if utilization < 0.3:
            return 'green'
        elif utilization < 0.7:
            return 'orange'
        else:
            return 'red'
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color based on node type"""
        color_map = {
            'core': 'red',
            'aggregation': 'orange', 
            'edge': 'blue',
            'server': 'green',
            'router': 'lightblue'
        }
        return color_map.get(node_type, 'lightblue')


class MetricsVisualizer:
    """Performance metrics visualization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def plot_training_progress(self, training_stats: Dict[str, Any], 
                              save_path: Optional[str] = None):
        """Plot training progress dashboard"""
        
        if PLOTLY_AVAILABLE:
            return self._plot_training_plotly(training_stats, save_path)
        else:
            return self._plot_training_matplotlib(training_stats, save_path)
    
    def _plot_training_plotly(self, training_stats: Dict[str, Any], save_path: Optional[str] = None):
        """Plot with Plotly (full featured)"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Training Losses', 
                           'Network Performance', 'Learning Progress'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Episode rewards
        if 'episode_rewards' in training_stats:
            rewards = training_stats['episode_rewards']
            episodes = list(range(len(rewards)))
            
            # Moving average
            window = min(50, len(rewards) // 10)
            if window > 1:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                moving_episodes = episodes[window-1:]
                
                fig.add_trace(
                    go.Scatter(x=moving_episodes, y=moving_avg, name='Moving Average',
                             line=dict(color='red', width=3)),
                    row=1, col=1
                )
            
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, name='Episode Rewards',
                          mode='lines', opacity=0.6),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Training Progress Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Training dashboard saved to {save_path}")
        
        return fig
    
    def _plot_training_matplotlib(self, training_stats: Dict[str, Any], save_path: Optional[str] = None):
        """Plot with matplotlib (fallback)"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training Progress Dashboard")
        
        # Episode rewards
        if 'episode_rewards' in training_stats:
            rewards = training_stats['episode_rewards']
            episodes = list(range(len(rewards)))
            
            axes[0, 0].plot(episodes, rewards, alpha=0.6, label='Episode Rewards')
            
            # Moving average
            window = min(50, len(rewards) // 10)
            if window > 1:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                moving_episodes = episodes[window-1:]
                axes[0, 0].plot(moving_episodes, moving_avg, color='red', linewidth=2, label='Moving Average')
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
        
        # Training losses
        if 'training_losses' in training_stats:
            losses = training_stats['training_losses']
            for loss_name, loss_values in losses.items():
                if loss_values:
                    steps = list(range(len(loss_values)))
                    axes[0, 1].plot(steps, loss_values, label=loss_name)
            
            axes[0, 1].set_title('Training Losses')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
        
        # Evaluation rewards
        if 'evaluation_rewards' in training_stats:
            eval_rewards = training_stats['evaluation_rewards']
            eval_steps = training_stats.get('timestamps', list(range(len(eval_rewards))))
            
            axes[1, 0].plot(eval_steps, eval_rewards, 'go-', linewidth=2, label='Evaluation Reward')
            axes[1, 0].set_title('Learning Progress')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Evaluation Reward')
            axes[1, 0].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.html', '.png'))
            self.logger.info(f"Training dashboard saved to {save_path}")
        
        return fig


def create_training_dashboard(training_stats: Dict[str, Any], config: Config) -> str:
    """Create comprehensive training dashboard"""
    
    try:
        visualizer = MetricsVisualizer(config)
        
        # Create dashboard
        dashboard = visualizer.plot_training_progress(training_stats)
        
        # Save dashboard
        output_dir = Path(config.experiment.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if PLOTLY_AVAILABLE:
            dashboard_path = output_dir / f"training_dashboard_{config.rl.algorithm.lower()}.html"
            dashboard.write_html(str(dashboard_path))
        else:
            dashboard_path = output_dir / f"training_dashboard_{config.rl.algorithm.lower()}.png"
            dashboard.savefig(str(dashboard_path))
            plt.close(dashboard)
        
        return str(dashboard_path)
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not create training dashboard: {e}")
        return "dashboard_creation_failed"


def create_evaluation_plots(evaluation_results: Dict[str, Any], config: Config) -> List[str]:
    """Create evaluation visualization plots"""
    return ["evaluation_plot_placeholder"]


def save_network_visualization(network_state: Dict[str, Any], config: Config, 
                              filename: str = "network_state") -> str:
    """Save current network state visualization"""
    try:
        visualizer = NetworkVisualizer(config)
        
        output_dir = Path(config.experiment.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract network information
        network = network_state['network']
        links = network_state.get('links', {})
        active_flows = network_state.get('active_flows', {})
        
        # Convert links to utilizations
        link_utilizations = {}
        for (u, v), link_state in links.items():
            if hasattr(link_state, 'utilization'):
                link_utilizations[(u, v)] = link_state.utilization
        
        # Create visualization
        viz_path = output_dir / f"{filename}.html"
        fig = visualizer.plot_network_topology(network, link_utilizations, active_flows, str(viz_path))
        
        return str(viz_path)
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not create network visualization: {e}")
        return "network_viz_failed"