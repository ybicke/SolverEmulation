import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict, Optional, Union

from triangle_files.triangleGraph import AtmosphericGraph, get_atmospheric_graph


class GraphVisualizer:
    """
    A class for visualizing 3D atmospheric graphs with both horizontal and vertical connections.
    Provides methods for 2D and 3D visualization.
    """
    
    def __init__(self, graph: AtmosphericGraph):
        """
        Initialize the visualizer with an atmospheric graph.
        
        Args:
            graph: An AtmosphericGraph instance
        """
        self.graph = graph
        
        # Ensure graph data is generated
        self.edge_index = self.graph.get_edge_index()
        self.triangle_indices = self.graph.get_triangle_indices()
        self.num_columns = self.graph.num_columns
        self.num_height_levels = self.graph.num_height_levels
        
        # Get centroid coordinates for visualization
        self.clon, self.clat = self.graph.get_centroids()
        
        # Convert to cartesian coordinates for 3D visualization
        self.x, self.y, self.z = self._spherical_to_cartesian(self.clon, self.clat)
    
    def _spherical_to_cartesian(self, lon: torch.Tensor, lat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates (longitude, latitude) to cartesian (x, y, z).
        
        Args:
            lon: Longitude values in radians
            lat: Latitude values in radians
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        # Convert to numpy for calculations
        lon_np = lon.numpy()
        lat_np = lat.numpy()
        
        # Simple conversion to unit sphere
        x = np.cos(lat_np) * np.cos(lon_np)
        y = np.cos(lat_np) * np.sin(lon_np)
        z = np.sin(lat_np)
        
        return x, y, z
    
    def plot_2d_horizontal(self, height_level: int = 0, figsize: Tuple[int, int] = (10, 8), 
                          node_size: int = 30, edge_width: float = 0.5):
        """
        Plot a 2D visualization of the graph at a specific height level.
        
        Args:
            height_level: The height level to visualize
            figsize: Figure size for the plot
            node_size: Size of the nodes in the plot
            edge_width: Width of the edges in the plot
        """
        if height_level >= self.num_height_levels:
            raise ValueError(f"Height level {height_level} exceeds maximum of {self.num_height_levels-1}")
        
        # Create a figure
        plt.figure(figsize=figsize)
        
        # Plot nodes
        for col_idx in range(self.num_columns):
            node_idx = col_idx * self.num_height_levels + height_level
            plt.scatter(self.clon[col_idx], self.clat[col_idx], s=node_size, c='blue')
            
            # Add node index text
            plt.text(self.clon[col_idx], self.clat[col_idx], f"{node_idx}", fontsize=8)
        
        # Get horizontal edges for the current height level
        horizontal_edges = []
        for edge_idx in range(self.edge_index.shape[1]):
            source = self.edge_index[0, edge_idx].item()
            target = self.edge_index[1, edge_idx].item()
            
            # Check if both nodes are at the current height level
            source_height = source % self.num_height_levels
            target_height = target % self.num_height_levels
            
            if source_height == height_level and target_height == height_level:
                source_col = source // self.num_height_levels
                target_col = target // self.num_height_levels
                horizontal_edges.append((source_col, target_col))
        
        # Plot horizontal edges
        for source_col, target_col in horizontal_edges:
            plt.plot([self.clon[source_col], self.clon[target_col]], 
                     [self.clat[source_col], self.clat[target_col]], 
                     'k-', linewidth=edge_width, alpha=0.6)
        
        plt.title(f"2D Graph Visualization at Height Level {height_level}")
        plt.xlabel("Longitude (radians)")
        plt.ylabel("Latitude (radians)")
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_3d_interactive(self, height_scale: float = 0.2, show_horizontal: bool = True, 
                           show_vertical: bool = True, node_size: int = 5):
        """
        Create an interactive 3D visualization of the graph using Plotly.
        
        Args:
            height_scale: Scale factor for height levels
            show_horizontal: Whether to show horizontal edges
            show_vertical: Whether to show vertical edges
            node_size: Size of nodes in the plot
            
        Returns:
            Plotly figure object
        """
        # Create node positions for all nodes
        node_xs = []
        node_ys = []
        node_zs = []
        node_texts = []
        
        # Define node positions across all height levels
        for col_idx in range(self.num_columns):
            for height_level in range(self.num_height_levels):
                node_idx = col_idx * self.num_height_levels + height_level
                
                # Use same x,y across heights, but scale z by height level
                node_xs.append(self.x[col_idx])
                node_ys.append(self.y[col_idx])
                node_zs.append(self.z[col_idx] + height_level * height_scale)
                
                node_texts.append(f"Node {node_idx} (col: {col_idx}, height: {height_level})")
        
        # Create figure
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=node_xs,
            y=node_ys,
            z=node_zs,
            mode='markers',
            marker=dict(
                size=node_size,
                color='blue',
                opacity=0.8
            ),
            text=node_texts,
            hoverinfo='text'
        ))
        
        # Process edges
        edge_xs = []
        edge_ys = []
        edge_zs = []
        
        # To avoid duplicate visualizations, keep track of processed edges
        processed_edges = set()
        
        for edge_idx in range(self.edge_index.shape[1]):
            source = self.edge_index[0, edge_idx].item()
            target = self.edge_index[1, edge_idx].item()
            
            # Skip if we've already processed this edge
            if (source, target) in processed_edges or (target, source) in processed_edges:
                continue
            
            processed_edges.add((source, target))
            
            # Extract column and height indices
            source_col = source // self.num_height_levels
            source_height = source % self.num_height_levels
            target_col = target // self.num_height_levels
            target_height = target % self.num_height_levels
            
            # Check if it's a horizontal or vertical edge
            is_horizontal = (source_height == target_height)
            
            if (is_horizontal and show_horizontal) or (not is_horizontal and show_vertical):
                # Get source node coordinates
                source_x = self.x[source_col]
                source_y = self.y[source_col]
                source_z = self.z[source_col] + source_height * height_scale
                
                # Get target node coordinates
                target_x = self.x[target_col]
                target_y = self.y[target_col]
                target_z = self.z[target_col] + target_height * height_scale
                
                # Add to edge lists
                edge_xs.extend([source_x, target_x, None])
                edge_ys.extend([source_y, target_y, None])
                edge_zs.extend([source_z, target_z, None])
        
        # Add edges
        fig.add_trace(go.Scatter3d(
            x=edge_xs,
            y=edge_ys,
            z=edge_zs,
            mode='lines',
            line=dict(
                color='black',
                width=2
            ),
            hoverinfo='none'
        ))
        
        # Set layout
        fig.update_layout(
            title="3D Atmospheric Graph Visualization",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z (Height)",
                aspectmode='data'
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def verify_graph_structure(self) -> Dict:
        """
        Run verification checks on the graph structure.
        
        Returns:
            Dictionary with verification results
        """
        total_nodes = self.num_columns * self.num_height_levels
        expected_horizontal_connections = self.num_columns * 3 * self.num_height_levels  # Each node has ~3 neighbors
        expected_vertical_connections = self.num_columns * (self.num_height_levels - 1) * 2  # Bidirectional
        
        # Count actual edge types
        horizontal_edges = 0
        vertical_edges = 0
        
        for edge_idx in range(self.edge_index.shape[1]):
            source = self.edge_index[0, edge_idx].item()
            target = self.edge_index[1, edge_idx].item()
            
            source_height = source % self.num_height_levels
            target_height = target % self.num_height_levels
            
            if source_height == target_height:
                horizontal_edges += 1
            else:
                vertical_edges += 1
        
        return {
            "total_nodes": total_nodes,
            "total_edges": self.edge_index.shape[1],
            "horizontal_edges": horizontal_edges,
            "vertical_edges": vertical_edges,
            "expected_horizontal": expected_horizontal_connections,
            "expected_vertical": expected_vertical_connections,
            "is_valid": total_nodes > 0 and self.edge_index.shape[1] > 0
        }


def visualize_graph(
    grid_file_path: str,
    triangle_id: int = 1,
    num_height_levels: int = 70,
    device: str = 'cuda'
) -> GraphVisualizer:
    """
    Convenience function to create and return a graph visualizer.
    
    Args:
        grid_file_path: Path to the ICON grid file
        triangle_id: ID of the triangle region to use
        num_height_levels: Number of vertical height levels
        device: Computation device ('cuda' or 'cpu')
        
    Returns:
        GraphVisualizer instance
    """
    graph = get_atmospheric_graph(
        grid_file_path=grid_file_path,
        triangle_id=triangle_id,
        num_height_levels=num_height_levels,
        device=device
    )
    
    return GraphVisualizer(graph) 