import numpy as np
import torch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os

def load_heights(file_path):
    """Load height data from NetCDF file"""
    try:
        with Dataset(file_path, 'r') as fh:
            if 'z_ifc' not in fh.variables:
                raise ValueError("Variable 'z_ifc' not found in dataset!")
            
            # Extract heights (first column - heights are the same across all coordinates)
            heights = fh.variables['z_ifc'][:, 0, 0]
            return heights
    except Exception as e:
        print(f"Error loading heights: {e}")
        return None

def generate_edge_features(heights, max_skip=3):
    """
    Generate edge features based on height differences for different skip connections.
    
    Args:
        heights (ndarray): Height values for each level
        max_skip (int): Maximum skip distance to consider
    
    Returns:
        dict: Dictionary with edge features for different skip distances
    """
    # Convert heights to tensor if they're not already
    if not isinstance(heights, torch.Tensor):
        heights = torch.tensor(heights, dtype=torch.float32)
    
    # Store edge features for each skip distance
    edge_features = {}
    
    # For analysis and visualization
    all_diffs = []
    
    print(f"Generating edge features for {len(heights)} height levels")
    print(f"Height range: {heights.min().item():.2f} m to {heights.max().item():.2f} m")
    
    # Process each skip distance
    for skip in range(1, max_skip + 1):
        # For skip=1, calculate differences between adjacent levels
        # For skip=2, calculate differences between levels with one level in between, etc.
        src_indices = torch.arange(len(heights) - skip)
        dst_indices = torch.arange(skip, len(heights))
        
        # Calculate height differences for this skip distance
        diffs = heights[dst_indices] - heights[src_indices]
        
        # Store in dictionary
        edge_features[skip] = diffs
        all_diffs.append(diffs)
        
        # Print statistics
        print(f"\nSkip distance {skip}:")
        print(f"  Number of edges: {len(diffs)}")
        print(f"  Min height difference: {diffs.min().item():.2f} m")
        print(f"  Max height difference: {diffs.max().item():.2f} m")
        print(f"  Mean height difference: {diffs.mean().item():.2f} m")
        
        # Print some sample edges
        print(f"  Sample edges (first 3):")
        for i in range(min(3, len(diffs))):
            print(f"    Level {src_indices[i]} → Level {dst_indices[i]}: {diffs[i].item():.2f} m")
    
    # Visualize the edge features
    plt.figure(figsize=(12, 10))
    
    for skip, diffs in edge_features.items():
        plt.subplot(max_skip, 1, skip)
        plt.plot(diffs.numpy(), marker='.', linestyle='-', label=f"Skip = {skip}")
        plt.title(f'Height Differences (Skip = {skip})')
        plt.xlabel('Edge Index')
        plt.ylabel('Height Difference (m)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('height_edge_features.png')
    print(f"\nPlot saved as 'height_edge_features.png'")
    
    return edge_features

def create_edge_indices_and_features(heights, max_skip=3, batch_size=1):
    """
    Create edge indices and corresponding edge features for the GNN.
    
    Args:
        heights (ndarray): Height values for each level
        max_skip (int): Maximum skip distance
        batch_size (int): Number of batches
        
    Returns:
        tuple: (edge_index, edge_attr) for use in the GNN
    """
    # Convert heights to tensor if they're not already
    if not isinstance(heights, torch.Tensor):
        heights = torch.tensor(heights, dtype=torch.float32)
    
    num_nodes = len(heights)
    all_edges_src = []
    all_edges_dst = []
    all_edge_features = []
    
    print(f"Creating edge indices and features for {num_nodes} nodes, max_skip={max_skip}")
    
    # For each skip distance
    for skip in range(1, max_skip + 1):
        # Source nodes for forward connections (i → i+skip)
        src_forward = torch.arange(num_nodes - skip)
        dst_forward = torch.arange(skip, num_nodes)
        
        # Source nodes for backward connections (i+skip → i)
        src_backward = torch.arange(skip, num_nodes)
        dst_backward = torch.arange(num_nodes - skip)
        
        # Calculate height differences (as features)
        forward_diffs = heights[dst_forward] - heights[src_forward]
        backward_diffs = heights[dst_backward] - heights[src_backward]
        
        # Add to our lists
        all_edges_src.append(src_forward)
        all_edges_dst.append(dst_forward)
        all_edge_features.append(forward_diffs.unsqueeze(1))  # Add dimension for feature channel
        
        all_edges_src.append(src_backward)
        all_edges_dst.append(dst_backward)
        all_edge_features.append(backward_diffs.unsqueeze(1))  # Add dimension for feature channel
    
    # Concatenate all edges
    src_indices = torch.cat(all_edges_src)
    dst_indices = torch.cat(all_edges_dst)
    edge_features = torch.cat(all_edge_features)
    
    # Create the edge_index tensor [2, num_edges]
    edge_index = torch.stack([src_indices, dst_indices])
    
    # If we have multiple batches, replicate and offset
    if batch_size > 1:
        # Replicate edges for each batch
        edge_index = edge_index.repeat(1, batch_size)
        edge_features = edge_features.repeat(batch_size, 1)
        
        # Create offsets for each batch
        batch_offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        batch_offsets = batch_offsets.repeat_interleave(edge_index.shape[1] // batch_size)
        
        # Apply offsets
        edge_index = edge_index + batch_offsets.view(1, -1)
    
    print(f"Created edge_index with shape {edge_index.shape} and edge_features with shape {edge_features.shape}")
    
    # Normalize edge features to a reasonable range
    edge_features = edge_features / heights.max()
    
    return edge_index, edge_features

def visualize_graph(heights, edge_index, num_nodes_to_show=10):
    """Visualize a small portion of the graph to verify connections"""
    import networkx as nx
    
    # Create a graph using only the first few nodes for visualization
    G = nx.DiGraph()
    
    # Add nodes with their heights as attributes
    for i in range(min(num_nodes_to_show, len(heights))):
        G.add_node(i, height=heights[i].item())
    
    # Add edges from the edge_index
    src, dst = edge_index
    for i in range(len(src)):
        s, d = src[i].item(), dst[i].item()
        if s < num_nodes_to_show and d < num_nodes_to_show:
            diff = heights[d].item() - heights[s].item()
            G.add_edge(s, d, weight=diff)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Position nodes vertically based on their height
    pos = {i: (i*0.2, heights[i].item()/5000) for i in G.nodes()}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, arrows=True)
    
    # Add edge labels (height differences)
    edge_labels = {(s, d): f"{G[s][d]['weight']:.0f}m" for s, d in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Visualization of Graph Structure with Height Differences")
    plt.tight_layout()
    plt.savefig('graph_visualization.png')
    print(f"Graph visualization saved as 'graph_visualization.png'")

def main():
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path, update as needed
        file_path = "/mydata/deepcloud/yves/SolverEmulation/data_exploration/ml_ecrad_ape_R2B05_myrunscript_ecRad5d_infero5d_70lev_atm_3d_ICONGRID_DOM01_ml_lonlat.nc"
    
    # Load heights
    heights = load_heights(file_path)
    
    if heights is None:
        print("Failed to load heights. Exiting.")
        return
    
    # Generate edge features
    edge_features = generate_edge_features(heights, max_skip=3)
    
    # Create edge indices and features ready for GNN
    edge_index, edge_attr = create_edge_indices_and_features(heights, max_skip=3, batch_size=1)
    
    # Visualize a small portion of the graph
    try:
        visualize_graph(torch.tensor(heights), edge_index)
    except ImportError:
        print("NetworkX not available for visualization. Skipping graph visualization.")
    
    # Save the edge features and indices for later use
    output_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save({
        'heights': torch.tensor(heights, dtype=torch.float32),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
    }, os.path.join(output_dir, 'height_edge_data.pt'))
    
    print(f"Height edge data saved to {os.path.join(output_dir, 'height_edge_data.pt')}")

if __name__ == "__main__":
    main() 