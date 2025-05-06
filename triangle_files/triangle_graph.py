import networkx as nx
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Function to get triangle indices (Original - Correct for your goal) ---
def get_triangle_indices(triangle_id=1, total_cols=81920):
    block_size = total_cols // (20*4*4*4*4)
    start_idx = triangle_id * block_size
    end_idx   = min(start_idx + block_size, total_cols)
    return torch.arange(start_idx, end_idx, dtype=torch.long)

# --- 3. Graph Creation Function (Corrected and Streamlined) ---
def create_atmospheric_graph_from_xarray(grid_data, triangle_indices, num_height_levels=70, device='cuda'):
    """
    Creates a graph structure for the atmospheric data and returns edge_index directly.
    """
    num_columns = len(triangle_indices)
    node_ids = {}  # (local_col_idx, height_level) -> node_index (now just an index)
    
    vertical_edges_list = []
    horizontal_edges_list = []

    # --- 1. Create ALL nodes and Node IDs (implicitly indexed) ---
    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            node_index = local_col_idx * num_height_levels + height_level 
            node_ids[(local_col_idx, height_level)] = node_index

    # --- 2. Add Horizontal Edges (within each height level) ---
    neighbor_indices_global = grid_data['neighbor_cell_index'].values  # (3, 81920)
    triangle_neighbor_indices = neighbor_indices_global[:, triangle_indices.numpy()]  # (3, num_columns)

    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            current_node_index = node_ids[(local_col_idx, height_level)]
            neighbors = triangle_neighbor_indices[:, local_col_idx]

            for neighbor_global in neighbors:
                local_neighbor_idx_array = np.where(triangle_indices.numpy() == neighbor_global - 1)[0] # Returns array

                if len(local_neighbor_idx_array) > 0:
                    local_neighbor_idx = local_neighbor_idx_array[0] # Take the first element
                    neighbor_node_index = node_ids[(local_neighbor_idx, height_level)]
                    horizontal_edges_list.append([current_node_index, neighbor_node_index]) # [source, target]
                    horizontal_edges_list.append([neighbor_node_index, current_node_index]) # Bidirectional

    # --- 3. Add Vertical Edges (connecting nodes at the same horizontal location across height levels) ---
    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            current_node_index = node_ids[(local_col_idx, height_level)]
            if height_level > 0:
                lower_node_index = node_ids[(local_col_idx, height_level - 1)]
                vertical_edges_list.append([current_node_index, lower_node_index]) # Vertical downward edge
                vertical_edges_list.append([lower_node_index, current_node_index]) # Vertical upward edge (bidirectional)

    # 4. Combine edges and convert to PyTorch Tensor
    edge_index_numpy = np.array(vertical_edges_list + horizontal_edges_list).T # Transpose to get shape (2, num_edges)
    edge_index = torch.tensor(edge_index_numpy, dtype=torch.long, device=device)
    

    return edge_index, node_ids, horizontal_edges_list


def create_edge_index(grid_file_path, triangle_id=1, total_cols=81920, num_height_levels=70, device='cuda'):
    """
    Creates edge_index tensor directly for PyTorch Geometric.
    Opens and closes the grid dataset internally.
    """
    grid_ds = xr.open_dataset(grid_file_path)
    triangle_indices = get_triangle_indices(triangle_id, total_cols)
    edge_index, _, _ = create_atmospheric_graph_from_xarray(
        grid_ds,
        triangle_indices,
        num_height_levels,
        device
    )
    grid_ds.close() # Close dataset after use
    return edge_index


def get_centroids(grid_file_path, triangle_id=1, total_cols=81920):
    """
    Returns the centroid coordinates for the given triangle indices.
    Opens and closes the grid dataset internally.
    """
    grid_ds = xr.open_dataset(grid_file_path)
    triangle_indices = get_triangle_indices(triangle_id, total_cols)
    clon, clat = get_centroids_from_indices(grid_ds, triangle_indices)
    grid_ds.close() # Close dataset after use
    return clon, clat

def get_centroids_from_indices(grid_data, triangle_indices):
    """
    Returns the centroid coordinates for pre-selected triangle indices from an opened dataset.
    """
    clon = grid_data['clon'].values[triangle_indices.numpy()]
    clat = grid_data['clat'].values[triangle_indices.numpy()]
    return torch.tensor(clon, dtype=torch.float32), torch.tensor(clat, dtype=torch.float32)


def plot_horizontal_graph_edges(grid_data, triangle_indices, horizontal_edges_list, node_ids):
    """
    Plots the horizontal edges of the graph on a map.
     """
    clon, clat = get_centroids_from_indices(grid_data, triangle_indices)
    clon_deg = np.rad2deg(clon.numpy())
    clat_deg = np.rad2deg(clat.numpy())
    num_columns = len(triangle_indices)
    num_height_levels = 1 

    # Create the plot
    plt.figure(figsize=(15, 15))
    ax = plt.axes() 

    plt.grid(True) 

    # Plot cell centroids
    scatter = plt.scatter(clon_deg, clat_deg, c=range(1, num_columns + 1), cmap='viridis', s=100) 
    
    # Plot horizontal edges
    unique_edges = set()
    for edge in horizontal_edges_list:
        src, tgt = edge
        if (src, tgt) not in unique_edges and (tgt, src) not in unique_edges:
            unique_edges.add((src, tgt))

    for src, tgt in unique_edges:
        # For visualization, convert node indices back to column indices (assuming 1 height level)
        src_col = (src) // num_height_levels
        tgt_col = (tgt) // num_height_levels
        
        # Extract longitude and latitude coordinates for source and target nodes
        src_lon, src_lat = clon_deg[src_col], clat_deg[src_col]
        tgt_lon, tgt_lat = clon_deg[tgt_col], clat_deg[tgt_col]
        
        # Plot a line between the source and target node coordinates
        plt.plot([src_lon, tgt_lon], [src_lat, tgt_lat], 
                 'r-', alpha=0.6, linewidth=1)

    # Add labels
    for i in range(num_columns):
        plt.annotate(str(i+1), (clon_deg[i], clat_deg[i]), # Use plt.annotate
                xytext=(5, 5), textcoords='offset points', fontsize=8, ha='center', va='bottom', transform=ccrs.PlateCarree())

    plt.title('Horizontal Edges of Atmospheric Graph (1-based Indexing)')
    plt.colorbar(scatter, label='Cell Index (1-based)')
    plt.xlabel('Longitude (degrees)') # Labels from grid_visualization.ipynb
    plt.ylabel('Latitude (degrees)') # Labels from grid_visualization.ipynb
    plt.savefig('horizontal_edges.png', dpi=300)
    plt.tight_layout()
    plt.show()


def verify_edge_index_matches_edges_list(edge_index, horizontal_edges_list, vertical_edges_list=None):
    """
    Verifies that the edge_index tensor correctly represents the edges in the edge lists.
    
    Parameters:
    -----------
    edge_index : torch.Tensor
        The edge_index tensor of shape (2, num_edges)
    horizontal_edges_list : list
        List of horizontal edges as [src, tgt] pairs
    vertical_edges_list : list, optional
        List of vertical edges as [src, tgt] pairs
        
    Returns:
    --------
    bool
        True if the edge_index correctly represents all edges, False otherwise
    """
    # Convert edge_index to a set of (src, tgt) tuples for comparison
    edge_index_set = set()
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        tgt = edge_index[1, i].item()
        edge_index_set.add((src, tgt))
    
    # Convert the horizontal edges list to a set
    horiz_edges_set = set()
    for edge in horizontal_edges_list:
        src, tgt = edge
        horiz_edges_set.add((src, tgt))
    
    # Check if all horizontal edges are in the edge_index
    horiz_edges_in_index = horiz_edges_set.issubset(edge_index_set)
    print(f"All horizontal edges in edge_index: {horiz_edges_in_index}")
    
    # If vertical edges are provided, check those too
    if vertical_edges_list:
        vert_edges_set = set()
        for edge in vertical_edges_list:
            src, tgt = edge
            vert_edges_set.add((src, tgt))
        
        vert_edges_in_index = vert_edges_set.issubset(edge_index_set)
        print(f"All vertical edges in edge_index: {vert_edges_in_index}")
        
        # Check if the combined set exactly matches
        combined_edges_set = horiz_edges_set.union(vert_edges_set)
        exact_match = combined_edges_set == edge_index_set
        print(f"Exact match between edge lists and edge_index: {exact_match}")
        
        return horiz_edges_in_index and vert_edges_in_index and exact_match
    else:
        # For 2D graph with only horizontal edges
        exact_match = horiz_edges_set == edge_index_set
        print(f"Exact match between horizontal edges and edge_index: {exact_match}")
        print(f"Edge_index size: {len(edge_index_set)}, Horizontal edges size: {len(horiz_edges_set)}")
        
        if not exact_match:
            missing_in_edge_index = horiz_edges_set - edge_index_set
            missing_in_horiz_list = edge_index_set - horiz_edges_set
            
            if missing_in_edge_index:
                print(f"Edges in horizontal_edges_list but missing in edge_index (sample): {list(missing_in_edge_index)[:5]}")
            if missing_in_horiz_list:
                print(f"Edges in edge_index but missing in horizontal_edges_list (sample): {list(missing_in_horiz_list)[:5]}")
        
        return horiz_edges_in_index and exact_match


def visualize_edge_index(grid_data, triangle_indices, edge_index, num_height_levels=1):
    """
    Directly visualizes the edge_index tensor from PyTorch Geometric.
    This shows the exact connectivity pattern that will be used in the model.
    
    Parameters:
    -----------
    grid_data : xarray.Dataset
        The grid data containing centroids
    triangle_indices : torch.Tensor
        The indices of triangles to plot
    edge_index : torch.Tensor
        The edge_index tensor of shape (2, num_edges) from PyTorch Geometric
    num_height_levels : int
        Number of height levels in the graph
    """
    # Get centroids for plotting
    clon, clat = get_centroids_from_indices(grid_data, triangle_indices)
    clon_deg = np.rad2deg(clon.numpy())
    clat_deg = np.rad2deg(clat.numpy())
    num_columns = len(triangle_indices)
    
    # Create a new figure
    plt.figure(figsize=(15, 15))
    ax = plt.axes()
    plt.grid(True)
    
    # Plot nodes - colorize by index
    scatter = plt.scatter(clon_deg, clat_deg, c=range(1, num_columns + 1), cmap='viridis', s=100)
    
    # Add node labels
    for i in range(num_columns):
        plt.annotate(str(i+1), (clon_deg[i], clat_deg[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, 
                    ha='center', va='bottom', transform=ccrs.PlateCarree())
    
    # Create a set to track plotted edges and avoid duplicates
    plotted_edges = set()
    
    # Directly plot edges from edge_index
    # Convert to numpy for easier iteration
    edge_index_np = edge_index.cpu().numpy()
    
    for i in range(edge_index_np.shape[1]):
        src = edge_index_np[0, i]
        tgt = edge_index_np[1, i]
        
        # Convert to column indices (if using multiple height levels)
        src_col = src // num_height_levels
        tgt_col = tgt // num_height_levels
        
        # Only plot edges between different columns (horizontal edges)
        # and avoid duplicate edges
        if src_col != tgt_col and (src, tgt) not in plotted_edges and (tgt, src) not in plotted_edges:
            plotted_edges.add((src, tgt))
            
            # Get coordinates
            src_lon, src_lat = clon_deg[src_col], clat_deg[src_col]
            tgt_lon, tgt_lat = clon_deg[tgt_col], clat_deg[tgt_col]
            
            # Plot the edge
            plt.plot([src_lon, tgt_lon], [src_lat, tgt_lat], 
                     'r-', alpha=0.6, linewidth=1)
    
    # Set labels and title
    plt.title('Connectivity Pattern from Edge Index Tensor')
    plt.colorbar(scatter, label='Cell Index (1-based)')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.savefig('edge_index_visualization.png', dpi=300)
    plt.tight_layout()
    plt.show()
    
    print(f"Plotted {len(plotted_edges)} unique edges from edge_index tensor")


def visualize_3d_graph_plotly(grid_data, triangle_indices, edge_index, node_ids, num_height_levels, height_scale=0.01):
    """
    Creates a simple interactive 3D visualization of the atmospheric graph using Plotly.
    This will display directly in Jupyter notebooks or open in a browser outside notebooks.
    
    Parameters:
    -----------
    grid_data : xarray.Dataset
        The grid data containing centroids
    triangle_indices : torch.Tensor
        The indices of triangles to plot
    edge_index : torch.Tensor
        The edge_index tensor from PyTorch Geometric
    node_ids : dict
        Dictionary mapping (column_idx, height_level) to node index
    num_height_levels : int
        Number of height levels in the graph
    height_scale : float
        Scaling factor for the height dimension
    """
    # Get centroids for plotting
    clon, clat = get_centroids_from_indices(grid_data, triangle_indices)
    clon_deg = np.rad2deg(clon.numpy())
    clat_deg = np.rad2deg(clat.numpy())
    num_columns = len(triangle_indices)
    
    # Create figure
    fig = go.Figure()
    
    # Prepare node coordinates
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    
    # For each node in the graph
    for col_idx in range(num_columns):
        for height in range(num_height_levels):
            node_idx = node_ids[(col_idx, height)]
            
            # Node coordinates
            node_x.append(clon_deg[col_idx])
            node_y.append(clat_deg[col_idx])
            node_z.append(height * height_scale)
            
            # Node text for hover
            node_text.append(f"Node {node_idx}<br>Column {col_idx}<br>Height {height}")
            
            # Color by height level
            node_color.append(height)
    
    # Add nodes trace
    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            marker=dict(
                size=5,
                color=node_color,
                colorscale='Viridis',
                colorbar=dict(title='Height Level'),
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text',
            name='Nodes'
        )
    )
    
    # Process edge coordinates
    edge_x, edge_y, edge_z = [], [], []
    horiz_edges_x, horiz_edges_y, horiz_edges_z = [], [], []
    vert_edges_x, vert_edges_y, vert_edges_z = [], [], []
    
    # Convert edge_index to numpy for easier handling
    edge_index_np = edge_index.cpu().numpy()
    
    # Create a set to track plotted edges and avoid duplicates
    plotted_edges = set()
    
    # For each edge in edge_index
    for i in range(edge_index_np.shape[1]):
        src = edge_index_np[0, i]
        tgt = edge_index_np[1, i]
        
        # Skip if we've already plotted this edge (or its reverse)
        if (src, tgt) in plotted_edges or (tgt, src) in plotted_edges:
            continue
        
        plotted_edges.add((src, tgt))
        
        # Get source and target column and height
        src_col = src // num_height_levels
        src_height = src % num_height_levels
        tgt_col = tgt // num_height_levels
        tgt_height = tgt % num_height_levels
        
        # Check if column indices are valid 
        if src_col >= num_columns or tgt_col >= num_columns:
            continue
            
        # Get coordinates
        src_x = clon_deg[src_col]
        src_y = clat_deg[src_col]
        src_z = src_height * height_scale
        
        tgt_x = clon_deg[tgt_col]
        tgt_y = clat_deg[tgt_col]
        tgt_z = tgt_height * height_scale
        
        # Separate horizontal and vertical edges
        if src_height == tgt_height:
            # Horizontal edge - same height level
            horiz_edges_x.extend([src_x, tgt_x, None])
            horiz_edges_y.extend([src_y, tgt_y, None])
            horiz_edges_z.extend([src_z, tgt_z, None])
        else:
            # Vertical edge - between height levels
            vert_edges_x.extend([src_x, tgt_x, None])
            vert_edges_y.extend([src_y, tgt_y, None])
            vert_edges_z.extend([src_z, tgt_z, None])
    
    # Add horizontal edges trace (red)
    fig.add_trace(
        go.Scatter3d(
            x=horiz_edges_x,
            y=horiz_edges_y,
            z=horiz_edges_z,
            mode='lines',
            line=dict(color='red', width=2),
            hoverinfo='none',
            name='Horizontal Edges'
        )
    )
    
    # Add vertical edges trace (blue)
    fig.add_trace(
        go.Scatter3d(
            x=vert_edges_x,
            y=vert_edges_y,
            z=vert_edges_z,
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='none',
            name='Vertical Edges'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='3D Atmospheric Graph Structure',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Height Level (scaled)',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Just show the figure directly (like in the notebook example)
    fig.show()
    
    print(f"Plotted {len(plotted_edges)} unique edges in 3D visualization")
    print(f"Horizontal edges: {len(horiz_edges_x)//3}, Vertical edges: {len(vert_edges_x)//3}")


if __name__ == '__main__':
    grid_file_path = "/mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc" # Replace with your actual path
    grid_ds = xr.open_dataset(grid_file_path)
    triangle_id = 1
    triangle_indices = get_triangle_indices(triangle_id, total_cols=81920)

    # Create a 2D graph with exactly 1 height level - this is specifically for visualization
    # This ensures one-to-one correspondence between graph creation and plotting
    visualization_height_levels = 2
    edge_index, node_ids, horizontal_edges_list = create_atmospheric_graph_from_xarray(
        grid_ds,
        triangle_indices,
        num_height_levels=visualization_height_levels,
        device='cpu' # Use CPU for visualization
    )

    print(f"Created Edge Index shape: {edge_index.shape}")
    print(f"Number of Horizontal Edges: {len(horizontal_edges_list)}")

    # After creating the graph and before plotting
    print("\nVerifying edge_index matches the edge lists:")
    verification_result = verify_edge_index_matches_edges_list(edge_index, horizontal_edges_list)
    print(f"Edge index verification result: {verification_result}")

    # After verification and before the regular plot
    print("\nVisualizing the edge_index tensor directly:")
    visualize_edge_index(grid_ds, triangle_indices, edge_index, visualization_height_levels)
    
    # Regular plot using horizontal_edges_list
    print("\nPlotting using horizontal_edges_list:")
    # plot_horizontal_graph_edges(grid_ds, triangle_indices, horizontal_edges_list, node_ids)

    # Add 3D interactive visualization after the existing plots
    print("\nCreating interactive 3D visualization with Plotly:")
    visualize_3d_graph_plotly(
        grid_ds, 
        triangle_indices, 
        edge_index, 
        node_ids, 
        visualization_height_levels, 
        height_scale=0.1
    )

    grid_ds.close()

    # For actual 3D atmospheric modeling, you would use:
    # full_edge_index, _, _ = create_atmospheric_graph_from_xarray(
    #     grid_ds, triangle_indices, num_height_levels=70, device='cuda'
    # )





