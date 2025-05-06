import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


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
    # graph = nx.Graph() # No longer using NetworkX graph directly
    num_columns = len(triangle_indices)
    # node_id_counter = 0 # Not needed with direct edge_index construction
    node_ids = {}  # (local_col_idx, height_level) -> node_index (now just an index)

    # Initialize lists to build edge_index components
    vertical_edges_list = []
    horizontal_edges_list = []

    # --- 1. Create ALL nodes (implicitly indexed) ---
    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            node_index = local_col_idx * num_height_levels + height_level # Unique index for each node
            node_ids[(local_col_idx, height_level)] = node_index
            # graph.add_node(node_id, features=[], id=node_id) # No NetworkX node addition

    # --- 2. Add Vertical Edges (directly to edge_index) ---
    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            current_node_index = node_ids[(local_col_idx, height_level)]

            # adds vertical downward edges
            if height_level > 0:
                lower_node_index = node_ids[(local_col_idx, height_level - 1)]
                vertical_edges_list.append([current_node_index, lower_node_index]) # [source, target]
                vertical_edges_list.append([lower_node_index, current_node_index]) # Bidirectional

            # adds vertical upward edges (already handled by bidirectional edges above)
            # if height_level < num_height_levels - 1:
            #     upper_node_index = node_ids[(local_col_idx, height_level + 1)]
            #     vertical_edges_list.append([current_node_index, upper_node_index])


    # --- 3. Add Horizontal Edges (with filtering, directly to edge_index) ---
    neighbor_indices_global = grid_data['neighbor_cell_index'].values  # (3, 81920)
    print(f"neighbor_indices_global.shape: {neighbor_indices_global.shape}")
    print(f"neighbor_indices_global: {neighbor_indices_global}")
    
    triangle_neighbor_indices = neighbor_indices_global[:, triangle_indices.numpy()]  # (3, num_columns)
    print(f"triangle_neighbor_indices.shape: {triangle_neighbor_indices.shape}")
    print(f"triangle_neighbor_indices: {triangle_neighbor_indices}")
    
    for local_col_idx in range(num_columns):
        print(f"local_col_idx: {local_col_idx}")
        for height_level in range(num_height_levels):
            print(f"height_level: {height_level}")
            current_node_index = node_ids[(local_col_idx, height_level)]
            print(f"current_node_index: {current_node_index}")
            neighbors = triangle_neighbor_indices[:, local_col_idx]
            print(f"neighbors: {neighbors}")

            for neighbor_global in neighbors:
                print(f"neighbor_global: {neighbor_global}")
                local_neighbor_idx_array = np.where(triangle_indices.numpy() == neighbor_global)[0] # Returns array
                print(f"local_neighbor_idx_array: {local_neighbor_idx_array}")

                if len(local_neighbor_idx_array) > 0:
                    local_neighbor_idx = local_neighbor_idx_array[0] # Take the first element
                    print(f"local_neighbor_idx: {local_neighbor_idx}")
                    neighbor_node_index = node_ids[(local_neighbor_idx, height_level)]
                    print(f"neighbor_node_index: {neighbor_node_index}")
                    horizontal_edges_list.append([current_node_index, neighbor_node_index]) # [source, target]
                    horizontal_edges_list.append([neighbor_node_index, current_node_index]) # Bidirectional


    # 4. Combine edges and convert to PyTorch Tensor
    edge_index_numpy = np.array(vertical_edges_list + horizontal_edges_list).T # Transpose to get shape (2, num_edges)
    edge_index = torch.tensor(edge_index_numpy, dtype=torch.long, device=device)

    # No NetworkX conversion needed anymore
    # data = from_networkx(graph)
    # edge_index = data.edge_index.to(device)

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
    num_height_levels = 1 # Assuming single height level for this plot

    # Create the plot
    plt.figure(figsize=(15, 15))
    ax = plt.axes() # Removed cartopy projection for simplified style

    plt.grid(True) # Use plt.grid for simplified style

    # Plot cell centroids
    scatter = plt.scatter(clon_deg, clat_deg, c=range(1, num_columns + 1), cmap='viridis', s=100) # Removed transform=ccrs.PlateCarree()
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
        plt.plot([clon_deg[src_col], clon_deg[tgt_col]], # Use plt.plot
                [clat_deg[src_col], clat_deg[tgt_col]],
                'r-', alpha=0.6, linewidth=1) # Use style from grid_visualization.ipynb, removed transform=ccrs.PlateCarree()

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


if __name__ == '__main__':
    grid_file_path = "/mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc" # Replace with your actual path
    grid_ds = xr.open_dataset(grid_file_path)
    triangle_id = 0
    triangle_indices = get_triangle_indices(triangle_id, total_cols=81920)

    # Create graph with only 1 height level for visualization
    edge_index, node_ids, horizontal_edges_list = create_atmospheric_graph_from_xarray(
        grid_ds,
        triangle_indices,
        num_height_levels=0, # Visualize only horizontal edges
        device='cpu' # Use CPU for visualization
    )

    print(f"Created Edge Index shape: {edge_index.shape}")
    print(f"Number of Horizontal Edges: {len(horizontal_edges_list)}")

    # Plot the horizontal edges
    plot_horizontal_graph_edges(grid_ds, triangle_indices, horizontal_edges_list, node_ids)

    grid_ds.close()





