import networkx as nx
import numpy as np
import xarray as xr
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

# --- 1. Function to get triangle indices (Original - Correct for your goal) ---
def get_triangle_indices(triangle_id=1, total_cols=81920):
    block_size = total_cols // 20
    start_idx = triangle_id * block_size
    end_idx   = min(start_idx + block_size, total_cols)
    return torch.arange(start_idx, end_idx, dtype=torch.long)

# --- 3. Graph Creation Function (Corrected and Streamlined) ---
def create_atmospheric_graph_from_xarray(grid_data, triangle_indices, num_height_levels=70, device='cuda'):
    """
    Creates a graph structure for the atmospheric data and returns edge_index directly.
    """
    graph = nx.Graph()
    num_columns = len(triangle_indices)
    node_id_counter = 0
    node_ids = {}  # (local_col_idx, height_level) -> node_id for each height and each column

    # --- 1. Create ALL nodes first ---
    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            node_id = node_id_counter
            node_ids[(local_col_idx, height_level)] = node_id
            graph.add_node(node_id, features=[], id=node_id)
            node_id_counter += 1

    # --- 2. Add Vertical Edges ---
    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            current_node_id = node_ids[(local_col_idx, height_level)]
            
            # adds vertical downward edges
            if height_level > 0:
                graph.add_edge(current_node_id, node_ids[(local_col_idx, height_level - 1)])
            # adds vertical upward edges
            if height_level < num_height_levels - 1:
                graph.add_edge(current_node_id, node_ids[(local_col_idx, height_level + 1)])

    # --- 3. Add Horizontal Edges (with filtering) ---
    neighbor_indices_global = grid_data['neighbor_cell_index'].values  # (3, 81920)
    # get the neighbor indices of the current triangle, based on the grid_data information file
    triangle_neighbor_indices = neighbor_indices_global[:, triangle_indices.numpy()]  # (3, num_columns)

    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            current_node_id = node_ids[(local_col_idx, height_level)]
            neighbors = triangle_neighbor_indices[:, local_col_idx]

            # special edge case filtering for the edges that are not in the current triangle
            for neighbor in neighbors:
                
                # check if the neighbor is in the current triangle
                # Handles the edge case of 2 or 1 neighbors.
                local_neighbor_idx = np.where(triangle_indices.numpy() == neighbor)[0]
                
                # if the neighbor is in the current triangle, add the neighbour edges. 
                if len(local_neighbor_idx) > 0:
                    local_neighbor_idx = local_neighbor_idx[0]
                    neighbor_node_id = node_ids[(local_neighbor_idx, height_level)]
                    graph.add_edge(current_node_id, neighbor_node_id)
                
                # could even add diagonal edges here.

    # Convert to PyG format and extract edge_index
    data = from_networkx(graph)
    edge_index = data.edge_index.to(device)

    return edge_index


def create_edge_index(grid_file_path, triangle_id=1, total_cols=81920, num_height_levels=70, device='cuda'):
    """
    Creates edge_index tensor directly for PyTorch Geometric.
    Opens and closes the grid dataset internally.
    """
    grid_ds = xr.open_dataset(grid_file_path)
    triangle_indices = get_triangle_indices(triangle_id, total_cols)
    edge_index = create_atmospheric_graph_from_xarray(
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




