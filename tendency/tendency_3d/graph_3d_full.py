import numpy as np
import xarray as xr
import torch
from data_utils import get_triangle_indices


def create_atmospheric_graph_from_xarray(grid_data, triangle_indices, num_height_levels, device, fully_connected, disable_horizontal):
    """
    Creates a graph structure for the atmospheric data and returns edge_index directly.
    
    Args:
        grid_data: ICON grid data as xarray dataset
        triangle_indices: Indices of triangle columns to include
        num_height_levels: Number of vertical height levels
        device: Device to place tensors on
        fully_connected_vertical: Whether to create fully connected vertical edges
        disable_horizontal: Whether to disable horizontal connections between columns
    """
    num_columns = len(triangle_indices)
    node_ids = {}  # (local_col_idx, height_level) -> node_index
    
    vertical_edges_list = []
    horizontal_edges_list = []

    # --- 1. Create ALL nodes and Node IDs (same as before) ---
    for local_col_idx in range(num_columns):
        for height_level in range(num_height_levels):
            node_index = local_col_idx * num_height_levels + height_level 
            node_ids[(local_col_idx, height_level)] = node_index

    # --- 2. Add Horizontal Edges (only if not disabled) ---
    if not disable_horizontal:
        neighbor_indices_global = grid_data['neighbor_cell_index'].values
        triangle_neighbor_indices = neighbor_indices_global[:, triangle_indices.numpy()]

        # Only process horizontal edges if not disabled
        for local_col_idx in range(num_columns):
            for height_level in range(num_height_levels):
                current_node_index = node_ids[(local_col_idx, height_level)]
                neighbors = triangle_neighbor_indices[:, local_col_idx]

                for neighbor_global in neighbors:
                    local_neighbor_idx_array = np.where(triangle_indices.numpy() == neighbor_global - 1)[0]
                        
                    if len(local_neighbor_idx_array) > 0:
                        local_neighbor_idx = local_neighbor_idx_array[0]
                        neighbor_node_index = node_ids[(local_neighbor_idx, height_level)]
                        horizontal_edges_list.append([current_node_index, neighbor_node_index])
                        horizontal_edges_list.append([neighbor_node_index, current_node_index])

    # --- 3. Add Vertical Edges (modified for fully connected option) ---
    if fully_connected:
        # Fully connected vertical approach
        for local_col_idx in range(num_columns):
            # Get all nodes in this column
            column_nodes = [node_ids[(local_col_idx, h)] for h in range(num_height_levels)]
            
            # Connect each node to every other node in the column
            for i in range(num_height_levels):
                for j in range(num_height_levels):
                    if i != j:  # Avoid self-loops
                        source_node = column_nodes[i]
                        target_node = column_nodes[j]
                        vertical_edges_list.append([source_node, target_node])
    else:
        # Original adjacent-only approach
        for local_col_idx in range(num_columns):
            for height_level in range(num_height_levels):
                current_node_index = node_ids[(local_col_idx, height_level)]
                if height_level > 0:
                    lower_node_index = node_ids[(local_col_idx, height_level - 1)]
                    vertical_edges_list.append([current_node_index, lower_node_index])
                    vertical_edges_list.append([lower_node_index, current_node_index])

    # 4. Combine edges and convert to PyTorch Tensor
    edge_index_numpy = np.array(vertical_edges_list + horizontal_edges_list).T
    edge_index = torch.tensor(edge_index_numpy, dtype=torch.long, device=device)
    
    return edge_index, node_ids, horizontal_edges_list



# Cache for edge indices (to avoid recomputing)
EDGE_INDEX_CACHE = {}

def get_3d_graph(grid_file_path,
                 triangle_id,
                 num_height_levels, 
                 batch_size,  
                 total_cols, 
                 division_factor, 
                 device, 
                 fully_connected, 
                 disable_horizontal):
    """
    Get edge index for 3D atmospheric graph, with caching.
    
    Args:
        grid_file_path: Path to the grid file
        triangle_id: ID of the triangle to use
        num_height_levels: Number of vertical height levels
        batch_size: Batch size for batched processing
        device: Device to place tensors on
        total_cols: Total number of columns in the grid
        division_factor: Factor by which to divide the triangle
        fully_connected_vertical: Whether to create fully connected vertical edges
        disable_horizontal: Whether to disable horizontal connections between columns
    """
    cache_key = f"{grid_file_path}_{triangle_id}_{num_height_levels}_{division_factor}_{device}_{fully_connected}_{disable_horizontal}"
    
    # Create or get from cache
    if cache_key not in EDGE_INDEX_CACHE:
        print(f"Creating new graph with key: {cache_key}")
        # Get triangle indices
        triangle_indices = get_triangle_indices(
            triangle_id=triangle_id, 
            total_cols=total_cols,
            division_factor=division_factor
        )
        
        # Load grid dataset
        grid_ds = xr.open_dataset(grid_file_path)
        
        # Create edge index
        edge_index, _, _ = create_atmospheric_graph_from_xarray(
            grid_ds,
            triangle_indices,
            num_height_levels,
            device,
            fully_connected,
            disable_horizontal
        )
        grid_ds.close()
        
        num_columns = len(triangle_indices)
        nodes_per_graph = num_height_levels * num_columns
        EDGE_INDEX_CACHE[cache_key] = (edge_index, num_columns, nodes_per_graph)
    
    edge_index, num_columns, nodes_per_graph = EDGE_INDEX_CACHE[cache_key]
    
    # Apply batching if needed
    if batch_size > 1:
        batched_edge_index = get_batched_edge_index(
            edge_index=edge_index,
            batch_size=batch_size,
            num_nodes_per_graph=nodes_per_graph,
            device=device
        )
        return batched_edge_index, num_columns
    
    return edge_index, num_columns



def get_batched_edge_index(edge_index, batch_size, num_nodes_per_graph, device='cuda'):
    if batch_size == 1:
        return edge_index
    
    # Create batched edge index
    batch_edge_index = edge_index.repeat(1, batch_size)
    offsets = torch.arange(batch_size, device=device) * num_nodes_per_graph
    offsets = offsets.repeat_interleave(edge_index.size(1))
    batch_edge_index = batch_edge_index + offsets.view(1, -1)
    
    return batch_edge_index
