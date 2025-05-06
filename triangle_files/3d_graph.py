import numpy as np
import xarray as xr
import torch


def get_triangle_indices(triangle_id=1, total_cols=81920):
    """Get indices of triangle cells for a specific triangle ID."""
    block_size = total_cols // (20*4*4*4*4)
    start_idx = triangle_id * block_size
    end_idx = min(start_idx + block_size, total_cols)
    return torch.arange(start_idx, end_idx, dtype=torch.long)


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



# Cache for edge indices (to avoid recomputing)
EDGE_INDEX_CACHE = {}

def get_3d_graph(grid_file_path, triangle_id=1, num_height_levels=70, batch_size=1, device='cuda', total_cols=81920):
    """
    Get edge index for 3D atmospheric graph, with caching.
    """
    cache_key = f"{grid_file_path}_{triangle_id}_{num_height_levels}_{device}"
    
    # Create or get from cache
    if cache_key not in EDGE_INDEX_CACHE:
        
        # Get triangle indices
        triangle_indices = get_triangle_indices(triangle_id, total_cols)
        
        # Load grid dataset
        grid_ds = xr.open_dataset(grid_file_path)
        
        # Create edge index
        edge_index, _, _ = create_atmospheric_graph_from_xarray(
            grid_ds,
            triangle_indices,
            num_height_levels,
            device
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
