
import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np
import xarray as xr
from os.path import join, dirname
from triangle_files.triangle_graph import create_edge_index, get_centroids, get_triangle_indices, create_atmospheric_graph_from_xarray







def visualize_horizontal_slice(grid_file_path, triangle_id=0, height_level_to_visualize=0, total_cols=81920, num_height_levels=70):
    """
    Visualizes the horizontal graph structure at a specific height level.
    """
    grid_ds = xr.open_dataset(grid_file_path)
    triangle_indices = get_triangle_indices(triangle_id, total_cols)
    graph_nx = create_atmospheric_graph_from_xarray(grid_ds, triangle_indices, num_height_levels, device='cpu') # Create NetworkX graph
    clon, clat = get_centroids(grid_file_path, triangle_id, total_cols)
    grid_ds.close()

    # 1. Get nodes at the specified height level
    nodes_to_draw = []
    pos = {} # positions for plotting
    node_ids_dict = {} # Dictionary to map (local_col_idx, height_level) to node index
    for local_col_idx in range(len(triangle_indices)):
        for height_level in range(num_height_levels):
            node_id = (local_col_idx, height_level)
            node_index = len(node_ids_dict) # Assign sequential index
            node_ids_dict[node_id] = node_index
            if height_level == height_level_to_visualize: # Only add nodes for the slice
                nodes_to_draw.append(node_index)
                pos[node_index] = (clon[local_col_idx].item(), clat[local_col_idx].item()) # Use centroid coords as positions

    # Attach node_ids_dict to the graph for easy lookup in edge processing
    graph_nx.graph['node_ids_dict'] = node_ids_dict

    # 2. Get horizontal edges
    horizontal_edges = []
    for u, v in graph_nx.edges():
        u_coords = None
        v_coords = None
        for coords, index in node_ids_dict.items(): # Find coordinates for each node index
            if index == u:
                u_coords = coords
            if index == v:
                v_coords = coords
        if u_coords and v_coords and u_coords[1] == height_level_to_visualize and v_coords[1] == height_level_to_visualize: # Check if both nodes are at the same height level
            horizontal_edges.append((u, v))

    # 3. Draw the subgraph
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(graph_nx, pos, nodelist=nodes_to_draw, node_size=20, node_color='skyblue')
    nx.draw_networkx_edges(graph_nx, pos, edgelist=horizontal_edges, width=0.5, alpha=0.5)
    plt.title(f"Horizontal Graph Slice at Height Level {height_level_to_visualize}, Triangle {triangle_id}")
    plt.xlabel("Centroid Longitude")
    plt.ylabel("Centroid Latitude")
    plt.savefig(f"horizontal_slice_triangle_{triangle_id}_height_{height_level_to_visualize}.png")


if __name__ == '__main__':
    grid_file_path = "/mydata/deepcloud/yves/SolverEmulation/data_exploration/icon_grid_0008_R02B05_G.nc" # Replace with your actual grid file path
    visualize_horizontal_slice(grid_file_path, triangle_id=0, height_level_to_visualize=30)