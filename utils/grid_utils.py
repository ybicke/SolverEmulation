# File: mydata/deepcloud/yves/dataset/grid_utils.py
import numpy as np
from netCDF4 import Dataset



def load_parent_cell_indices(grid_file):
    # Open the NetCDF grid file
    dataset = Dataset(grid_file, 'r')
    
    # Extract parent cell indices
    parent_cell_indices = dataset.variables['parent_cell_index'][:]  # Shape: (num_cells,)
    
    # Close the dataset
    dataset.close()
    
    # Adjust for 0-based indexing if necessary
    if parent_cell_indices.min() >= 1:
        parent_cell_indices -= 1  # Convert to 0-based indexing

    return parent_cell_indices  # Shape: (num_cells,)



def build_hierarchical_neighborhoods(parent_cell_indices, neighborhood_size):
    num_cells = parent_cell_indices.shape[0]
    
    # Convert parent_cell_indices to a NumPy array if it's a tuple
    parent_cell_indices = np.array(parent_cell_indices)
    
    # Initialize the hierarchy mapping
    cell_to_ancestor = parent_cell_indices.copy()

    # Calculate the number of levels to move up in the hierarchy
    levels_up = int(np.log2(neighborhood_size) / np.log2(4))  # Since each parent has 4 children

    for _ in range(levels_up - 1):
        # Move up one level in the hierarchy by finding the parent of the parent
        cell_to_ancestor = parent_cell_indices[cell_to_ancestor]

    # Now, group cells based on their ancestor at the desired level
    unique_ancestors, inverse_indices = np.unique(cell_to_ancestor, return_inverse=True)
    neighborhoods = [[] for _ in range(len(unique_ancestors))]
    for idx, ancestor_idx in enumerate(inverse_indices):
        neighborhoods[ancestor_idx].append(idx)

    return np.array(neighborhoods)




# # Load parent cell indices

# # Open the grid file
# grid_file = '/mydata/deepcloud/salman/dataset/icon_grid_0008_R02B05_G.nc'
# parent_cell_indices = load_parent_cell_indices(grid_file)
# print(parent_cell_indices)


# # Define the neighborhood size
# neighborhood_size = 4  # For example

# # Build neighborhoods
# neighborhoods = build_hierarchical_neighborhoods(parent_cell_indices, neighborhood_size)

# print(neighborhoods)



# def build_hierarchical_neighborhoods(parent_cell_indices, neighborhood_size):
#     num_cells = parent_cell_indices.shape[0]
#     # For neighborhood_size = 4, group by parent_cell_index
#     # For neighborhood_size = 16, group by grandparent_cell_index, and so on.

#     # First, create a mapping from cell index to parent index
#     cell_to_parent = parent_cell_indices

#     # Initialize the hierarchy mapping
#     cell_to_ancestor = cell_to_parent.copy()

#     # Calculate the number of levels to move up the hierarchy
#     levels_up = int(np.log2(neighborhood_size) / np.log2(4))  # Since each parent has 4 children

#     for _ in range(levels_up - 1):
#         # Move up one level in the hierarchy by finding the parent of the parent
#         cell_to_ancestor = cell_to_parent[cell_to_ancestor]

#     # Now, group cells based on their ancestor at the desired level
#     unique_ancestors = np.unique(cell_to_ancestor)
#     ancestor_to_cells = {ancestor: np.where(cell_to_ancestor == ancestor)[0] for ancestor in unique_ancestors}

#     # Build the neighborhoods
#     neighborhoods = [ancestor_to_cells[ancestor] for ancestor in unique_ancestors]

#     return np.array(neighborhoods)