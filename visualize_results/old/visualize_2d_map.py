import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import os

# Load the original NetCDF file
file_tendencies = (
    '../../../../../s3/deepcloud/deepcloud/icon_tendencies/year1/'
    'ml_ecrad_ape_R2B05_myrunscript_1year_183min_tendencies_DOM01_ml_0001_lonlat.nc'
)
ds = xr.open_dataset(file_tendencies)

# For clarity, store clon and clat as 1D NumPy arrays
# Confirm that ds['clon'] and ds['clat'] each have shape (81920,)
clon = ds['clon'].values
clat = ds['clat'].values

def plot_data_on_map(data, title, filename):
    """
    Plots the data on a global map in an unstructured manner.
    data, clon, and clat must all be 1D arrays of the same length.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Because clon/clat/data are 1D arrays, we can use tricontourf on unstructured data
    im = ax.tricontourf(clon, clat, data, cmap='coolwarm', transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    #ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Load the pickle files
with open('/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test/y_true_2d.pickle', 'rb') as handle:
    y_true = pickle.load(handle)

with open('/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_clean_tendency_normTarg/test/y_pred_2d.pickle', 'rb') as handle:
    y_pred = pickle.load(handle)

# Convert PyTorch tensors to NumPy arrays (if they are tensors)
if hasattr(y_true, "numpy"):
    y_true = y_true.numpy()
if hasattr(y_pred, "numpy"):
    y_pred = y_pred.numpy()

# Make sure y_true and y_pred have the shape (81920, height_levels, features)
print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)

# Create output directory if it doesn't exist
output_dir = '/mydata/deepcloud/shared/results-temp/2d_plots'
os.makedirs(output_dir, exist_ok=True)

# Number of features
num_features = y_true.shape[2]

for feature_index in range(num_features):
    # Take the mean over all height levels (axis=1) for the current feature
    y_true_mean = np.mean(y_true[:, :, feature_index], axis=1)
    y_pred_mean = np.mean(y_pred[:, :, feature_index], axis=1)
    
    # Calculate MAE
    mae_mean = np.abs(y_true_mean - y_pred_mean)
    
    # Plot results for this feature
    plot_data_on_map(
        y_true_mean,
        f"True Values (Mean over Height, Feature Index: {feature_index})",
        os.path.join(output_dir, f"true_values_feature_{feature_index}.png")
    )
    
    plot_data_on_map(
        y_pred_mean,
        f"Predicted Values (Mean over Height, Feature Index: {feature_index})",
        os.path.join(output_dir, f"predicted_values_feature_{feature_index}.png")
    )
    
    plot_data_on_map(
        mae_mean,
        f"Mean Absolute Error (Mean over Height, Feature Index: {feature_index})",
        os.path.join(output_dir, f"mae_feature_{feature_index}.png")
    )