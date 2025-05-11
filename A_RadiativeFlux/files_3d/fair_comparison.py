import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import both models
from models.gnn import AtmosphericColumnGNN
from files_3d.gnn_3d import GNN3d
from files_3d.data_loader_3d import IconIterableDataset_3D

class FairModelComparison:
    """
    A class to handle fair comparison between single column and 3D GNN approaches.
    Both approaches will model the exact same atmospheric columns.
    """
    def __init__(
        self,
        triangle_id=1,
        total_cols=81920,
        division_factor=1,
        grid_file_path="path/to/icon_grid.nc",
        model_params=None,
        device='cuda',
        h5_files=None,
        output_dir=None
    ):
        self.triangle_id = triangle_id
        self.total_cols = total_cols
        self.division_factor = division_factor
        self.grid_file_path = grid_file_path
        self.device = device
        self.h5_files = h5_files or []
        self.output_dir = output_dir or "/mydata/deepcloud/yves/A_RadiativeFlux/comparison_results"
        
        # Default model parameters if none provided
        self.model_params = model_params or {
            'embed_dim': 32,
            'depth': 3,
            'dropout': 0.1,
            'channels_in_3d': 4,  # Adjust based on your data
            'channels_in_2d': 6,  # Adjust based on your data
            'channels_out': 4,    # Adjust based on your data
            'edge_channels_in': 0,
            'num_height_levels': 80,  # Adjust based on your data
            'max_skip': 3,
            'emb_dropout': 0.1
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize both single column and 3D GNN models with the same parameters"""
        # Common parameters
        params = self.model_params
        
        # Single column model - will be applied to each column independently
        self.column_model = AtmosphericColumnGNN(
            embed_dim=params['embed_dim'],
            depth=params['depth'],
            dropout=params['dropout'],
            max_skip=params['max_skip'],
            emb_dropout=params['emb_dropout'],
            channel_3d=params['channels_in_3d'],
            channel_2d=params['channels_in_2d'],
            channels_out=params['channels_out'],
            edge_channels_in=params['edge_channels_in'],
            fully_connected=False  # Vertical connections only within each column
        ).to(self.device)
        
        # 3D graph model - models all columns together with horizontal connections
        self.graph_model = GNN3d(
            total_cols=self.total_cols,
            grid_file_path=self.grid_file_path,
            triangle_id=self.triangle_id,
            embed_dim=params['embed_dim'],
            depth=params['depth'],
            dropout=params['dropout'],
            channels_in_3d=params['channels_in_3d'],
            channels_in_2d=params['channels_in_2d'],
            channels_out=params['channels_out'],
            edge_channels_in=params['edge_channels_in'],
            num_height_levels=params['num_height_levels'],
            device=self.device,
            division_factor=self.division_factor,
            fully_connected=False  # Use standard horizontal connections
        ).to(self.device)
        
        # 3D graph model WITHOUT horizontal connections - treats the entire area as independent columns
        self.no_connections_model = GNN3d(
            total_cols=self.total_cols,
            grid_file_path=self.grid_file_path,
            triangle_id=self.triangle_id,
            embed_dim=params['embed_dim'],
            depth=params['depth'],
            dropout=params['dropout'],
            channels_in_3d=params['channels_in_3d'],
            channels_in_2d=params['channels_in_2d'],
            channels_out=params['channels_out'],
            edge_channels_in=params['edge_channels_in'],
            num_height_levels=params['num_height_levels'],
            device=self.device,
            division_factor=self.division_factor,
            fully_connected=False,  # Use standard vertical connections
            disable_horizontal=True  # New parameter to disable horizontal connections
        ).to(self.device)
        
    def load_models(self, column_model_path, graph_model_path, no_connections_path=None):
        """Load pre-trained model weights"""
        self.column_model.load_state_dict(torch.load(column_model_path))
        self.graph_model.load_state_dict(torch.load(graph_model_path))
        
        self.column_model.eval()
        self.graph_model.eval()
        
        # Load no_connections model if a path is provided, otherwise use graph model weights
        if no_connections_path:
            self.no_connections_model.load_state_dict(torch.load(no_connections_path))
        else:
            self.no_connections_model.load_state_dict(torch.load(graph_model_path))
        self.no_connections_model.eval()
    
    def prepare_data(self):
        """Prepare data loader for inference"""
        dataset = IconIterableDataset_3D(
            filenames=self.h5_files,
            triangle_id=self.triangle_id,
            shuffle=False,
            dtype='float32',
            total_cols=self.total_cols,
            division_factor=self.division_factor
        )
        
        return DataLoader(dataset, batch_size=1, num_workers=0)
    
    def run_comparison(self, num_samples=10):
        """Run both models on the same data and compare results"""
        loader = self.prepare_data()
        
        all_mse_column = []
        all_mse_graph = []
        all_mse_no_connections = []
        all_sample_ids = []
        
        with torch.no_grad():
            for sample_id, (x3d, x2d, y_true) in enumerate(loader):
                if sample_id >= num_samples:
                    break
                    
                # Move data to device
                x3d = x3d.to(self.device)
                x2d = x2d.to(self.device)
                y_true = y_true.to(self.device)
                
                # Get predictions from 3D graph model with horizontal connections
                y_pred_graph = self.graph_model(x3d, x2d, x2d)
                
                # Get predictions from 3D model without horizontal connections
                y_pred_no_connections = self.no_connections_model(x3d, x2d, x2d)
                
                # Process each column independently with the column model
                # Reshape to iterate over columns
                num_columns = x3d.shape[1]
                y_pred_columns = []
                
                for col_idx in range(num_columns):
                    x3d_col = x3d[:, col_idx:col_idx+1]  # Keep batch dimension
                    x2d_col = x2d[:, col_idx:col_idx+1]  # Keep batch dimension
                    
                    # Get prediction for this column
                    y_col = self.column_model(
                        x3d_col.squeeze(1),  # Remove column dimension for column model
                        x2d_col.squeeze(1),  # Remove column dimension for column model
                        x2d_col.squeeze(1)   # Remove column dimension for column model
                    )
                    
                    # Add back column dimension
                    y_col = y_col.unsqueeze(1)
                    y_pred_columns.append(y_col)
                
                # Combine all column predictions
                y_pred_column = torch.cat(y_pred_columns, dim=1)
                
                # Calculate MSE for all approaches
                mse_column = torch.mean((y_pred_column - y_true)**2)
                mse_graph = torch.mean((y_pred_graph - y_true)**2)
                mse_no_connections = torch.mean((y_pred_no_connections - y_true)**2)
                
                all_mse_column.append(mse_column.item())
                all_mse_graph.append(mse_graph.item())
                all_mse_no_connections.append(mse_no_connections.item())
                all_sample_ids.append(sample_id)
                
                # Visualize first few samples
                if sample_id < 3:
                    self._visualize_comparison(
                        y_true[0],                   # Remove batch dimension
                        y_pred_column[0],            # Remove batch dimension
                        y_pred_graph[0],             # Remove batch dimension
                        y_pred_no_connections[0],    # Remove batch dimension
                        sample_id
                    )
        
        # Summarize and plot results
        self._plot_mse_comparison(all_sample_ids, all_mse_column, all_mse_graph, all_mse_no_connections)
        self._save_summary(all_mse_column, all_mse_graph, all_mse_no_connections)
    
    def _visualize_comparison(self, y_true, y_pred_column, y_pred_graph, y_pred_no_connections, sample_id):
        """Visualize predictions from all models against ground truth"""
        # Detach and move to CPU for visualization
        y_true = y_true.detach().cpu().numpy()
        y_pred_column = y_pred_column.detach().cpu().numpy()
        y_pred_graph = y_pred_graph.detach().cpu().numpy()
        y_pred_no_connections = y_pred_no_connections.detach().cpu().numpy()
        
        # Create figure with subplots for each output channel
        num_channels = y_true.shape[-1]
        fig, axes = plt.subplots(num_channels, 4, figsize=(20, 4*num_channels))
        
        for ch in range(num_channels):
            # Get data for this channel
            true_ch = y_true[..., ch]
            col_ch = y_pred_column[..., ch]
            graph_ch = y_pred_graph[..., ch]
            no_conn_ch = y_pred_no_connections[..., ch]
            
            # Get value range for consistent colormap
            vmin = min(true_ch.min(), col_ch.min(), graph_ch.min(), no_conn_ch.min())
            vmax = max(true_ch.max(), col_ch.max(), graph_ch.max(), no_conn_ch.max())
            
            # Plot ground truth
            im1 = axes[ch, 0].imshow(true_ch, vmin=vmin, vmax=vmax)
            axes[ch, 0].set_title(f'Channel {ch} - Ground Truth')
            plt.colorbar(im1, ax=axes[ch, 0])
            
            # Plot column model prediction
            im2 = axes[ch, 1].imshow(col_ch, vmin=vmin, vmax=vmax)
            axes[ch, 1].set_title(f'Channel {ch} - Single Column')
            plt.colorbar(im2, ax=axes[ch, 1])
            
            # Plot graph model prediction
            im3 = axes[ch, 2].imshow(graph_ch, vmin=vmin, vmax=vmax)
            axes[ch, 2].set_title(f'Channel {ch} - 3D Graph')
            plt.colorbar(im3, ax=axes[ch, 2])
            
            # Plot no connections model prediction
            im4 = axes[ch, 3].imshow(no_conn_ch, vmin=vmin, vmax=vmax)
            axes[ch, 3].set_title(f'Channel {ch} - No Horizontal')
            plt.colorbar(im4, ax=axes[ch, 3])
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'comparison_sample_{sample_id}.png'))
        plt.close()
    
    def _plot_mse_comparison(self, sample_ids, mse_column, mse_graph, mse_no_connections):
        """Plot MSE comparison between all approaches"""
        plt.figure(figsize=(10, 6))
        plt.plot(sample_ids, mse_column, 'o-', label='Single Column')
        plt.plot(sample_ids, mse_graph, 's-', label='3D Graph')
        plt.plot(sample_ids, mse_no_connections, '^-', label='No Horizontal')
        plt.xlabel('Sample ID')
        plt.ylabel('MSE')
        plt.title('Model Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'mse_comparison.png'))
        plt.close()
        
        # Also plot the ratios compared to the 3D graph model
        plt.figure(figsize=(10, 6))
        ratios_column = [c/g if g > 0 else 1 for c, g in zip(mse_column, mse_graph)]
        ratios_no_conn = [n/g if g > 0 else 1 for n, g in zip(mse_no_connections, mse_graph)]
        
        x = np.arange(len(sample_ids))
        width = 0.35
        
        plt.bar(x - width/2, ratios_column, width, label='Single Column / 3D Graph')
        plt.bar(x + width/2, ratios_no_conn, width, label='No Horizontal / 3D Graph')
        
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.xlabel('Sample ID')
        plt.ylabel('MSE Ratio')
        plt.title('Performance Ratio (>1 means 3D Graph is better)')
        plt.xticks(x, sample_ids)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'mse_ratio.png'))
        plt.close()
    
    def _save_summary(self, mse_column, mse_graph, mse_no_connections):
        """Save numerical summary of results"""
        avg_mse_column = np.mean(mse_column)
        avg_mse_graph = np.mean(mse_graph)
        avg_mse_no_connections = np.mean(mse_no_connections)
        
        improvement_over_column = ((avg_mse_column - avg_mse_graph) / avg_mse_column) * 100
        improvement_over_no_conn = ((avg_mse_no_connections - avg_mse_graph) / avg_mse_no_connections) * 100
        
        with open(os.path.join(self.output_dir, 'comparison_summary.txt'), 'w') as f:
            f.write(f"Fair Comparison Results\n")
            f.write(f"======================\n\n")
            f.write(f"Triangle ID: {self.triangle_id}\n")
            f.write(f"Model parameters: {self.model_params}\n\n")
            f.write(f"Single Column Model Average MSE: {avg_mse_column:.6f}\n")
            f.write(f"3D Graph Model Average MSE: {avg_mse_graph:.6f}\n")
            f.write(f"No Horizontal Connections Model Average MSE: {avg_mse_no_connections:.6f}\n\n")
            f.write(f"3D Graph improvement over Single Column: {improvement_over_column:.2f}%\n")
            f.write(f"3D Graph improvement over No Horizontal: {improvement_over_no_conn:.2f}%\n\n")
            f.write(f"Per-sample MSE (Column): {mse_column}\n")
            f.write(f"Per-sample MSE (Graph): {mse_graph}\n")
            f.write(f"Per-sample MSE (No Horizontal): {mse_no_connections}\n")

if __name__ == "__main__":
    # Example usage
    h5_files = [
        # Add your data file paths here
    ]
    
    comparison = FairModelComparison(
        triangle_id=1,
        total_cols=81920,
        division_factor=1,
        grid_file_path="/path/to/icon_grid.nc",
        h5_files=h5_files,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    comparison.load_models(
        column_model_path="/path/to/single_column_model.pt",
        graph_model_path="/path/to/3d_graph_model.pt",
        no_connections_path=None  # Can use same weights as graph model if needed
    )
    
    comparison.run_comparison(num_samples=10) 