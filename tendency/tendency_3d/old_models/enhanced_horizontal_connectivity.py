import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple, Optional


class EnhancedHorizontalConnectivity:
    """
    Enhanced horizontal connectivity that incorporates mesh-inspired improvements
    while working directly on the ICON triangular grid.
    
    Improvements over basic triangular connectivity:
    1. Extended neighborhood search (beyond immediate neighbors)
    2. Distance-weighted connections
    3. Multi-scale connectivity patterns
    4. Adaptive neighborhood sizes based on local grid density
    """
    
    def __init__(self, grid_data, triangle_indices, max_radius_km=240, max_neighbors=12):
        """
        Args:
            grid_data: ICON grid dataset with spatial information
            triangle_indices: Selected triangle indices for processing
            max_radius_km: Maximum radius for connectivity (in km)
            max_neighbors: Maximum number of neighbors per node
        """
        self.grid_data = grid_data
        self.triangle_indices = triangle_indices
        self.max_radius_km = max_radius_km
        self.max_neighbors = max_neighbors
        
        # Precompute enhanced connectivity patterns
        self._build_enhanced_connectivity()
    
    def _build_enhanced_connectivity(self):
        """Build enhanced connectivity patterns with multiple strategies"""
        num_columns = len(self.triangle_indices)
        
        # Extract spatial coordinates
        clon = self.grid_data['clon'].values[self.triangle_indices.numpy()]
        clat = self.grid_data['clat'].values[self.triangle_indices.numpy()]
        
        # Strategy 1: Distance-based connectivity (mesh-inspired)
        self.distance_neighbors = self._compute_distance_neighbors(clon, clat, num_columns)
        
        # Strategy 2: Extended k-hop connectivity (improved triangular)
        self.extended_neighbors = self._compute_extended_neighbors(num_columns)
        
        # Strategy 3: Adaptive connectivity (density-aware)
        self.adaptive_neighbors = self._compute_adaptive_neighbors(clon, clat, num_columns)
        
    def _compute_distance_neighbors(self, clon, clat, num_columns):
        """Compute neighbors based on spatial distance (mesh-inspired approach)"""
        distance_neighbors = {}
        
        for i in range(num_columns):
            distances = []
            
            for j in range(num_columns):
                if i != j:
                    # Haversine distance on sphere
                    dist_km = self._haversine_distance(clon[i], clat[i], clon[j], clat[j])
                    distances.append((dist_km, j))
            
            # Sort by distance and take closest neighbors within radius
            distances.sort()
            neighbors = [j for dist, j in distances 
                        if dist <= self.max_radius_km][:self.max_neighbors]
            distance_neighbors[i] = neighbors
            
        return distance_neighbors
    
    def _compute_extended_neighbors(self, num_columns):
        """Compute extended k-hop neighbors using triangular connectivity"""
        # Get basic triangular neighbors
        neighbor_indices_global = self.grid_data['neighbor_cell_index'].values
        triangle_neighbor_indices = neighbor_indices_global[:, self.triangle_indices.numpy()]
        
        # Build adjacency for extended search
        adj_list = {i: set() for i in range(num_columns)}
        
        for local_col_idx in range(num_columns):
            neighbors = triangle_neighbor_indices[:, local_col_idx]
            for neighbor_global in neighbors:
                local_neighbor_idx_array = np.where(self.triangle_indices.numpy() == neighbor_global - 1)[0]
                if len(local_neighbor_idx_array) > 0:
                    local_neighbor_idx = local_neighbor_idx_array[0]
                    adj_list[local_col_idx].add(local_neighbor_idx)
        
        # Extend to k-hop neighbors
        extended_neighbors = {}
        for node in range(num_columns):
            neighbors = self._get_k_hop_neighbors(adj_list, node, max_hops=3)
            extended_neighbors[node] = list(neighbors)[:self.max_neighbors]
        
        return extended_neighbors
    
    def _compute_adaptive_neighbors(self, clon, clat, num_columns):
        """Compute adaptive neighbors based on local grid density"""
        adaptive_neighbors = {}
        
        for i in range(num_columns):
            # Compute local density
            nearby_count = 0
            for j in range(num_columns):
                if i != j:
                    dist_km = self._haversine_distance(clon[i], clat[i], clon[j], clat[j])
                    if dist_km <= 160:  # 2x grid spacing
                        nearby_count += 1
            
            # Adapt number of neighbors based on density
            if nearby_count < 6:
                # Sparse region - include more distant neighbors
                max_neighbors_adaptive = min(self.max_neighbors, nearby_count + 4)
                radius_adaptive = self.max_radius_km * 1.5
            else:
                # Dense region - focus on closer neighbors  
                max_neighbors_adaptive = min(self.max_neighbors, 8)
                radius_adaptive = self.max_radius_km
            
            # Find adaptive neighbors
            distances = []
            for j in range(num_columns):
                if i != j:
                    dist_km = self._haversine_distance(clon[i], clat[i], clon[j], clat[j])
                    if dist_km <= radius_adaptive:
                        distances.append((dist_km, j))
            
            distances.sort()
            neighbors = [j for dist, j in distances[:max_neighbors_adaptive]]
            adaptive_neighbors[i] = neighbors
            
        return adaptive_neighbors
    
    def _get_k_hop_neighbors(self, adj_list, start_node, max_hops):
        """Get k-hop neighbors through graph traversal"""
        visited = {start_node}
        current_hop = {start_node}
        
        for hop in range(max_hops):
            next_hop = set()
            for node in current_hop:
                next_hop.update(adj_list[node])
            current_hop = next_hop - visited
            visited.update(current_hop)
            
            if not current_hop:
                break
        
        return visited - {start_node}
    
    def _haversine_distance(self, lon1, lat1, lon2, lat2):
        """Compute Haversine distance between two points on sphere (in km)"""
        R = 6371.0  # Earth radius in km
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_connectivity_strategy(self, strategy="hybrid"):
        """
        Get connectivity based on chosen strategy
        
        Args:
            strategy: "distance", "extended", "adaptive", or "hybrid"
        """
        if strategy == "distance":
            return self.distance_neighbors
        elif strategy == "extended":
            return self.extended_neighbors
        elif strategy == "adaptive":
            return self.adaptive_neighbors
        elif strategy == "hybrid":
            # Combine all strategies with weights
            return self._combine_strategies()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _combine_strategies(self):
        """Combine multiple connectivity strategies"""
        hybrid_neighbors = {}
        num_columns = len(self.triangle_indices)
        
        for i in range(num_columns):
            # Collect neighbors from all strategies
            all_neighbors = set()
            
            # Weight different strategies
            all_neighbors.update(self.distance_neighbors.get(i, []))
            all_neighbors.update(self.extended_neighbors.get(i, []))
            all_neighbors.update(self.adaptive_neighbors.get(i, []))
            
            # Remove self-connections
            all_neighbors.discard(i)
            
            # Limit to max_neighbors
            hybrid_neighbors[i] = list(all_neighbors)[:self.max_neighbors]
        
        return hybrid_neighbors
    
    def get_connectivity_stats(self, strategy="hybrid"):
        """Get statistics about the connectivity pattern"""
        neighbors = self.get_connectivity_strategy(strategy)
        
        sizes = [len(neighbors.get(i, [])) for i in range(len(self.triangle_indices))]
        
        return {
            'min_neighbors': min(sizes) if sizes else 0,
            'max_neighbors': max(sizes) if sizes else 0,
            'avg_neighbors': sum(sizes) / len(sizes) if sizes else 0,
            'total_nodes': len(self.triangle_indices),
            'total_edges': sum(sizes),
            'strategy': strategy
        }


class MeshInspiredGraphTransformer3D(nn.Module):
    """
    Enhanced Graph Transformer that uses mesh-inspired connectivity
    while working directly on ICON grid.
    """
    
    def __init__(self, enhanced_connectivity: EnhancedHorizontalConnectivity, 
                 embed_dim, heads, max_hops, connectivity_strategy="hybrid"):
        super().__init__()
        
        self.enhanced_connectivity = enhanced_connectivity
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_hops = max_hops
        self.connectivity_strategy = connectivity_strategy
        
        # Get the enhanced neighborhood structure
        self.neighborhoods = enhanced_connectivity.get_connectivity_strategy(connectivity_strategy)
        
    def get_enhanced_neighborhoods(self, num_nodes, num_columns, num_height_levels):
        """
        Create enhanced neighborhoods for all nodes using the improved connectivity.
        Incorporates mesh-inspired patterns while respecting ICON grid structure.
        """
        node_neighborhoods = {}
        
        for node_id in range(num_nodes):
            col_id = node_id // num_height_levels
            height_level = node_id % num_height_levels
            
            # Get enhanced horizontal neighbors for this column
            neighbor_cols = self.neighborhoods.get(col_id, [])
            
            # Create node-level neighborhood
            neighbors = set()
            
            # Add all nodes in same column (full vertical connectivity)
            for h in range(num_height_levels):
                same_col_node = col_id * num_height_levels + h
                if same_col_node < num_nodes:
                    neighbors.add(same_col_node)
            
            # Add nodes in neighboring columns (enhanced horizontal connectivity)
            for neighbor_col in neighbor_cols:
                for h in range(num_height_levels):
                    neighbor_node = neighbor_col * num_height_levels + h
                    if neighbor_node < num_nodes:
                        neighbors.add(neighbor_node)
            
            node_neighborhoods[node_id] = sorted(list(neighbors))
        
        return node_neighborhoods
    
    def analyze_improvement(self):
        """Analyze improvement over basic triangular connectivity"""
        stats = self.enhanced_connectivity.get_connectivity_stats(self.connectivity_strategy)
        
        # Compare with basic triangular (3 neighbors per node)
        basic_neighbors_per_node = 3
        enhanced_neighbors_per_node = stats['avg_neighbors']
        
        improvement_factor = enhanced_neighbors_per_node / basic_neighbors_per_node
        
        return {
            'basic_connectivity': {
                'neighbors_per_node': basic_neighbors_per_node,
                'connectivity_type': 'triangular_only'
            },
            'enhanced_connectivity': stats,
            'improvement_factor': improvement_factor,
            'recommendation': self._get_recommendation(improvement_factor, stats)
        }
    
    def _get_recommendation(self, improvement_factor, stats):
        """Provide recommendation based on improvement analysis"""
        if improvement_factor > 2.0:
            return {
                'use_enhanced': True,
                'reason': f"Significant improvement: {improvement_factor:.1f}x more neighbors",
                'benefits': ["Better horizontal information flow", "Multi-scale interactions", "Improved coverage"]
            }
        elif improvement_factor > 1.5:
            return {
                'use_enhanced': True,
                'reason': f"Moderate improvement: {improvement_factor:.1f}x more neighbors",
                'benefits': ["Enhanced connectivity", "Better spherical coverage"]
            }
        else:
            return {
                'use_enhanced': False,
                'reason': f"Limited improvement: {improvement_factor:.1f}x more neighbors",
                'recommendation': "Basic triangular connectivity may be sufficient for your use case"
            } 