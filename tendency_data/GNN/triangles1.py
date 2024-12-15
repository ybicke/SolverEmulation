import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from matplotlib.patches import Polygon
import math

def midpoint(p1, p2):
    """Calculate the midpoint between two points."""
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def subdivide_triangle(vertices, depth):
    """
    Recursively subdivide a triangle into 4 smaller triangles.

    :param vertices: List of three vertices of the triangle.
    :param depth: Number of subdivision levels.
    :return: List of all subdivided triangles.
    """
    if depth == 0:
        return [vertices]
    
    v1, v2, v3 = vertices
    # Calculate midpoints of each side
    m12 = midpoint(v1, v2)
    m23 = midpoint(v2, v3)
    m31 = midpoint(v3, v1)
    
    # Create 4 smaller triangles
    t1 = [v1, m12, m31]
    t2 = [m12, v2, m23]
    t3 = [m31, m23, v3]
    t4 = [m12, m23, m31]
    
    # Recursively subdivide the smaller triangles
    return (subdivide_triangle(t1, depth - 1) +
            subdivide_triangle(t2, depth - 1) +
            subdivide_triangle(t3, depth - 1) +
            subdivide_triangle(t4, depth - 1))

def calculate_centroid(triangle):
    """Calculate the centroid of a triangle."""
    x = sum([vertex[0] for vertex in triangle]) / 3
    y = sum([vertex[1] for vertex in triangle]) / 3
    return (x, y)

def build_adjacency(triangles):
    """
    Build adjacency list based on shared edges between triangles.

    :param triangles: List of triangles, each triangle is a tuple of three vertices.
    :return: Dictionary mapping centroid to set of adjacent centroids.
    """
    edge_dict = {}
    
    for tri in triangles:
        # Each tri is a tuple of tuples
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            if edge in edge_dict:
                edge_dict[edge].append(tri)
            else:
                edge_dict[edge] = [tri]
    
    # Calculate centroids for all triangles
    centroids = {tri: calculate_centroid(tri) for tri in triangles}
    
    adjacency = {}
    for tri in triangles:
        centroid = centroids[tri]
        adjacency.setdefault(centroid, set())
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            neighbors = edge_dict[edge]
            for neighbor in neighbors:
                if neighbor != tri:
                    neighbor_centroid = centroids[neighbor]
                    adjacency[centroid].add(neighbor_centroid)
    
    return adjacency

def plot_triangle_graph(adjacency, triangles, first_level_centroids, first_level_adjacent_edges):
    """
    Plot the subdivided triangles and the graph of centroids connected by edges,
    including additional edges among the first-level centroids based on adjacency.

    :param adjacency: Dictionary mapping nodes to their adjacent nodes.
    :param triangles: List of subdivided triangles (tuples of tuples).
    :param first_level_centroids: List of centroids of the first four subdivided triangles.
    :param first_level_adjacent_edges: List of tuples representing additional edges among first-level centroids.
    """
    G = nx.Graph()
    
    # Add nodes and edges based on adjacency
    for node, neighbors in adjacency.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Add additional edges among the first-level centroids based on adjacency
    for centroid1, centroid2 in first_level_adjacent_edges:
        G.add_edge(centroid1, centroid2, color='red', weight=2)  # Customize as needed
    
    # Extract positions
    pos = {node: node for node in G.nodes()}
    
    plt.figure(figsize=(10, 10))
    
    # Plot subdivided triangles
    ax = plt.gca()
    for tri in triangles:
        polygon = Polygon(tri, closed=True, edgecolor='black', facecolor='none', linewidth=0.5)
        ax.add_patch(polygon)
    
    # Separate edges by type for different styling
    # Regular adjacency edges
    regular_edges = [(u, v) for u, v in G.edges() if not (u in first_level_centroids and v in first_level_centroids)]
    # Additional first-level centroid edges
    additional_edges = [(u, v) for u, v in G.edges() if (u in first_level_centroids and v in first_level_centroids)]
    
    # Draw regular edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color='gray', alpha=0.6)
    # Draw additional edges in red
    nx.draw_networkx_edges(G, pos, edgelist=additional_edges, edge_color='red', width=2)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='blue', alpha=0.8)
    
    # Highlight the first-level centroids
    nx.draw_networkx_nodes(G, pos, nodelist=first_level_centroids, node_size=50, node_color='green', alpha=1.0, label='First-Level Centroids')
    
    # Optionally, draw labels
    # nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.legend(scatterpoints=1)
    plt.axis('equal')
    plt.axis('off')
    plt.title('Subdivided Triangle with Centroid Graph and Selective Additional Connections')
    plt.show()
    plt.savefig('triangles.png')

def main():
    # Define the initial large triangle (equilateral for simplicity)
    height = math.sqrt(3) / 2
    initial_triangle = [[0, 0], [1, 0], [0.5, height]]
    
    # Subdivide the initial triangle to depth=1 to get first four subdivisions
    first_level_subdivisions = subdivide_triangle(initial_triangle, depth=1)  # 4 triangles
    
    # Calculate centroids of the first four subdivided triangles
    first_level_centroids = [calculate_centroid(tri) for tri in first_level_subdivisions]
    
    # Further subdivide each of the first four triangles to depth=1, resulting in 16 small triangles
    all_subdivided_triangles = []
    for tri in first_level_subdivisions:
        subdivided = subdivide_triangle(tri, depth=1)  # 4 subdivisions per triangle
        all_subdivided_triangles.extend(subdivided)
    
    # Convert triangles to tuples of tuples for consistency
    all_subdivided_triangles_tuples = [tuple(tuple(vertex) for vertex in tri) for tri in all_subdivided_triangles]
    first_level_subdivisions_tuples = [tuple(tuple(vertex) for vertex in tri) for tri in first_level_subdivisions]
    
    # Build adjacency based on shared edges
    adjacency = build_adjacency(all_subdivided_triangles_tuples)
    
    # Determine additional edges among first-level centroids based on adjacency
    first_level_adjacent_edges = []
    # Iterate over all unique pairs of first-level subdivisions
    for tri1, tri2 in combinations(first_level_subdivisions_tuples, 2):
        # Check if tri1 and tri2 share an edge (i.e., have exactly two common vertices)
        common_vertices = set(tri1) & set(tri2)
        if len(common_vertices) == 2:
            centroid1 = calculate_centroid(tri1)
            centroid2 = calculate_centroid(tri2)
            first_level_adjacent_edges.append((centroid1, centroid2))
    
    # Plot the triangles and the graph, including selective additional connections
    plot_triangle_graph(adjacency, all_subdivided_triangles_tuples, first_level_centroids, first_level_adjacent_edges)

if __name__ == "__main__":
    main()
