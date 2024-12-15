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
    :param first_level_centroids: List of centroids of the first-level subdivided triangles.
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

    plt.figure(figsize=(12, 12))

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
    plt.title('Subdivided Neighboring Triangles with Centroid Graph Connectivity and First-Level Connections')
    plt.show()
    plt.savefig('triangles2.png')

def main():
    # Define the initial large triangles (equilateral) sharing a common side
    # Triangle 1: A, B, C
    # Triangle 2: B, D, C
    height = math.sqrt(3) / 2
    A = [0, 0]
    B = [1, 0]
    C = [0.5, height]
    D = [1.5, height]

    # Define the two large triangles
    large_triangles = [
        [A, B, C],  # Triangle 1
        [B, D, C]   # Triangle 2 (sharing side B-C with Triangle 1)
    ]

    # Subdivide each large triangle to first level to get first-level subdivisions
    first_level_subdivisions = []
    for tri in large_triangles:
        subdivided = subdivide_triangle(tri, depth=1)  # depth=1 for first-level subdivisions
        first_level_subdivisions.extend(subdivided)  # 4 small triangles per large triangle

    # Convert first-level subdivisions to sorted tuples to identify duplicates
    first_level_subdivisions_sorted = []
    for tri in first_level_subdivisions:
        sorted_tri = tuple(sorted([tuple(vertex) for vertex in tri]))
        first_level_subdivisions_sorted.append(sorted_tri)

    # Remove duplicate small triangles (those along the shared side)
    unique_first_level_triangles = list(set(first_level_subdivisions_sorted))

    # Calculate centroids of first-level subdivided triangles
    first_level_centroids = [calculate_centroid(tri) for tri in unique_first_level_triangles]

    # Further subdivide each first-level triangle to get deeper subdivisions (depth=1 per first-level triangle)
    all_subdivided_triangles = []
    for tri in unique_first_level_triangles:
        subdivided = subdivide_triangle(tri, depth=1)  # depth=1 for second-level subdivisions
        all_subdivided_triangles.extend(subdivided)  # 4 small triangles per first-level triangle

    # Convert all subdivided triangles to sorted tuples to identify duplicates
    all_subdivided_triangles_sorted = []
    for tri in all_subdivided_triangles:
        sorted_tri = tuple(sorted([tuple(vertex) for vertex in tri]))
        all_subdivided_triangles_sorted.append(sorted_tri)

    # Remove duplicate small triangles resulting from shared subdivisions
    unique_all_subdivided_triangles = list(set(all_subdivided_triangles_sorted))

    # Calculate centroids for all unique small triangles
    centroids_all = [calculate_centroid(tri) for tri in unique_all_subdivided_triangles]

    # Build adjacency based on shared edges
    adjacency = build_adjacency(unique_all_subdivided_triangles)

    # Determine additional edges among first-level centroids based on adjacency
    first_level_adjacent_edges = []
    # Iterate over all unique pairs of first-level subdivisions
    for tri1, tri2 in combinations(unique_first_level_triangles, 2):
        # Check if tri1 and tri2 share an edge (i.e., have exactly two common vertices)
        common_vertices = set(tri1) & set(tri2)
        if len(common_vertices) == 2:
            centroid1 = calculate_centroid(tri1)
            centroid2 = calculate_centroid(tri2)
            first_level_adjacent_edges.append((centroid1, centroid2))

    # Plot the subdivided triangles and the graph, including selective additional connections
    plot_triangle_graph(adjacency, unique_all_subdivided_triangles, first_level_centroids, first_level_adjacent_edges)

if __name__ == "__main__":
    main()
