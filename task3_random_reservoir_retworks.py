import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

# ============================================================
# Helper functions
# ============================================================

def generate_reservoir_matrix(N, p, rho_target, weight_distribution="normal", seed=None):
    """
    Generate a random reservoir connection matrix A with:
    - Erdős–Rényi topology G(N, p)
    - Random weights
    - Spectral radius scaled to rho_target
    """

    if seed is not None:
        np.random.seed(seed)

    # Step 1: Generate Erdős–Rényi random graph
    G = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)

    # Step 2: Create weighted connection matrix A
    A = np.zeros((N, N))

    for i, j in G.edges():
        if weight_distribution == "normal":
            weight = np.random.normal(0, 1)
        elif weight_distribution == "uniform":
            weight = np.random.uniform(-1, 1)
        else:
            raise ValueError("Unknown weight distribution.")

        # Undirected graph → symmetric weights
        A[i, j] = weight
        A[j, i] = weight

    # Step 3: Compute current spectral radius
    eigenvalues = eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))

    # Avoid division by zero
    if spectral_radius == 0:
        raise ValueError("Spectral radius is zero. Try increasing p.")

    # Step 4: Rescale matrix to target spectral radius
    A *= rho_target / spectral_radius

    return A, G


def plot_reservoir_network(G, title):
    """
    Plot reservoir network with nodes colored by degree
    """

    degrees = np.array([deg for _, deg in G.degree()])
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(7, 7))
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=80,
        node_color=degrees,
        cmap=plt.cm.viridis
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    plt.colorbar(nodes, label="Node degree")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# Task 3 parameters (STRICTLY from the assignment)
# ============================================================

N = 100  # Network size

parameter_sets = {
    "A": {"p": 0.06, "rho": 0.6},
    "B": {"p": 0.08, "rho": 1.0},
    "C": {"p": 0.10, "rho": 1.6},
    "D": {"p": 0.12, "rho": 1.2},
}

# ============================================================
# Generate and visualize all four reservoir networks
# ============================================================

reservoir_matrices = {}

for label, params in parameter_sets.items():
    p = params["p"]
    rho = params["rho"]

    print(f"\nGenerating reservoir {label}: p = {p}, rho = {rho}")

    A, G = generate_reservoir_matrix(
        N=N,
        p=p,
        rho_target=rho,
        weight_distribution="normal",
        seed=42
    )

    # Store matrix
    reservoir_matrices[label] = A

    # Verify spectral radius
    eigs = eigvals(A)
    rho_actual = np.max(np.abs(eigs))
    print(f"Actual spectral radius: {rho_actual:.4f}")

    # Plot network
    plot_reservoir_network(
        G,
        title=f"Reservoir Network {label} (p = {p}, ρ = {rho})"
    )
