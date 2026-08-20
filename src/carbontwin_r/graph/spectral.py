import numpy as np

def graph_laplacian(A):
    A = np.asarray(A, dtype=float)
    D = np.diag(A.sum(axis=1))
    return D - A

def normalized_laplacian(A):
    A = np.asarray(A, dtype=float)
    deg = A.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    Dm = np.diag(inv_sqrt)
    return np.eye(len(A)) - Dm @ A @ Dm

def graph_smoothness(x, L):
    x = np.asarray(x, dtype=float)
    return float(x.T @ L @ x)

def graph_smoothness_series(matrix, L):
    return np.asarray([graph_smoothness(row, L) for row in np.asarray(matrix)])
