"""RBF kernel, Stein operator on the kernel, and Stein Gram matrix (eqs 8, 13, 33, 39)."""

from __future__ import annotations

import jax.numpy as jnp


def pairwise_sq_distances(X):
    """||Xi - Xj||^2 for particles X of shape (N, d)."""
    diffs = X[:, None, :] - X[None, :, :]
    return jnp.sum(diffs * diffs, axis=-1)


def pairwise_distances(X):
    sq = pairwise_sq_distances(X)
    return jnp.sqrt(jnp.maximum(sq, 0.0))


def median_pairwise_distance(X):
    """Median of unique pairwise distances (i < j)."""
    n = X.shape[0]
    dist = pairwise_distances(X)
    ii, jj = jnp.triu_indices(n, k=1)
    return jnp.median(dist[ii, jj])


def median_bandwidth(X):
    """RBF bandwidth from the median heuristic, eq (39): sigma_k^2 = median^2 / log N."""
    n = X.shape[0]
    med = median_pairwise_distance(X)
    return (med * med) / jnp.log(jnp.array(n, dtype=X.dtype))


def rbf(x, y, h):
    """k(x, y) = exp(-||x-y||^2 / (2 sigma_k^2)) with h = sigma_k^2."""
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2.0 * h))


def rbf_matrix(X, h):
    """Pairwise RBF kernel matrix, shape (N, N)."""
    return jnp.exp(-pairwise_sq_distances(X) / (2.0 * h))


def grad_y_rbf(x, y, h):
    """Gradient of k(x, y) with respect to y: k(x,y) (x-y) / h."""
    k = rbf(x, y, h)
    return k * (x - y) / h


def grad_x_rbf(x, y, h):
    """Gradient of k(x, y) with respect to x: k(x,y) (y-x) / h."""
    k = rbf(x, y, h)
    return k * (y - x) / h


def kernel_stein_features(x, particles, scores, h):
    """Vector field features in eq (13): k(x,Xj) score(Xj) + grad_{Xj} k(x, Xj).

    x: (d,), particles: (N, d), scores: (N, d). Returns (N, d).
    """
    diffs = x - particles
    sq = jnp.sum(diffs * diffs, axis=-1)
    k = jnp.exp(-sq / (2.0 * h))
    return k[:, None] * (scores + diffs / h)


def stein_gram(X, scores, h):
    """Gram matrix of the Stein kernel, eq (33).

    [xi]_{ij} = S^{Xi} S^{Xj} k(Xi, Xj)
              = <score_i, grad_{Xj} k> + <score_j, grad_{Xi} k>
                + k <score_i, score_j> + div_{Xi} grad_{Xj} k
    """
    n, d = X.shape
    diffs = X[:, None, :] - X[None, :, :]
    sq = jnp.sum(diffs * diffs, axis=-1)
    k = jnp.exp(-sq / (2.0 * h))
    s_dot_d = jnp.einsum("id,ijd->ij", scores, diffs)
    s_j_dot_d = jnp.einsum("jd,ijd->ij", scores, diffs)
    s_inner = scores @ scores.T
    lap = (d / h) - sq / (h * h)
    xi = k * (s_dot_d / h - s_j_dot_d / h + s_inner + lap)
    return xi


def evaluate_u(x, particles, scores, coeff, h):
    """Corrective drift u(x) = (1/N) sum_j coeff_j [k score_j + grad_{Xj} k], eq (13)."""
    feats = kernel_stein_features(x, particles, scores, h)
    n = particles.shape[0]
    return (coeff @ feats) / n


def evaluate_u_particles(X, scores, coeff, h):
    """u at each particle, shape (N, d)."""
    n, d = X.shape
    diffs = X[:, None, :] - X[None, :, :]
    sq = jnp.sum(diffs * diffs, axis=-1)
    k = jnp.exp(-sq / (2.0 * h))
    terms = k[:, :, None] * (scores[None, :, :] + diffs / h)
    return jnp.einsum("j,ijd->id", coeff, terms) / n


def stein_apply(u_fn, x, score):
    """Stein operator S_p u = div u + <u, grad log p>, eq (8), at a point x."""
    import jax

    u = u_fn(x)
    ju = jax.jacfwd(u_fn)(x)
    return jnp.trace(ju) + jnp.dot(u, score)
