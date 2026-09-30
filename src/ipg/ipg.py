"""IPG and IPG-CV corrective drift, linear systems (eqs 13-15), and particle step (Alg 1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ipg.kernel import evaluate_u, evaluate_u_particles, median_bandwidth, stein_gram


def centering_matrix(n, dtype=jnp.float32):
    """Pi = I - 11^T / N."""
    eye = jnp.eye(n, dtype=dtype)
    return eye - jnp.ones((n, n), dtype=dtype) / n


def center(g):
    return g - jnp.mean(g)


def solve_phi_ipg(xi, g, lam):
    """Solve (xi/N + lambda I) phi = -g, eq (13). g is already centered."""
    n = xi.shape[0]
    dtype = xi.dtype
    a = xi / n + lam * jnp.eye(n, dtype=dtype)
    return jnp.linalg.solve(a, -g)


def solve_phi_ipg_cv(xi, g, lam):
    """Solve (Pi xi Pi / N + lambda I) phi = -g, eq (15)."""
    n = xi.shape[0]
    dtype = xi.dtype
    pi = centering_matrix(n, dtype=dtype)
    a = (pi @ xi @ pi) / n + lam * jnp.eye(n, dtype=dtype)
    return jnp.linalg.solve(a, -g)


def ipg_phi_and_u(X, scores, g, h, lam, cv=False):
    """Stein Gram, solve for phi, return (phi, coeff, u, xi) with coeff used in u."""
    g = center(g)
    xi = stein_gram(X, scores, h)
    if cv:
        phi = solve_phi_ipg_cv(xi, g, lam)
        coeff = centering_matrix(X.shape[0], dtype=X.dtype) @ phi
    else:
        phi = solve_phi_ipg(xi, g, lam)
        coeff = phi
    u = evaluate_u_particles(X, scores, coeff, h)
    return phi, coeff, u, xi


def particle_step(X, v, score_p, u, dt, sigma, eps):
    """Alg 1 Euler-Maruyama: X + (v + sigma grad log p + u) dt + sqrt(2 sigma dt) eps."""
    noise = jnp.sqrt(jnp.maximum(2.0 * sigma * dt, 0.0)) * eps
    return X + (v + sigma * score_p + u) * dt + noise


def ipg_step(X, v, score_q, grad_r, g, dt, sigma, lam, key, cv=False, h=None):
    """One unweighted IPG / IPG-CV particle update.

    score_p = score_q + grad r, as in Alg 1. Bandwidth from eq (39) if h is None.
    """
    score_p = score_q + grad_r
    if h is None:
        h = median_bandwidth(X)
        h = jnp.maximum(h, jnp.asarray(1e-8, dtype=X.dtype))
    _, _, u, xi = ipg_phi_and_u(X, score_p, g, h, lam, cv=cv)
    eps = jax.random.normal(key, X.shape, dtype=X.dtype)
    X_next = particle_step(X, v, score_p, u, dt, sigma, eps)
    return X_next, u, xi


def run_ipg(
    key,
    X,
    times,
    sigma,
    lam,
    score_q_fn,
    v_fn,
    grad_r_fn,
    g_fn,
    cv=False,
):
    """Propagate particles over times (K+1,), using Alg 1 at each step."""
    n_steps = times.shape[0] - 1

    def body(i, carry):
        X, key = carry
        t = times[i]
        dt = times[i + 1] - times[i]
        key, kstep = jax.random.split(key)
        score_q = score_q_fn(X, t)
        v = v_fn(X, t)
        grad_r = grad_r_fn(X, t)
        g = g_fn(X, t, v)
        X, _, _ = ipg_step(X, v, score_q, grad_r, g, dt, sigma, lam, kstep, cv=cv)
        return X, key

    X, _ = jax.lax.fori_loop(0, n_steps, body, (X, key))
    return X


def make_u_fn(particles, scores, coeff, h):
    def u_fn(x):
        return evaluate_u(x, particles, scores, coeff, h)

    return u_fn
