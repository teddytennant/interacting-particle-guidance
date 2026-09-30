"""Linear-Gaussian log-likelihood reward, schedule r(x,t)=t R(x), and g_t (eq 3)."""

from __future__ import annotations

import jax.numpy as jnp


def linear_gaussian_loglik(X, A, y, sigma_obs):
    """R(x) = log N(y; A x, sigma_obs^2 I). X is (d,) or (N, d)."""
    Xb = X if X.ndim == 2 else X[None, :]
    residual = y - Xb @ A.T
    m = y.shape[0]
    var = sigma_obs ** 2
    logp = -0.5 * (m * jnp.log(2.0 * jnp.pi * var) + jnp.sum(residual ** 2, axis=-1) / var)
    return logp[0] if X.ndim == 1 else logp


def grad_R(X, A, y, sigma_obs):
    """grad_x log N(y; A x, sigma_obs^2 I) = A^T (y - A x) / sigma_obs^2."""
    Xb = X if X.ndim == 2 else X[None, :]
    residual = y - Xb @ A.T
    g = residual @ A / (sigma_obs ** 2)
    return g[0] if X.ndim == 1 else g


def r_and_grad(X, t, A, y, sigma_obs):
    """r(x,t) = t R(x), grad r = t grad R."""
    R = linear_gaussian_loglik(X, A, y, sigma_obs)
    gR = grad_R(X, A, y, sigma_obs)
    return t * R, t * gR


def gt(X, t, v, A, y, sigma_obs):
    """g_t(x) = dt r + <v_t, grad r>, eq (3), with r = t R."""
    R = linear_gaussian_loglik(X, A, y, sigma_obs)
    gR = grad_R(X, A, y, sigma_obs)
    grad_r = t * gR
    return R + jnp.sum(v * grad_r, axis=-1)
