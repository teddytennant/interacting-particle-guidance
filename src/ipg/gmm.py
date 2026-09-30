"""Isotropic GMM prior and closed-form OU noising scores / probability-flow drift (eqs 37-38)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Noising OU: dZ = -a Z ds + b dW, a=3, b=sqrt(6), s=1-t (eq 37).
OU_A = 3.0
OU_B2 = 6.0


def ou_stationary_variance(a=OU_A, b2=OU_B2):
    return b2 / (2.0 * a)


def noising_time(t):
    """s = 1 - t, generative t=0 is noise, t=1 is data."""
    return 1.0 - t


def ou_gaussian_params(mean, variance, t, a=OU_A, b2=OU_B2):
    """Mean and isotropic variance of N(mean, variance I) after OU noising to time t.

    Z_s = e^{-a s} Z_0 + ..., s=1-t. Component remains Gaussian.
    """
    s = noising_time(t)
    decay = jnp.exp(-a * s)
    mean_t = mean * decay
    var_t = variance * jnp.exp(-2.0 * a * s) + (b2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * s))
    return mean_t, var_t


def ou_gmm_params(means, variance, t, a=OU_A, b2=OU_B2):
    """means (K, d), shared isotropic data-time variance. Returns (means_t, var_t)."""
    s = noising_time(t)
    decay = jnp.exp(-a * s)
    means_t = means * decay
    var_t = variance * jnp.exp(-2.0 * a * s) + (b2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * s))
    return means_t, var_t


def _log_norm_iso(x, mean, var):
    """log N(x; mean, var I). x (d,), mean (d,)."""
    d = x.shape[-1]
    quad = jnp.sum((x - mean) ** 2) / var
    return -0.5 * (d * jnp.log(2.0 * jnp.pi * var) + quad)


def gmm_log_prob(x, means, variance, log_weights):
    """log q(x) for an isotropic GMM. x (d,), means (K, d)."""
    log_comp = jax.vmap(lambda m: _log_norm_iso(x, m, variance))(means)
    return jax.nn.logsumexp(log_weights + log_comp)


def gmm_score(x, means, variance, log_weights):
    """grad log q(x). For one component this is -(x-mean)/variance."""
    log_comp = jax.vmap(lambda m: _log_norm_iso(x, m, variance))(means)
    log_wcomp = log_weights + log_comp
    resp = jax.nn.softmax(log_wcomp)
    comp_scores = jax.vmap(lambda m: -(x - m) / variance)(means)
    return jnp.einsum("k,kd->d", resp, comp_scores)


def gmm_score_particles(X, means, variance, log_weights):
    return jax.vmap(lambda x: gmm_score(x, means, variance, log_weights))(X)


def qt_score(X, t, means, variance, log_weights, a=OU_A, b2=OU_B2):
    """Score of the OU-noised GMM q_t."""
    means_t, var_t = ou_gmm_params(means, variance, t, a=a, b2=b2)
    return gmm_score_particles(X, means_t, var_t, log_weights)


def probability_flow_drift(X, t, means, variance, log_weights, a=OU_A, b2=OU_B2):
    """v_t(x) = a x + (b^2 / 2) grad log q_t(x), eq (38)."""
    score = qt_score(X, t, means, variance, log_weights, a=a, b2=b2)
    return a * X + (b2 / 2.0) * score


def sample_gmm(key, n, means, variance, log_weights):
    """Sample n points from isotropic GMM at the given component parameters."""
    k1, k2 = jax.random.split(key)
    w = jax.nn.softmax(log_weights)
    idx = jax.random.choice(k1, means.shape[0], shape=(n,), p=w)
    chosen = means[idx]
    noise = jax.random.normal(k2, (n, means.shape[1]), dtype=means.dtype)
    return chosen + jnp.sqrt(variance) * noise


def sample_qt(key, n, t, means, variance, log_weights, a=OU_A, b2=OU_B2):
    means_t, var_t = ou_gmm_params(means, variance, t, a=a, b2=b2)
    return sample_gmm(key, n, means_t, var_t, log_weights)


def posterior_gmm_linear_gaussian(means, variance, log_weights, A, y, sigma_obs):
    """Exact posterior GMM for isotropic prior and y ~ N(Ax, sigma_obs^2 I).

    Returns (posterior_mean, component_means, log_weights_post, Sigma).
    Shared component variance implies a shared posterior covariance.
    """
    k, d = means.shape
    m = y.shape[0]
    v = variance
    s2 = sigma_obs ** 2
    prec = jnp.eye(d, dtype=means.dtype) / v + (A.T @ A) / s2
    sigma = jnp.linalg.inv(prec)

    def one_component(mu):
        mu_post = sigma @ (mu / v + A.T @ y / s2)
        mean_y = A @ mu
        cov_y = s2 * jnp.eye(m, dtype=means.dtype) + v * (A @ A.T)
        diff = y - mean_y
        _, logdet = jnp.linalg.slogdet(cov_y)
        quad = diff @ jnp.linalg.solve(cov_y, diff)
        log_marg = -0.5 * (m * jnp.log(2.0 * jnp.pi) + logdet + quad)
        return mu_post, log_marg

    mu_posts, log_margs = jax.vmap(one_component)(means)
    log_w = log_weights + log_margs
    log_w = log_w - jax.nn.logsumexp(log_w)
    w = jnp.exp(log_w)
    mean = jnp.sum(w[:, None] * mu_posts, axis=0)
    return mean, mu_posts, log_w, sigma
