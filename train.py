"""CPU smoke: unweighted IPG on a tiny OU-noised GMM with linear-Gaussian reward."""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from ipg._host import preload

preload()

import jax
import jax.numpy as jnp

from ipg.gmm import (
    posterior_gmm_linear_gaussian,
    probability_flow_drift,
    qt_score,
    sample_qt,
)
from ipg.ipg import run_ipg
from ipg.reward import gt, r_and_grad


def main():
    d = 2
    n_comp = 2
    n_particles = 32
    n_steps = 40
    lam = 1e-3
    sigma_t = 0.05
    sigma_obs = jnp.asarray(0.25, dtype=jnp.float64)
    seed = 0

    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(seed)
    means = jnp.array([[1.4, 0.0], [-1.4, 0.0]], dtype=jnp.float64)
    variance = jnp.asarray(0.2, dtype=jnp.float64)
    log_weights = jnp.log(jnp.array([0.5, 0.5], dtype=jnp.float64))

    A = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    y = jnp.array([1.1], dtype=jnp.float64)
    k0, krun = jax.random.split(key, 2)

    post_mean, _, _, _ = posterior_gmm_linear_gaussian(
        means, variance, log_weights, A, y, sigma_obs
    )
    prior_mean = jnp.einsum("k,kd->d", jax.nn.softmax(log_weights), means)
    prior_mean_error = float(jnp.linalg.norm(prior_mean - post_mean))

    X0 = sample_qt(k0, n_particles, 0.0, means, variance, log_weights)
    times = jnp.linspace(0.0, 1.0, n_steps + 1, dtype=means.dtype)

    def score_q_fn(X, t):
        return qt_score(X, t, means, variance, log_weights)

    def v_fn(X, t):
        return probability_flow_drift(X, t, means, variance, log_weights)

    def grad_r_fn(X, t):
        return r_and_grad(X, t, A, y, sigma_obs)[1]

    def g_fn(X, t, v):
        return gt(X, t, v, A, y, sigma_obs)

    X = run_ipg(
        krun,
        X0,
        times,
        sigma_t,
        lam,
        score_q_fn,
        v_fn,
        grad_r_fn,
        g_fn,
        cv=True,
    )
    ipg_mean = jnp.mean(X, axis=0)
    mean_error = float(jnp.linalg.norm(ipg_mean - post_mean))

    out = {
        "mean_error": mean_error,
        "prior_mean_error": prior_mean_error,
        "ipg_mean": [float(v) for v in ipg_mean],
        "exact_posterior_mean": [float(v) for v in post_mean],
        "n_particles": n_particles,
        "dim": d,
        "n_components": n_comp,
        "n_steps": n_steps,
        "lambda": lam,
        "sigma_t": sigma_t,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
