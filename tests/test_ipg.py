"""IPG linear systems, Stein identity, eq (14), IPG-CV eq (15), and a tiny posterior smoke."""

import jax
import jax.numpy as jnp

from ipg.gmm import (
    posterior_gmm_linear_gaussian,
    probability_flow_drift,
    qt_score,
    sample_qt,
)
from ipg.ipg import (
    centering_matrix,
    ipg_phi_and_u,
    make_u_fn,
    run_ipg,
    solve_phi_ipg,
    solve_phi_ipg_cv,
)
from ipg.kernel import median_bandwidth, stein_apply, stein_gram
from ipg.reward import gt, r_and_grad


def test_stein_identity_mean_near_zero():
    key = jax.random.PRNGKey(0)
    n, d = 1024, 3
    X = jax.random.normal(key, (n, d))
    scores = -X

    def u_fn(x):
        return x

    su = jax.vmap(lambda x, s: stein_apply(u_fn, x, s))(X, scores)
    # S_p u = d - ||x||^2 for u(x)=x, p=N(0,I). Population mean is 0.
    assert jnp.abs(jnp.mean(su)) < 0.2


def test_ipg_linear_system_residual():
    key = jax.random.PRNGKey(4)
    X = jax.random.normal(key, (8, 2))
    scores = -X
    h = 0.7
    g = jnp.arange(8.0)
    g = g - jnp.mean(g)
    lam = 1e-3
    xi = stein_gram(X, scores, h)
    phi = solve_phi_ipg(xi, g, lam)
    resid = xi @ phi / X.shape[0] + lam * phi + g
    assert jnp.allclose(resid, 0.0, atol=1e-9)


def test_eq14_weight_residual_is_minus_lambda_phi():
    key = jax.random.PRNGKey(5)
    X = jax.random.normal(key, (10, 2))
    scores = -X
    h = median_bandwidth(X)
    g_raw = jnp.sin(jnp.linspace(0.0, 3.0, 10))
    lam = 1e-3
    phi, coeff, u, xi = ipg_phi_and_u(X, scores, g_raw, h, lam, cv=False)
    g = g_raw - jnp.mean(g_raw)
    n = X.shape[0]
    assert jnp.allclose(xi @ phi / n + lam * phi + g, 0.0, atol=1e-8)

    u_fn = make_u_fn(X, scores, coeff, h)
    su = jax.vmap(lambda x, s: stein_apply(u_fn, x, s))(X, scores)
    # S_p u at particles equals xi phi / N, so eq (14): S_p u + g = -lambda phi
    assert jnp.allclose(su, xi @ phi / n, rtol=1e-6, atol=1e-6)
    lhs = su + g_raw - jnp.mean(g_raw)
    assert jnp.allclose(lhs, -lam * phi, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(u, jax.vmap(u_fn)(X), rtol=1e-8, atol=1e-8)


def test_ipg_cv_eq15_and_centered_residual():
    key = jax.random.PRNGKey(6)
    X = jax.random.normal(key, (9, 2))
    scores = -X
    h = 0.6
    g_raw = jnp.linspace(-1.0, 2.0, 9)
    lam = 1e-3
    phi, coeff, u, xi = ipg_phi_and_u(X, scores, g_raw, h, lam, cv=True)
    n = X.shape[0]
    pi = centering_matrix(n, dtype=X.dtype)
    g = g_raw - jnp.mean(g_raw)
    resid = (pi @ xi @ pi) @ phi / n + lam * phi + g
    assert jnp.allclose(resid, 0.0, atol=1e-8)
    assert jnp.allclose(coeff, pi @ phi, atol=1e-10)

    phi_direct = solve_phi_ipg_cv(xi, g, lam)
    assert jnp.allclose(phi, phi_direct, atol=1e-8)

    u_fn = make_u_fn(X, scores, coeff, h)
    su = jax.vmap(lambda x, s: stein_apply(u_fn, x, s))(X, scores)
    # eq (16): S_p u(Xi) + g_t(Xi) - mean(S_p u + g_t) = -lambda phi_i
    lhs = su + g_raw - jnp.mean(su + g_raw)
    assert jnp.allclose(lhs, -lam * phi, rtol=1e-5, atol=1e-5)


def test_unweighted_ipg_beats_prior_mean_error():
    """2-component 2D GMM, linear-Gaussian reward. Seed-fixed, loose bound."""
    key = jax.random.PRNGKey(0)
    means = jnp.array([[-2.0, 0.0], [2.0, 0.0]])
    variance = 0.25
    log_weights = jnp.log(jnp.array([0.5, 0.5]))
    A = jnp.array([[1.0, 0.0]])
    y = jnp.array([2.0])
    sigma_obs = 0.35
    n_particles = 32
    n_steps = 40
    lam = 1e-3
    sigma_t = 3.0

    post_mean, _, _, _ = posterior_gmm_linear_gaussian(
        means, variance, log_weights, A, y, sigma_obs
    )
    prior_mean = jnp.einsum("k,kd->d", jax.nn.softmax(log_weights), means)
    prior_err = jnp.linalg.norm(prior_mean - post_mean)

    key, k0, krun = jax.random.split(key, 3)
    X0 = sample_qt(k0, n_particles, 0.0, means, variance, log_weights)
    times = jnp.linspace(0.0, 1.0, n_steps + 1)

    def score_q_fn(X, t):
        return qt_score(X, t, means, variance, log_weights)

    def v_fn(X, t):
        return probability_flow_drift(X, t, means, variance, log_weights)

    def grad_r_fn(X, t):
        return r_and_grad(X, t, A, y, sigma_obs)[1]

    def g_fn(X, t, v):
        return gt(X, t, v, A, y, sigma_obs)

    X = run_ipg(
        krun, X0, times, sigma_t, lam, score_q_fn, v_fn, grad_r_fn, g_fn, cv=False
    )
    ipg_err = jnp.linalg.norm(jnp.mean(X, axis=0) - post_mean)
    assert prior_err > 1.0
    assert ipg_err < prior_err
