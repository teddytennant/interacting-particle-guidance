"""Closed-form OU GMM scores (eqs 37-38) and linear-Gaussian posterior."""

import jax
import jax.numpy as jnp

from ipg.gmm import (
    OU_A,
    OU_B2,
    gmm_score,
    ou_gaussian_params,
    ou_gmm_params,
    posterior_gmm_linear_gaussian,
    probability_flow_drift,
    qt_score,
)
from ipg.reward import grad_R, linear_gaussian_loglik, r_and_grad


def test_one_component_ou_score_matches_gaussian():
    mean = jnp.array([0.4, -0.2, 0.1, 0.3])
    variance = 0.5
    means = mean[None, :]
    log_w = jnp.zeros((1,))
    x = jnp.array([0.1, 0.0, -0.2, 0.4])

    for t in (0.0, 0.4, 1.0):
        mean_t, var_t = ou_gaussian_params(mean, variance, t)
        analytic = -(x - mean_t) / var_t
        numeric = qt_score(x[None, :], t, means, variance, log_w)[0]
        assert jnp.allclose(numeric, analytic, rtol=1e-10, atol=1e-10)
        # data time t=1 is s=0: noising has not started
        if t == 1.0:
            assert jnp.allclose(mean_t, mean)
            assert jnp.allclose(var_t, variance)
            assert jnp.allclose(gmm_score(x, means, variance, log_w), -(x - mean) / variance)


def test_probability_flow_drift_eq38():
    means = jnp.array([[1.0, -0.5], [-0.7, 0.9]])
    variance = 0.4
    log_w = jnp.log(jnp.array([0.4, 0.6]))
    X = jnp.array([[0.2, 0.1], [-1.0, 0.3], [0.5, -0.8]])
    t = 0.3
    v = probability_flow_drift(X, t, means, variance, log_w)
    score = qt_score(X, t, means, variance, log_w)
    expected = OU_A * X + (OU_B2 / 2.0) * score
    assert jnp.allclose(v, expected)
    means_t, var_t = ou_gmm_params(means, variance, t)
    s = 1.0 - t
    assert jnp.allclose(means_t, means * jnp.exp(-OU_A * s))
    stat = OU_B2 / (2.0 * OU_A)
    var_expected = variance * jnp.exp(-2.0 * OU_A * s) + stat * (1.0 - jnp.exp(-2.0 * OU_A * s))
    assert jnp.allclose(var_t, var_expected)


def test_one_component_posterior_is_linear_gaussian():
    mean = jnp.array([0.2, -0.3])
    variance = 0.5
    means = mean[None, :]
    log_w = jnp.zeros((1,))
    A = jnp.array([[1.0, 0.2], [0.0, 1.0]])
    y = jnp.array([0.4, -0.1])
    sigma_obs = 0.3
    post_mean, mu_posts, log_w_post, sigma = posterior_gmm_linear_gaussian(
        means, variance, log_w, A, y, sigma_obs
    )
    s2 = sigma_obs ** 2
    prec = jnp.eye(2) / variance + (A.T @ A) / s2
    sigma_expected = jnp.linalg.inv(prec)
    mu_expected = sigma_expected @ (mean / variance + A.T @ y / s2)
    assert jnp.allclose(sigma, sigma_expected)
    assert jnp.allclose(post_mean, mu_expected)
    assert jnp.allclose(mu_posts[0], mu_expected)
    assert jnp.allclose(jnp.exp(log_w_post), jnp.array([1.0]))


def test_reward_schedule_beta_t_equals_t():
    X = jnp.array([[0.1, -0.2], [0.4, 0.5]])
    A = jnp.array([[1.0, 0.0]])
    y = jnp.array([0.2])
    sigma_obs = 0.5
    t = 0.3
    R = linear_gaussian_loglik(X, A, y, sigma_obs)
    gR = grad_R(X, A, y, sigma_obs)
    r, gr = r_and_grad(X, t, A, y, sigma_obs)
    assert jnp.allclose(r, t * R)
    assert jnp.allclose(gr, t * gR)
    # autodiff of R wrt x
    g_auto = jax.vmap(lambda x: jax.grad(lambda z: linear_gaussian_loglik(z, A, y, sigma_obs))(x))(X)
    assert jnp.allclose(gR, g_auto, rtol=1e-8, atol=1e-8)
