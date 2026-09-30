"""Kernel identities: RBF gradients, Stein Gram symmetry, median heuristic eq (39)."""

import jax
import jax.numpy as jnp

from ipg.kernel import (
    grad_y_rbf,
    median_bandwidth,
    median_pairwise_distance,
    rbf,
    rbf_matrix,
    stein_gram,
)


def test_median_bandwidth_eq39():
    X = jnp.array([[0.0], [1.0], [4.0]])
    med = median_pairwise_distance(X)
    expected_med = jnp.median(jnp.array([1.0, 4.0, 3.0]))
    assert jnp.allclose(med, expected_med)
    n = X.shape[0]
    h = median_bandwidth(X)
    assert jnp.allclose(h, (med ** 2) / jnp.log(n))


def test_grad_y_rbf_matches_autodiff():
    x = jnp.array([0.3, -1.1, 0.7])
    y = jnp.array([-0.4, 0.2, 1.5])
    h = 0.8
    analytic = grad_y_rbf(x, y, h)
    auto = jax.grad(lambda yy: rbf(x, yy, h))(y)
    assert jnp.allclose(analytic, auto, rtol=1e-10, atol=1e-10)


def test_stein_gram_symmetric_and_psd():
    key = jax.random.PRNGKey(2)
    X = jax.random.normal(key, (6, 3))
    scores = -X
    h = 0.5
    xi = stein_gram(X, scores, h)
    assert xi.shape == (6, 6)
    assert jnp.allclose(xi, xi.T, rtol=1e-10, atol=1e-10)
    evals = jnp.linalg.eigvalsh(xi)
    assert jnp.min(evals) >= -1e-8
    K = rbf_matrix(X, h)
    assert jnp.allclose(K, K.T)
    assert jnp.allclose(jnp.diag(K), 1.0)
