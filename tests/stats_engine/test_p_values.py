import numpy as np
import pytest

from clique_homology.stats_engine import p_values

def make_random_null_dist(n, m):
    """
    Generates a matrix of size m x n with random integers between 1 and 10.
    Returns a list of numpy arrays, where each array represents a row.
    """
    # Generate a 2D numpy array with random integers
    # 1 is inclusive, 11 is exclusive (so values range 1-10)
    matrix_2d = np.random.randint(0, 11, size=(m, n))

    # Convert the 2D array into a list of 1D arrays (rows)
    return list(matrix_2d)


def generate_random_observation(m):
    """
    Generates a 1D numpy array of length n with random integers between 0 and 10.
    """
    # 0 is inclusive, 11 is exclusive (so values range 0-10)
    return np.random.randint(0, 11, size=m)


def test_p_values_basic():
    random_obs = generate_random_observation(50)
    random_null_dist = make_random_null_dist(50, 50)

    p_val, d2_obs, d2_null = p_values.calculate_p_vector(random_obs, random_null_dist)

    assert 0.0 <= p_val <= 1.0
    assert d2_obs >= 0.0


def test_get_mahalanobis_matches_manual_value() -> None:
    vector = np.array([2.0, 3.0])
    mean = np.array([1.0, 1.0])
    inv_cov = np.array([[1.0, 0.0], [0.0, 2.0]])

    observed = p_values.get_mahalanobis(vector, mean, inv_cov)
    assert observed == pytest.approx(9.0)


def test_calculate_p_vector_basic_properties() -> None:
    obs = np.array([1.0, 2.0])
    null = np.array([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5], [1.2, 2.2]])

    p_val, d2_obs, d2_null = p_values.calculate_p_vector(obs, null)

    assert 0.0 <= p_val <= 1.0
    assert isinstance(d2_obs, float)
    assert isinstance(d2_null, np.ndarray)
    assert d2_null.shape == (4,)


def test_calculate_p_vector_handles_singular_covariance() -> None:
    obs = np.array([2.0, 2.0])
    null = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    p_val, d2_obs, d2_null = p_values.calculate_p_vector(obs, null)

    assert 0.0 <= p_val <= 1.0
    assert d2_obs >= 0.0
    assert np.all(d2_null >= 0.0)


def test_calculate_p_vector_supports_1d_observations() -> None:
    obs = np.array([2.0])
    null = np.array([[1.0], [2.0], [3.0], [4.0]])

    p_val, d2_obs, d2_null = p_values.calculate_p_vector(obs, null)

    assert 0.0 <= p_val <= 1.0
    assert isinstance(d2_obs, float)
    assert d2_null.shape == (4,)


def test_calculate_p_vector_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        p_values.calculate_p_vector(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]))
