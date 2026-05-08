import numpy as np
import pytest
from unittest.mock import patch

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

def test_calculate_p_vector_linalg_error_fallback():
    """
    Tests the LinAlgError exception block by forcing cho_factor to fail,
    ensuring the pinvh fallback calculates distances correctly.
    """
    obs_betti = np.array([1.0, 2.0])
    # Create a simple null matrix (n permutations x m dimensions)
    null_betti_matrix = np.array([
        [1.1, 2.1], 
        [0.9, 1.9], 
        [1.0, 2.0]
    ])

    # Patch cho_factor to simulate a singular matrix error
    with patch('clique_homology.stats_engine.p_values.cho_factor', side_effect=np.linalg.LinAlgError):
        p_val, d2_obs, d2_null = p_values.calculate_p_vector(obs_betti, null_betti_matrix)

    # Assert that the function still returns the expected data types and shapes
    assert isinstance(p_val, float)
    assert 0.0 <= p_val <= 1.0
    assert isinstance(d2_obs, float)
    assert isinstance(d2_null, np.ndarray)
    assert d2_null.shape == (3,)  # Matches the number of rows in null_betti_matrix


def test_validate_p_vector_inputs_exceptions():
    """Tests all ValueError branches in the input validation."""
    
    # 1. obs_betti is not 1D
    with pytest.raises(ValueError, match="obs_betti must be a 1D array"):
        p_values._validate_p_vector_inputs(np.array([[1, 2]]), np.array([[1, 2], [3, 4]]))

    # 2. null_betti_matrix is not 2D
    with pytest.raises(ValueError, match="null_betti_matrix must be a 2D array"):
        p_values._validate_p_vector_inputs(np.array([1, 2]), np.array([1, 2, 3]))

    # 3. null_betti_matrix is empty
    with pytest.raises(ValueError, match="null_betti_matrix must contain at least one row"):
        p_values._validate_p_vector_inputs(np.array([1, 2]), np.empty((0, 2)))

    # 4. Shape mismatch between obs and null
    with pytest.raises(ValueError, match="obs_betti shape must match null_betti_matrix column count"):
        p_values._validate_p_vector_inputs(np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6]]))