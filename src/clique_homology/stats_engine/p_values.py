import numpy as np
from scipy.linalg import cho_factor, cho_solve, pinvh


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


def get_mahalanobis(vector, mean, inv_cov):
    diff = vector - mean
    return float(diff @ inv_cov @ diff.T)


def _validate_p_vector_inputs(obs_betti, null_betti_matrix) -> tuple[np.ndarray, np.ndarray]:
    clean_obs = np.asarray(obs_betti, dtype=float)
    clean_null = np.asarray(null_betti_matrix, dtype=float)

    if clean_obs.ndim != 1:
        raise ValueError("obs_betti must be a 1D array.")
    if clean_null.ndim != 2:
        raise ValueError("null_betti_matrix must be a 2D array.")
    if clean_null.shape[0] == 0:
        raise ValueError("null_betti_matrix must contain at least one row.")
    if clean_null.shape[1] != clean_obs.shape[0]:
        raise ValueError(
            "obs_betti shape must match null_betti_matrix column count."
        )

    return clean_obs, clean_null


def calculate_p_vector(obs_betti, null_betti_matrix):
    """
    obs_betti: 1D array (The C. elegans vector)
    null_betti_matrix: 2D array (n permutations x m dimensions)
    """
    
    clean_obs, clean_null = _validate_p_vector_inputs(obs_betti, null_betti_matrix)
    
    clean_null2 = np.vstack((clean_obs, clean_null))
    # 2. Calculate Null Statistics
    mu_null = np.mean(clean_null2, axis=0)
    cov_null = np.atleast_2d(np.cov(clean_null2, rowvar=False))
    
    # Inverse Covariance (Precision Matrix)
    # Use pseudo-inverse if n < m, otherwise standard inv
    try:
        inv_cov = np.linalg.inv(cov_null)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_null)

    # --- REGULARIZATION STEP ---
    # Add a small value to the diagonal of the covariance matrix
    # This prevents numerical instability and singular matrices (0-values)
    epsilon = 1e-1
    # epsilon = 1e-6 * max(np.max(np.diag(cov_null)), 1e-9)
    cov_null_reg = cov_null + np.eye(cov_null.shape[0]) * epsilon

    diff_obs = clean_obs - mu_null
    diff_null = clean_null - mu_null
    
    # Inverse Covariance: Use pseudo-inverse if n < m, 
    #   otherwise standard inv
    try:
        # Attempt Cholesky decomposition
        c, lower = cho_factor(cov_null_reg)

        # Solve for the observation
        y_obs = cho_solve((c, lower), diff_obs)
        d2_obs = np.dot(diff_obs, y_obs)

        # solve for all null vectors simultaneously
        # diff_null.T is (m x n), cho_solve returns (m x n), 
        # so we transpose back to (n x m)
        Y_null = cho_solve((c, lower), diff_null.T).T

        # Use einsum for optimized speed
        d2_null = np.einsum('ij,ij->i', diff_null, Y_null)

    except np.linalg.LinAlgError:
        # Fallback for highly singular/sparse matricies: Symmetric Psoudo-Inverse
        inv_cov = pinvh(cov_null_reg)

        # Calculate distances using the pseudo-inverse
        d2_obs = np.dot(diff_obs, inv_cov @ diff_obs)
        d2_null = np.einsum('ij,jk,ik->i', diff_null, inv_cov, diff_null)
    
    # P-Value
    p_val = float(np.mean(d2_null >= d2_obs))
    
    return p_val, float(d2_obs), d2_null


if __name__ == "__main__":
    random_obs = generate_random_observation(50)
    random_null_dist = make_random_null_dist(50, 50)

    print(random_obs)
    print(random_null_dist)

    p_val, d2_obs, d2_null = calculate_p_vector(random_obs, random_null_dist)

    print("P-value:", p_val)
    print("d2_obs:", d2_obs)
    print("d2_null:", d2_null)
