import numpy as np
from scipy.linalg import cho_factor, cho_solve, pinvh


def _validate_p_vector_inputs(
    obs_betti: np.ndarray, 
    null_betti_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Perform validation on the observed betti vector and the null dist.

    Args:
        obs_betti (ndarray): 1D numpy array (The C. elegans vector of betti
            numbers)
        null_betti_matrix (ndarray): 2D numpy array (n permutations x m
            dimensions)

    Raises:
        ValueError: raised if observed betti vector is something other than a 1D
            numpy array.
        ValueError: raised if the null distribution is something other than a
            2D numpy array.
        ValueError: Raised if the null distribution has zero rows.
        ValueError: Raised if the shape of the observed betti vector is
            different than the number of columns of the null distribution.

    Returns:
        tuple[np.ndarray, np.ndarray]: validated observed betti vector (1D numpy
            array), validated null distribution (2D numpy array)
    """

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


def calculate_p_value(
    obs_betti: np.ndarray, 
    null_betti_matrix: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Calculate the likelihood that the observed betti vector could have been
        produced by random chance, given a provided null distribution of random
        betti vector permutations.

    Args:
        obs_betti (ndarray): 1D numpy array (The C. elegans vector of betti
            numbers)
        null_betti_matrix (ndarray): 2D numpy array (n permutations x m
            dimensions)

    Returns:
        tuple[float, float, np.ndarray]: p-value (float), distance of observed
            betti vector from mean (float), and distances of null distribution
            vectors from mean (np.ndarray)
    """
    
    clean_obs, clean_null = _validate_p_vector_inputs(obs_betti, 
                                                        null_betti_matrix)
    
    # Calculate mean of the null dist and the covariance matrix of the null dist
    mu_null = np.mean(clean_null, axis=0)
    cov_null = np.atleast_2d(np.cov(clean_null, rowvar=False))

    # null_betti_matrix could be rank-deficient or sparse, either of which would
    #   render the covariance matrix singular and cause serious numerical
    #   stability problems. In order to avoid that, we regularize by a factor of
    #   epsilon dynamically based on the maximum diagonal value of the
    #   covariance matrix.
    epsilon = 1e-6 * max(np.max(np.diag(cov_null)), 1e-9)
    cov_null_reg = cov_null + np.eye(cov_null.shape[0]) * epsilon

    # Subtract the mean from the observed betti vector and from the null dist
    #   (we use this to calculate mahalanobis distance)
    diff_obs = clean_obs - mu_null
    diff_null = clean_null - mu_null
    
    # Since regularization guarantees the covariance matrix to be
    #   positive-definite and symmetric, we can use the Cholensky decomposition
    #   to bypass the sluggishness of the pseudo-inverse calculation
    try:
        # Attempt Cholesky decomposition
        c, lower = cho_factor(cov_null_reg)

        # Calculate mahalanobis distance of the observed betti vector
        y_obs = cho_solve((c, lower), diff_obs)
        d2_obs = np.dot(diff_obs, y_obs)

        # Calculate mahalanobis distance of each vector in the null dist.
        #   solve for all null vectors simultaneously. diff_null.T is (m x n),
        #   so cho_solve returns (m x n), so we transpose back to (n x m)
        Y_null = cho_solve((c, lower), diff_null.T).T
        d2_null = np.einsum('ij,ij->i', diff_null, Y_null) # einsum for speed

    # This should never happen, but if cov_null_reg happens to be extremely
    #   small, roundoff errors could potentially produce negative values
    #   which would cause errors in the cholensky decomposition. If this happens
    #   we instead use the pseudo-inverse of cov_null_reg 
    except np.linalg.LinAlgError:
        
        # Calculate pseudo-inverse
        inv_cov = pinvh(cov_null_reg)

        # Calculate mahalanobis distances using pseudo-inverse. Again, using
        #   einsum for optimized matrix multiplication speed
        d2_obs = np.dot(diff_obs, inv_cov @ diff_obs)
        d2_null = np.einsum('ij,jk,ik->i', diff_null, inv_cov, diff_null)
    
    # Calculate P-value
    p_val = float(np.mean(d2_null >= d2_obs))
    
    return p_val, float(d2_obs), d2_null
