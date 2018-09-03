"""
Prior covariance operators
==========================

Methods for computing prior covariance matrices explicitly, by computing distances
between samples, then evaluating the covariance function.  For speed, uses vectorized
numpy operations exculsively.

Implemented covariance functions:

- Squared-exponential (with derivatives for gradient-enhanced Kriging)
- Matern with nu=1/2 (non-differentiable), 3/2 (once diff'ble), and 5/2 (twice diff'ble).
- Identity.

All covariances have unit standard-deviation; results should be multiplied by the 
desired \sigma**2.
"""

import numpy as np

def compute_delta_vectors(xi1, xi2):
    """
    Return vectors between all points in xi2 to all points in xi1, d-dimensions.  

	Args:
	  xi1 (ndarray): Sample locations in d-dimensions, shape `(n1, d)`.
	  xi2 (ndarray): Sample locations in d-dimensions, shape `(n2, d)`.  Note, `xi1` and
	                 `xi2` may be the same array.
	Return:
	  out (ndarray): Vectors connecting every point in `xi2` with every point in `xi1` (in
	                 that direction).  Shape `(n1,n2,d)`, ordering identical to `xi1`, `xi2`.
    """
    assert xi1.shape[1] == xi2.shape[1]
    n1, d = xi1.shape
    n2 = xi2.shape[0]
    return np.einsum('ik,j', xi1, np.ones(n2)) - np.einsum('i,jk', np.ones(n1), xi2)


def compute_distances_squared(xi1, xi2):
    """
    Return matrix of squared Euclidian distance between all pairs of points in d-dimensions.
	
	Args:
	  xi1 (ndarray): Sample locations in d-dimensions, shape `(n1, d)`.
	  xi2 (ndarray): Sample locations in d-dimensions, shape `(n2, d)`.
	Return:
	  out (ndarray): Shape `(n1,n2)`.	
    """
    return np.sum(compute_delta_vectors(xi1, xi2) ** 2, axis=2)


def covariance_squaredexponential(xi1, xi2, gamma):
    """
    Compute prior covariance matrix P, shape `(n1, n2)` with squared-exponential
	covariance and unit standard-deviation.

	Args:
      xi1 (ndarray): Vector of sample locations, shape `(n1, d)`
      xi2 (ndarray): Vector of sample locations, shape `(n2, d)`
      gamma (float): Correlation function scale parameter - corresponds to 1-standard-
                     deviation for a Gaussian.
	Return:
	  out (ndarray): Covariance matrix for squared-exponential with unit sigma.
    """
    theta = 1. / (2 * gamma ** 2)
    return np.exp(-theta * compute_distances_squared(xi1, xi2))


def covariance_squaredexponential_dxi(xi1, xi2, gamma, P00):
    """
	Compute matrices of derivatives of squared-exponential covariance function.  Return
	P_01 and P_10 with dimensions of (n1 x d*n2), (n1*d x n2), corresponding to
	differentiation with respect to `xi2` and `xi1` respectively.

	Args:
      xi1 (ndarray): Vector of sample locations, shape `(n1, d)`
      xi2 (ndarray): Vector of sample locations, shape `(n2, d)`
      gamma (float): Correlation function scale parameter - corresponds to 1-standard-
                     deviation for a Gaussian.
      P00 (ndarray): Output of `covariance_squaredexponential()` with identical `xi1`,`xi2`.
             	     Saves recalculation.
	Return:
	  P_01 (ndarray): Covariance matrix differentiated wrt xi2, shape `(n1, n2*d)`
	  P_10 (ndarray): Covariance matrix differentiated wrt xi1, shape `(n1*d, n2)`

	"""
    assert xi1.shape[1] == xi2.shape[1]
    n1, d = xi1.shape
    n2 = xi2.shape[0]
    theta = 1. / (2 * gamma ** 2)
    tmp0 = 2. * theta * compute_delta_vectors(xi1, xi2) * np.einsum('ij,k', P00, np.ones(d))

    out1 = np.zeros((n1, d * n2))  ### Flatten the 3-tensor in 2 different ways.
    out2 = np.zeros((n1 * d, n2))
    for i in range(d):
        out1[:, i::d] = tmp0[:, :, i]
        out2[i::d, :] = -tmp0[:, :, i]
    return out1, out2


def covariance_squaredexponential_dxidxi(xi1, xi2, gamma, P00):
    """
	Compute matrix of 2nd-derivatives of squared-exponential covariance function.  Return
    P_11 (n1 d x n2 d), covariance differentiated wrt both xi1 and xi2.

	Args:
      xi1 (ndarray): Vector of sample locations, shape `(n1, d)`
      xi2 (ndarray): Vector of sample locations, shape `(n2, d)`
      gamma (float): Correlation function scale parameter - corresponds to 1-standard-
                     deviation for a Gaussian.
      P00 (ndarray): Output of `covariance_squaredexponential()` with identical `xi1`,`xi2`.
             	     Saves recalculation.
	Return:
      P11 (ndarray): Covariance matrix differentiated wrt xi1,xi2, shape `(n1*d, n2*d)`.
    """
    assert xi1.shape[1] == xi2.shape[1]
    n1, d = xi1.shape
    n2 = xi2.shape[0]
    theta = 1. / (2 * gamma ** 2)
    tmp0 = compute_delta_vectors(xi1, xi2)
    tmp1 = np.einsum(
        'i,j,kl', np.ones(n1), np.ones(n2), np.identity(d)
    ) - 2. * theta * np.einsum('ijk,ijl->ijkl', tmp0, tmp0)
    tmp2 = 2. * theta * tmp1 * np.einsum('ij,k,l', P00, np.ones(d), np.ones(d))

    P11 = np.zeros(
        (d * n1, d * n2)
    )  ### Flatten the 4-tensor to an (d*n1 x d*n2) matrix
    for i in range(d):
        for j in range(d):
            P11[i::d, j::d] = tmp2[:, :, i, j]
    return P11


def covariancefn_identity(xi1, xi2, gamma):
    """White-noise covariance - no correlation between neighbouring points"""
    d = np.sqrt(compute_distances_squared(xi1, xi2))
    return np.where(d < 1.e-8, 1.,0.)

def covariancefn_matern_12(xi1, xi2, gamma):
    """Matern nu=1/2 - continuous, non-differentiable covariance - rough"""
    d = np.sqrt(compute_distances_squared(xi1, xi2))
    return np.exp(-d/gamma)

def covariancefn_matern_32(xi1, xi2, gamma):
    """Matern nu=3/2 - continuous, non-differentiable covariance - smoother"""
    d = np.sqrt(compute_distances_squared(xi1, xi2))
    return (1+np.sqrt(3)*d/gamma)*np.exp(-np.sqrt(3)*d/gamma)

def covariancefn_matern_52(xi1, xi2, gamma):
    """Matern nu=5/2 - continuous, once-differentiable covariance - smoothest"""
    d = np.sqrt(compute_distances_squared(xi1, xi2))
    return (1+np.sqrt(5)*d/gamma + 5*d**2/(3*gamma**2))*np.exp(-np.sqrt(5)*d/gamma)

