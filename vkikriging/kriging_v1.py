"""
Kriging and Gradient-Enhanced Kriging - version 1 (`kriging_v1`)
================================================================

Simple Kriging and GEK in d-dimensions.

Implementation follows the Bayesian derivation of Kriging exactly, with the same symbols
as in the supplied tutorial.  No efforts are made for efficiency.  For a (slightly more)
efficient implementation of simple Kriging see `kriging_v2.py`.  For an implementation
of universal Kriging (i.e. with regression) see `kriging_v3.py`.
"""
import numpy as np
import copy

from .mylib import gek_composite, gek_separate, Timing
from .covariance import covariance_squaredexponential, \
	covariance_squaredexponential_dxi, \
	covariance_squaredexponential_dxidxi


def kriging(xi, x, observed, sigma_y, mu_x, sd_x, gamma, verbose=True):
    """
	Kriging in d-dimensions for a single variable - following the Bayesian derivation and
    notation.  The following assumptions are limitations of the current implementation
    which may be lifted without substancially modifying the derivation (just coding).

    Assumptions:
      - Constant regression at specified mean mu.
      - Same constant error for all observations (sigma_y)
      - Stationarity of the Gaussian process (constant standard 
        deviation of the prior).

    Args:  
      xi (ndarray): Sample locations (both observations and predictions), shape `(n,d)`
      x (ndarray): Sample values (values not at observation locations are not used).  
                   Shape `n`.
      observed (ndarray): Bool array specifying which values are observed.  Shape `n`,
                          `True` - observed, `False` - not observed.
      sigma_y (float): Standard-deviation of observation error.  Scalar.
      mu_x, sd_x (float): (Sample) mean and standard-deviation of the approximated 
	                      function, used in the prior.  Scalars.
      gamma (float): Correlation coefficient in all directions.  Scalar.

    Return:
      out (dict): Dictionary of prior and posterior statistics.

    NOTE: This intended to be a simple teaching implementation, it is not efficient.  In
    particular is is not necessary to construct the prior covariance for unobserved
    locations with respect to each other as we do here.  A better implementation can be
    obtained with a little linear algebra.
	"""
    ### Determine problem dimensions from input.
    n, d = xi.shape  #
    H = np.identity(n)[observed]  # Observation operator
    y = np.dot(H, x)  # Observations
    m = y.size  # Number of observations

    ### Observation error covar matrix
    R = np.diag(np.ones(m) * max(sigma_y, 1.e-4) ** 2)

    ### Prior mean and covariance at the sample locations.
    t = Timing()
    mu_prior = mu_x * np.ones(n)
    P = sd_x ** 2 * covariance_squaredexponential(xi, xi, gamma)
    t.monitor('Build prior covariance')

    ### The gain matrix.
    A = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(P, np.dot(H.T, np.linalg.inv(A)))
    t.monitor('Invert K')

    ### Posterior mean and covariance (prediction):
    #   E(x|y) ("predictor")
    muhat = mu_prior + np.dot(K, y - np.dot(H, mu_prior))
    t.monitor('Evaluate posterior mean')
    #   Cov(x|y) ("mean-squared error estimator")
    Sigmahat = np.dot(np.identity(n) - np.dot(K, H), P)
    t.monitor('Evaluate posterior covariance')

    ### Return all this statistical information.
    return {
        'mu_prior': mu_prior,
        'cov_prior': P,  # Prior
        'muhat': muhat,
        'Sigmahat': Sigmahat,
    }  # Posterior


def gek(xi, x, dx, observed, sigma_y, sigma_dy, mu_x, sd_x, gamma, verbose=True):
    """
    Simple Gradient-Enhanced Kriging (GEK) in d-dimensions for a single
    variable - following the Bayesian derivation and notation.  Constant regression at 
	given mean mu, mean gradient assumed zero.

    Assumptions (as for kriging() and...):
      - Gradients observations colocated with value observations.
      - Gradients in all d directions observed at all observation locations.
      - Constant gradient error for all locations and directions.

    Args:  
      xi (ndarray): Sample locations (both observations and predictions), shape `(n,d)`
      x (ndarray): Sample values (values not at observation locations are not used).  
                   Shape `n`.
      dx (ndarray): Sample gradients, shape `(n, d)`.
      observed (ndarray): Bool array specifying which values are observed.  Shape `n`,
                          `True` - observed, `False` - not observed.
      sigma_y (float): Standard-deviation of observation error.  Scalar.
      sigma_dy (float): Standard-deviations of observed gradient error.  Scalar.
      mu_x, sd_x (float): (Sample) mean and standard-deviation of the approximated 
	                      function, used in the prior.  Scalars.
      gamma (float): Correlation coefficient in all directions.  Scalar.

    Return:
      out (dict): Dictionary of prior and posterior statistics.
    """
    # Create extended variable vectors containing values then gradients.  The ordering used is:
    #   (x_1, x_2, ... x_n, dx_1/dxi_1, ... dx_1/dxi_d, dx_2/dxi_1, ..., dx_n/dxi_d)
    # The total size is n*(d+1).
    n, d = xi.shape  # Number of locations, dimension
    xc = gek_composite(x, dx)  # Extended sample values (defines ordering)

    observedc = copy.copy(observed)  # Extended observed array
    for i in range(n):
        observedc = np.hstack((observedc, observed[i] * np.ones(d, dtype=bool)))
    Hc = np.identity((d + 1) * n)[observedc]  # Extended observation operator
    yc = np.dot(Hc, xc)  # Extended observation vector
    m = yc.size // (d + 1)  # Number of observation locations

    assert (
        xc.shape[0] == (d + 1) * n
    ), 'Gradients at all observed locations not available'
    assert observedc.shape[0] == (d + 1) * n, 'Implementation error'
    assert (yc.size % (d + 1)) == 0, 'Implementation error'

    # Extended observation error covar matrix
    Rc = np.diag(
        np.hstack(
            (
                np.ones(m) * max(sigma_y, 1.e-4) ** 2,
                np.ones(m * d) * max(sigma_dy, 1.e-4) ** 2,
            )
        )
    )

    # Prior mean and covariance at the sample locations.
    t = Timing()
    mu_prior = np.zeros((d + 1) * n)  # Assume zero gradient mean
    mu_prior[:n] = mu_x

    Pc00 = covariance_squaredexponential(xi, xi, gamma)
    Pc01, Pc10 = covariance_squaredexponential_dxi(xi, xi, gamma, Pc00)
    Pc11 = covariance_squaredexponential_dxidxi(xi, xi, gamma, Pc00)

    # Build prior covariance matrix P from sub-matrices.
    Pc = np.zeros(((d + 1) * n, (d + 1) * n))
    Pc[:n, :n], Pc[:n, n:], Pc[n:, :n], Pc[n:, n:] = Pc00, Pc01, Pc10, Pc11
    Pc *= sd_x ** 2
    t.monitor('Build prior covariance')

    # Now everything is exactly as before, with the extended vectors:
    # The Kalman gain matrix K.
    Ac = Rc + np.dot(Hc, np.dot(Pc, Hc.T))
    Kc = np.dot(Pc, np.dot(Hc.T, np.linalg.inv(Ac)))
    t.monitor('Invert K')

    # Posterior mean and covariance (prediction):
    #   E(xc|yc) ("predictor")
    muhat = mu_prior + np.dot(Kc, yc - np.dot(Hc, mu_prior))
    t.monitor('Evaluate posterior mean, muhat')
    #   Cov(xc|yc) ("mean-squared error estimator")
    Sigmahat = np.dot(np.identity(n * (d + 1)) - np.dot(Kc, Hc), Pc)
    t.monitor('Evaluate posterior covariance, Sigmahat')

    # Return all statistical information.
    return {
        'mu_prior': mu_prior,
        'cov_prior': Pc,  # Prior
        'muhat': muhat,
        'Sigmahat': Sigmahat,
    }  # Posterior

