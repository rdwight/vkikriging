"""
Universal Kriging - version 3 (`kriging_v3`)
============================================

Universal Kriging in d-dimensions.  This differs from `kriging_v1` and `kriging_v2`
which implement only simple Kriging.

"""
import numpy as np

from .mylib import Timing
from .covariance import covariance_squaredexponential, covariance_squaredexponential_dxi, covariance_squaredexponential_dxidxi


def F_linear(xi):
    """
    Basis functions for parameterization of non-stationary mean.  This version of the
    function implements a linear basis.

    Args:
      xi (ndarray): Coordinates of points in parameter space, shape `(n, d)`
    Return:
      out (ndarray): Matrix F shape `(n, M)`, where `M` is the number of basis functions.
    """
    n, d = xi.shape
    return np.hstack((np.ones((n, 1)), xi))


def dF_linear(xi):
    """
    Derivatives of basis functions defined in F_linear().  (Would be) needed for
    non-stationary mean with GEK.

    Args:
      xi (ndarray): Coordinates of points in parameter space, shape `(n, d)`
    Return:
      out (ndarray): Tensor of derivatives, shape `(n, M, d)`.
    """
    n, d = xi.shape
    M = d + 1  # Must be equal to M = F_linear(xi).shape[1]
    out = np.zeros((n, M, d))
    for i in range(n):
        out[i, 1:, :] = np.identity(d)
    return out


def kriging(xi, x, observed, sigma_y, F_mean, sd_x, gamma):
    """
	Function kriging_v1.kriging() modified for universal Kriging (spatially variable
	mean based on general regression).  This is achived by introducing a function-basis
	F (e.g. `F_linear()`) for representing the *variable* mean, and new unknown vector
	\lambda.  The mean is then \lambda . F, and the unknown vector x is augmented:

      x_a = [x, \lambda],

    given which the new observation operator is:

      H_a = [H, F].
 
    The prior mean (of the Gaussian process) is now always zero, instead of specifying
	the mean `mu_x`, the function-basis must be specified in the argument `F_mean`.

    Args:  
      xi (ndarray): Sample locations (both observations and predictions), shape `(n,d)`
      x (ndarray): Sample values (values not at observation locations are not used).  
                   Shape `n`.
      observed (ndarray): Bool array specifying which values are observed.  Shape `n`,
                          `True` - observed, `False` - not observed.
      sigma_y (float): Standard-deviation of observation error.  Scalar.
      F_mean (function): A function in the template of F_linear(), providing a basis for 
	                     the description of the non-stationary mean (in d-dimensions).
      sd_x (float): (Sample) standard-deviation of the approximated function,
                    used in the prior.  Scalars.
      gamma (float): Correlation coefficient in all directions.  Scalar.

    Return:
      out (dict): Dictionary of prior and posterior statistics.

	"""
    ### Determine problem dimensions from input.
    n, d = xi.shape  #
    H = np.identity(n)[observed]  # Observation operator
    y = np.dot(H, x)  # Observations
    m = y.size  # Number of observations
    F = F_mean(xi)  # Basis for non-stationary mean
    Fy = F[observed]  # Restricted to observation locations
    M = F.shape[1]  # Size of basis
    Ha = np.hstack((H, Fy))  # Augmented observation operator

    ### Observation error covar matrix
    R = np.diag(np.ones(m) * max(sigma_y, 1.e-4) ** 2)

    ### Prior mean and covariance at the sample locations.  Augmented
    ### with priors of coefficients (TODO: atm normal dist with large
    ### std, should be non-informative).
    t = Timing()
    mua_prior = np.zeros(n + M)
    Pa = np.zeros((n + M, n + M))
    Pa[:n, :n] = sd_x ** 2 * covariance_squaredexponential(xi, xi, gamma)
    Pa[n:, n:] = 1.e6 * np.identity(M)  # Prior on mean coefficients
    t.monitor('Build prior covariance')

    ### The gain matrix.
    Aa = R + np.dot(Ha, np.dot(Pa, Ha.T))
    Ka = np.dot(Pa, np.dot(Ha.T, np.linalg.inv(Aa)))
    t.monitor('Invert K')

    ### Posterior mean and covariance (prediction):
    #   E(x|y) ("predictor")
    muahat = mua_prior + np.dot(Ka, y - np.dot(Ha, mua_prior))
    muhat = np.dot(F, muahat[n:]) + muahat[:n]
    t.monitor('Evaluate posterior mean')
    #   Cov(x|y) ("mean-squared error estimator")
    covahat = np.dot(np.identity(n + M) - np.dot(Ka, Ha), Pa)
    covPhat, covFhat = covahat[:n, :n], covahat[n:, n:]
    # covhat  = np.dot(F, np.dot(covFhat, F.T)) + covPhat
    covhat = covPhat
    t.monitor('Evaluate posterior covariance')

    ### Return all this statistical information.
    return {
        'mua_prior': mua_prior,
        'cov_prior': Pa,  # Prior (augmented)
        'muahat': muahat,
        'covahat': covahat,  # Posterior (augmented)
        'muhat': muhat,
        'Sigmahat': covhat,  # Posterior
    }
