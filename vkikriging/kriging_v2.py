"""
Kriging and Gradient-Enhanced Kriging - version 2 (`kriging_v2`)
================================================================

Simple Kriging and GEK in d-dimensions.

This differs from `kriging_v1` mainly in that sample locations are calculated
separately from prediction locations - the implementation does not follow the Bayesian
derivation as closely, but this saves a lot of time and memory.
"""
import numpy as np
from scipy.linalg import solve_triangular

from . import mylib, covariance
from .mylib import Timing
from .covariance import covariance_squaredexponential, \
	covariance_squaredexponential_dxi, \
	covariance_squaredexponential_dxidxi


class Kriging:
	"""
	Simple Kriging in d-dimensions for a single variable.  Construction of the
	surrogate happens in the __init__() call, after which predictions
	can be made with multiple predict() calls.

	Assumptions:
	  - Constant regression at specified mean mu.
	  - Same constant error for all observations (sigma_y)
	  - Stationarity of the Gaussian process (constant standard 
		deviation of the prior).

	Args:  
	  xi (ndarray): Sample locations (observations only), shape `(n,d)`
	  x (ndarray): Sample values. Shape `n`.
	  sigma_y (float): Standard-deviation of observation error.	 Scalar.
	  mu_x, sd_x (float): (Sample) mean and standard-deviation of the approximated function,
						  used in the prior.  Scalars.
	  gamma (float): Correlation coefficient in all directions.	 Scalar.
	"""

	def __init__(self, xi, y, sigma_y, mu_x, sd_x, gamma):
		### Determine problem dimensions from input.
		assert xi.shape[0] == y.shape[0]
		self.xi, self.y = xi, y
		self.n, self.d = xi.shape
		self.sigma_y, self.mu_x, self.sd_x, self.gamma = sigma_y, mu_x, sd_x, gamma

		### The gain matrix A = R + P
		self.t = Timing()
		self.A = np.diag(
			np.ones(self.n) * max(sigma_y, 1.e-4) ** 2
		) + sd_x ** 2 * covariance_squaredexponential(xi, xi, self.gamma)
		self.t.monitor('Build prior covariance')
		self.L = np.linalg.cholesky(self.A)
		self.t.monitor('Cholesky factorization of A')

		### Assuming prior mean constant here.
		mu_prior = np.ones(self.n) * self.mu_x
		tmp = solve_triangular(self.L, self.y - mu_prior, lower=True)
		self.s = solve_triangular(self.L.T, tmp, lower=False)
		self.t.monitor('Solve for s')

	def predict(self, xip, posterior_cov=None):
		"""
		Predict response at locations xip in d-dimensions.
		  xip			- Prediction locations, array (m x d)
		  posterior_cov - None, return posterior mean only,
						  'diag', diagonal part only, array (m),
						  'full', full cov matrix, array (m x m).
		Return
		  muhat, [Sigmahat] - Posterior mean, covariance
		"""
		assert xip.shape[1] == self.d
		### Compute mean.
		self.t.reinit()
		P_pd = self.sd_x ** 2 * covariance_squaredexponential(xip, self.xi, self.gamma)
		self.t.monitor('Build prediction-data covariance')
		mu_prior = np.ones(xip.shape[0]) * self.mu_x
		muhat = mu_prior + np.dot(P_pd, self.s)
		self.t.monitor('Evaluate posterior mean')

		### Computation of posterior covariance is much more expensive than the mean,
		### only perform if required.
		if posterior_cov:
			LP = solve_triangular(self.L, P_pd.T, lower=True)
			if posterior_cov == 'diag':
				Sigmahat = self.sd_x ** 2 - np.sum(LP ** 2, axis=0)
			elif posterior_cov == 'full':
				P_pp = self.sd_x ** 2 * covariance_squaredexponential(xip, xip, self.gamma)
				Sigmahat = P_pp - np.dot(LP.T, LP)
			else:
				raise ValueError("Arg. posterior_cov should be None, 'diag' or 'full'")
			self.t.monitor('Evaluate posterior covariance')
			return muhat, Sigmahat
		else:
			return muhat


class GEK:
	"""
	Gradient-Enhanced Kriging (GEK) in d-dimensions for a single
	variable.  This differs from kriging_v1.gek() mainly in that
	sample locations are calculated separately from prediction
	locations - the implementation does not follow the Bayesian
	derivation as closely, but this saves a lot of time and memory.
	Construction of the surrogate happens in the __init__() call,
	after which predictions can be made with one or more predict()
	calls.

	Assumptions (as for Kriging class and...):
	  - Gradients observations colocated with value observations.
	  - Gradients in all d direcitons observed at all locations.
	  - Constant gradient error for all locations and directions.
	Constant regression at given mean mu, mean gradient assumed zero.

	"""

	def __init__(self, xi, y, dy, sigma_y, sigma_dy, mu_x, sd_x, gamma):
		### Determine problem dimensions from input.
		assert xi.shape[0] == y.shape[0]
		assert xi.shape == dy.shape
		self.xi, self.y, self.dy = xi, y, dy
		self.n, self.d = xi.shape
		n, d = self.n, self.d
		self.sigma_y, self.sigma_dy, self.mu_x, self.sd_x, self.gamma = (
			sigma_y,
			sigma_dy,
			mu_x,
			sd_x,
			gamma,
		)
		### The gain matrix A = R + P
		# Prior
		self.t = Timing()
		self.A = sd_x ** 2 * self.prior_cov(xi, xi, partial=False)
		# Likelihood
		R = self.composite(
			np.ones(n) * max(sigma_y, 1.e-4) ** 2,
			np.ones((n, d)) * max(sigma_dy, 1.e-4) ** 2,
		)
		self.A += np.diag(R)
		self.t.monitor('Build prior covariance')

		### Factorize A.
		self.L = np.linalg.cholesky(self.A)
		self.t.monitor('Cholesky factorization of A')

		### Assuming prior mean constant here (=> gradient mean zero)
		mu_prior = self.composite(np.ones(n) * mu_x, np.zeros((n, d)))
		tmp = solve_triangular(
			self.L, self.composite(self.y, self.dy) - mu_prior, lower=True
		)
		self.s = solve_triangular(self.L.T, tmp, lower=False)
		self.t.monitor('Solve for s')

	def predict(self, xip, posterior_cov=None, partial=False):
		"""
		Predict response at locations xip in d-dimensions.

		Args:
		  xip (ndarray): Prediction locations, shape `(m, d)`
		  posterior_cov (str):  None, return posterior mean only,
						  'diag', diagonal part only, array (m),
						  'full', full cov matrix, array (m x m).
		  partial (bool): Construct only part of the covariance matrix,
						  reduces cost, but allows output of values only,
						  not gradients, and not CoV.
		Return:
		  muhat, [Sigmahat] - Posterior mean, covariance
		"""
		if partial == True:
			assert posterior_cov == None
		assert xip.shape[1] == self.d
		m = xip.shape[0]
		self.t.reinit()
		P_pd = self.sd_x ** 2 * self.prior_cov(xip, self.xi, partial=partial)
		self.t.monitor('Build prediction-data covariance')
		if partial:
			mu_prior = np.ones(m) * self.mu_x
		else:
			mu_prior = self.composite(np.ones(m) * self.mu_x, np.zeros((m, self.d)))
		muhat = mu_prior + np.dot(P_pd, self.s)
		self.t.monitor('Evaluate posterior mean')

		# Computation of posterior covariance is much more expensive than the mean,
		# only perform if required.
		if posterior_cov:
			LP = solve_triangular(self.L, P_pd.T, lower=True)
			if posterior_cov == 'diag':
				P_pp_diag = self.composite(
					np.ones(m) * self.sd_x, np.zeros((m, self.d))
				)  ### RPD ???
				Sigmahat = P_pp_diag - np.sum(
					LP ** 2, axis=0
				)  ### RPD need to fix this...
			elif posterior_cov == 'full':
				P_pp = self.sd_x ** 2 * self.prior_cov(xip, xip, partial=False)
				Sigmahat = P_pp - np.dot(LP.T, LP)
			else:
				raise ValueError("Arg. posterior_cov should be None, 'diag' or 'full'")
			self.t.monitor('Evaluate posterior covariance')
			return muhat, Sigmahat
		else:
			return muhat

	def predict_split(self, xip, max_array=1e7):
		"""
		Same as predict(), but splits a very large job into multiple smaller jobs to
		save memory.  Arg. max_array specifies the maximum size of the array that should
		be created in the process.	Handy when doing Monte-Carlo on the GEK surface.  Only
		values, not gradients predicted.
		"""
		M = xip.shape[0]  # Number of prediction points
		n = self.n	# Number of support points
		row_size = (self.d + 1) * self.n  # Size of single row
		tot_size = (self.d + 1) * self.n * M  # Size of full matrix (partial version!)
		if max_array < row_size:
			raise ValueError("max_array must be bigger than row_size: %d" % row_size)
		stepsize = max_array // row_size	 # Integer!
		count = 0
		muhat = np.array([])
		while count < M:
			out = self.predict(
				xip[count : count + stepsize, :], posterior_cov=None, partial=True
			)  # Partial matrix only!
			count += stepsize
			muhat = np.hstack((muhat, out))
		return muhat

	def composite(self, x, dx):
		"""
		Create composite vector of values and gradients - return 1d
		vector xc.	This function defines the order of entries in
		composite vectors, and we must be consistent with prior_cov().
		The reverse of this is separate().
		"""
		assert x.shape[0] == dx.shape[0], 'Amount of derivative info wrong.'
		assert self.d == dx.shape[1], 'Dimension of derivative info wrong.'
		n = dx.shape[0]
		return np.hstack((x, dx.reshape(n * self.d)))

	def separate(self, xc):
		"""
		Map composite vector returned by GEK into individual value and
		derivate vectors.  This is useful for postprocessing output of
		GEK.predict() for plotting etc.	 Return x (n), dx (n x d).
		"""
		assert xc.ndim == 1 and xc.size % (self.d + 1) == 0, 'Dimension of input wrong.'
		n = xc.size // (self.d + 1)
		x = xc[:n]	# Values
		dx = xc[n:].reshape((n, self.d))  # Gradients
		return x, dx

	def prior_cov(self, xi1, xi2, partial=False):
		"""
		Construct the prior covariance matrix for GEK - from 4 parts.

		Args:
		  partial (bool): If True construct only the upper two blocks, sufficient
						  for prediction of values only (not gradients).
		Return:
		  P (ndarray): Prior covariance.
		"""
		assert self.d == xi1.shape[1]
		n1, d = xi1.shape
		n2 = xi2.shape[0]
		Pc00 = covariance_squaredexponential(xi1, xi2, self.gamma)
		if not partial:
			P = np.zeros(((d + 1) * n1, (d + 1) * n2))
			P[:n1, :n2] = Pc00	# The 4 parts.
			P[:n1, n2:], P[n1:, :n2] = covariance_squaredexponential_dxi(xi1, xi2, self.gamma, Pc00)
			P[n1:, n2:] = covariance_squaredexponential_dxidxi(xi1, xi2, self.gamma, Pc00)
		else:
			P = np.zeros((n1, (d + 1) * n2))
			P[:, :n2] = Pc00
			P[:, n2:],_ = covariance_squaredexponential_dxi(xi1, xi2, self.gamma, Pc00)
		return P
