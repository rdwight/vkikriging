"""
Example surrogate modelling in 2d (`vkikriging.example2d`)
==========================================================

Class to do surrogate modelling and plotting of a function `f` of two variables, with 
derivative, another function `df`.
"""
import numpy as np
import matplotlib.pyplot as plt

from .mylib import (
	covariance_to_stddev,
	enter_debugger_on_uncaught_exception,
	iowrite,
	gek_separate,
	get_aspect,
)
from . import test_functions, sampling
from . import kriging_v1, kriging_v2, kriging_v3

class Example2d:
	"""
	Example demonstrating how to use kriging and gek, versions v1,v2 and v3, in 2d.
	For a function `f` (e.g. analytically defined), with gradient `df`.

	Args:
	  f (test_functions.TestFunction): Function to approximate, includes derivate,
	                                   and bounds in each coodinate direction.  See
	                                   `test_functions` for examples.
	  xi_samples (ndarray): Locations xi at which to sample `f` and `df`, shape `(N,2)`.
	  gamma (float): Correlation length in prior.
	  sigma_d, sigma_dg (float): Standard errors in observations and gradients.

	Example usage::

	  from vkikriging.test_functions import Camel
	  from vkikriging.sampling import sobol

	  # 1. Initialize the function to approxiate, samples and Kriging paramters
	  f = Camel()
	  xi = sobol(20, 2) * (f.xmax - f.xmin) + f.xmin
	  ex2d = Example2d(f, xi, gamma, sigma_d, sigma_dg)

	  # 2. Build the Kriging model, v1, v2 or v3.  For v1, v2 GEK model is also built.
	  ex2d.build_surrogate_v1()

	  # 3. Plot the model
	  ex2d.plot_contours()
	"""
	def __init__(self, f, xi_samples, gamma, sigma_d, sigma_dg):
		self.d = 2
		self.n_o = xi_samples.shape[0]
		self.xi_o = xi_samples

		# Define reconstruction locations - use rectangular grid.
		self.N = 41
		self.n_r = self.N ** 2
		self.xi_r1 = np.linspace(f.xmin[0], f.xmax[0], self.N)
		self.xi_r2 = np.linspace(f.xmin[1], f.xmax[1], self.N)
		A, B = np.meshgrid(self.xi_r1, self.xi_r2)
		self.xi_r = np.vstack((A.flatten(), B.flatten())).T

		# Put everything together - repeated coordinates are okay.
		self.n = self.n_o + self.n_r
		self.xi = np.vstack((self.xi_o, self.xi_r))

		# State which of the coordinates will be observed (in this case the first n_o).
		self.observed = np.zeros(self.n, dtype=bool)
		self.observed[:self.n_o] = True

		# Observe the function (we also observe the reconstruction locations for reference)
		self.x = np.zeros(self.n)
		self.dx = np.zeros((self.n, self.d))
		self.f = f
		for i, xi_i in enumerate(self.xi):
			self.x[i] = f.f(xi_i)
			self.dx[i, :] = f.df(xi_i)

		# Specify parameters of the prior
		self.gamma = gamma
		self.sigma_d, self.sigma_dg = sigma_d, sigma_dg	 # Obvervation error.
		# Sample mean + (conservative) stddev estimate.
		self.mu, self.std = np.mean(self.x), 3 * np.std(self.x)

	def build_surrogate_v1(self):
		print('Building simple Kriging model (version 1):')
		self.krig = kriging_v1.kriging(self.xi, self.x, self.observed, self.sigma_d, self.mu,
								  self.std, self.gamma)
		print('Building simple GEK model (version 1):')
		self.gek = kriging_v1.gek(
			self.xi, self.x, self.dx, self.observed, self.sigma_d, self.sigma_dg,
			self.mu, self.std, self.gamma
		)

	def build_surrogate_v2(self):
		# WORKING
		self.krig = kriging_v2.Kriging(xi_o, x[:n_o], sigma_d, mu, std, gamma)
		self.gek = kriging_v2.GEK(
				xi_o, x[:n_o], dx[:n_o, :], sigma_d, sigma_dg, mu, std, gamma
			)
		k2mean, k2cov = krig.predict(xi_r, posterior_cov='full')
		kr = k2mean[:n_r].reshape((N, N))
		mse = np.sqrt(np.abs(np.diag(k2cov[:n_r, :n_r]))).reshape((N, N))

	def build_surrogate_v3(self):
		# WORKING
	    self.gek = None

	def plot_contours(self):
		fig = plt.figure(figsize=(9,9))		
		axes = fig.subplots(nrows=3, ncols=3, sharex=True, sharey=True, squeeze=False)
		levels = np.linspace(np.min(self.x), np.max(self.x), 21)
		# Top row - values of f
		ax = axes[0][0]
		xf = self.x[self.n_o:].reshape((self.N,self.N))
		ax.contour(self.xi_r1, self.xi_r2, xf, levels=levels)
		ax.plot(self.f.xopt[0], self.f.xopt[1], 'o', markerfacecolor='none', markeredgecolor='r')
		ax.set_title(f'Reference, min={np.min(xf):.2f}')
		
		ax = axes[0][1]
		krigf = self.krig['muhat'][self.n_o:].reshape((self.N,self.N))
		ax.contour(self.xi_r1, self.xi_r2, krigf, levels=levels)
		i = np.argmin(krigf)
		ax.plot(self.xi_r[i,0], self.xi_r[i,1], 'xr')		
		ax.plot(self.f.xopt[0], self.f.xopt[1], 'o', markerfacecolor='none', markeredgecolor='r')
		ax.set_title(f'Kriging, min={np.min(krigf):.2f}')
		
		ax = axes[0][2]
		gekf, _ = gek_separate(self.gek['muhat'], 2)
		gekf = gekf[self.n_o:].reshape((self.N,self.N))
		i = np.argmin(gekf)
		ax.plot(self.xi_r[i,0], self.xi_r[i,1], 'xr')		
		ax.contour(self.xi_r1, self.xi_r2, gekf, levels=levels)
		ax.plot(self.f.xopt[0], self.f.xopt[1], 'o', markerfacecolor='none', markeredgecolor='r')
		ax.set_title(f'GEK, min={np.min(gekf):.2f}')
		
		# Middle row - true error
		errcmap = plt.get_cmap('YlOrRd')
		ax = axes[1][1]
		krigerr = np.abs(krigf-xf)
		levelserr = np.linspace(0, np.max(krigerr), 21)
		ax.contourf(self.xi_r1, self.xi_r2, krigerr, levels=levelserr, cmap=errcmap)
		ax.set_title(f'Kriging error, mse={np.sqrt(np.mean(krigerr**2)):.2f}')
		
		ax = axes[1][2]
		gekerr = np.abs(gekf-xf)
		ax.contourf(self.xi_r1, self.xi_r2, gekerr, levels=levelserr, cmap=errcmap)
		ax.set_title(f'GEK error, mse={np.sqrt(np.mean(gekerr**2)):.2f}')

		# Bottom row - prediction stddev
		ax = axes[2][1]
		krigsd = covariance_to_stddev(self.krig['Sigmahat'])[self.n_o:].reshape((self.N,self.N))
		levelserr = np.linspace(0, np.max(krigerr), 21)
		ax.contourf(self.xi_r1, self.xi_r2, krigsd, levels=levelserr, cmap=errcmap)
		ax.set_title(f'Kriging stddev, mse={np.sqrt(np.mean(krigsd**2)):.2f}')
		
		ax = axes[2][2]
		geksd, _ = gek_separate(covariance_to_stddev(self.gek['Sigmahat']), 2) 
		geksd = geksd[self.n_o:].reshape((self.N,self.N))
		ax.contourf(self.xi_r1, self.xi_r2, geksd, levels=levelserr, cmap=errcmap)
		ax.set_title(f'GEK stddev, mse={np.sqrt(np.mean(geksd**2)):.2f}')

		# Make pretty
		axes[1][0].axis('off')
		axes[2][0].axis('off')
		for ax in (axes[0][0], axes[0][1], axes[0][2], axes[1][1], axes[1][2], axes[2][1], axes[2][2]):
			ax.plot(self.xi_o[:, 0], self.xi_o[:, 1], 'ok')
			ax.set_xlabel(r'$\xi_1$')
			ax.set_ylabel(r'$\xi_2$')
		fig.suptitle('')
		
		iowrite('example2d.pdf')
		fig.savefig('example2d.pdf')


