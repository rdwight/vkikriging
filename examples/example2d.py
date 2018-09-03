import numpy as np
import matplotlib.pyplot as plt

from vkikriging.mylib import (
	covariance_to_stddev,
	enter_debugger_on_uncaught_exception,
	iowrite,
	gek_separate,
	get_aspect,
)
from vkikriging import test_functions, sampling
from vkikriging import kriging_v1, kriging_v2, kriging_v3

class Example2d:
	"""
	Example demonstrating how to use kriging.kriging() and
	kriging.gek() in 2d.  Generalization to d-dimensions is similar.
	"""
	def __init__(self, testfn, n_o, gamma, sigma_d, sigma_dg):
		# Define observation locations - use random-sampling.
		self.d = 2
		self.n_o = n_o
		self.xi_o = sampling.sobol(self.n_o, self.d)
		self.xi_o[:, 0] = self.xi_o[:, 0] * (testfn.xmax[0] - testfn.xmin[0]) + testfn.xmin[0]
		self.xi_o[:, 1] = self.xi_o[:, 1] * (testfn.xmax[1] - testfn.xmin[1]) + testfn.xmin[1]

		# Define reconstruction locations - use rectangular grid.
		N = 41
		self.n_r = N ** 2
		xi_r1 = np.linspace(testfn.xmin[0], testfn.xmax[0], N)
		xi_r2 = np.linspace(testfn.xmin[1], testfn.xmax[1], N)
		A, B = np.meshgrid(xi_r1, xi_r2)
		self.xi_r = np.vstack((A.flatten(), B.flatten())).T

		# Put everything together - repeated coordinates are okay.
		self.n = n_o + self.n_r
		self.xi = np.vstack((self.xi_o, self.xi_r))

		# State which of the coordinates will be observed (in this case the first n_o).
		self.observed = np.zeros(self.n, dtype=bool)
		self.observed[:n_o] = True

		# Observe the function (we also observe the reconstruction locations for reference)
		self.x = np.zeros(self.n)
		self.dx = np.zeros((self.n, self.d))
		for i, xi_i in enumerate(self.xi):
			self.x[i] = testfn.f(xi_i)
			self.dx[i, :] = testfn.df(xi_i)

		# Specify parameters of the prior
		self.gamma = gamma
		self.sigma_d, self.sigma_dg = sigma_d, sigma_dg	 # Obvervation error.
		# Sample mean + (conservative) stddev estimate.
		self.mu, self.std = np.mean(self.x), 3 * np.std(self.x)

	def build_surrogate_v1(self):
		print('Building simple Kriging model (version 1):')
		krig = kriging_v1.kriging(self.xi, self.x, self.observed, self.sigma_d, self.mu,
								  self.std, self.gamma)
		print('Building simple GEK model (version 1):')
		gek = kriging_v1.gek(
			self.xi, self.x, self.dx, self.observed, self.sigma_d, self.sigma_dg,
			self.mu, self.std, self.gamma
		)

	def plot_(self, ax):
		fig = plt.figure(figsize=(14,8))
		axes = fig.subplots(nrows=2, ncols=3, sharex=True, sharey=True, squeeze=True)
		axes[0][0].
		#kr = krig['muhat'][n_o : n_o + n_r].reshape((N, N))	 # Kriging mean
		#mse = kriging_v1.covariance_to_stddev(krig['Sigmahat'])[
		#	n_o : n_o + n_r
		#].reshape((N, N))

	def build_surrogate_v2(self):
		if not gek:
			krig = kriging_v2.Kriging(xi_o, x[:n_o], sigma_d, mu, std, gamma)
		else:
			krig = kriging_v2.GEK(
				xi_o, x[:n_o], dx[:n_o, :], sigma_d, sigma_dg, mu, std, gamma
			)
		k2mean, k2cov = krig.predict(xi_r, posterior_cov='full')
		kr = k2mean[:n_r].reshape((N, N))
		mse = np.sqrt(np.abs(np.diag(k2cov[:n_r, :n_r]))).reshape((N, N))

	def build_surrogate_v3(self):
		pass

	def blah(self):
		### Extract posterior variables correponding to predictions only (remove observations).
		### These will correspond to coordinates xi_r.
		ex = x[n_o:].reshape((N, N))  # Exact solution

		### Plotting
		pylab.figure(figsize=(12, 12))
		pylab.clf()
		# -----
		pylab.subplot(2, 2, 1)
		pylab.contour(xi_r1, xi_r2, ex, camel_lev)
		pylab.plot(xi_o[:, 0], xi_o[:, 1], 'ok')
		pylab.xlabel(r'$\xi_1$', size='x-large')
		pylab.ylabel(r'$\xi_2$', size='x-large')
		pylab.title('Original')
		# -----
		pylab.subplot(2, 2, 2)
		pylab.contour(xi_r1, xi_r2, kr, camel_lev)
		pylab.plot(xi_o[:, 0], xi_o[:, 1], 'ok')
		pylab.xlabel(r'$\xi_1$', size='x-large')
		pylab.ylabel(r'$\xi_2$', size='x-large')
		pylab.title('Surrogate')
		# -----
		pylab.subplot(2, 2, 3)
		pylab.contour(xi_r1, xi_r2, np.abs(ex - kr), camel_errlev)
		pylab.plot(xi_o[:, 0], xi_o[:, 1], 'ok')
		pylab.xlabel(r'$\xi_1$', size='x-large')
		pylab.ylabel(r'$\xi_2$', size='x-large')
		pylab.title('Surrogate error')
		# -----
		pylab.subplot(2, 2, 4)
		pylab.contour(xi_r1, xi_r2, mse, camel_errlev)
		pylab.plot(xi_o[:, 0], xi_o[:, 1], 'ok')
		pylab.xlabel(r'$\xi_1$', size='x-large')
		pylab.ylabel(r'$\xi_2$', size='x-large')
		pylab.title('Posterior covariance')
		# -----

		pylab.subplots_adjust(
			left=0.07, bottom=0.07, right=0.95, top=0.95, wspace=None, hspace=None
		)
		pylab.savefig('example2d.pdf')


# Run it.
if __name__ == '__main__':
	enter_debugger_on_uncaught_exception()
	N = 30
	testfn = test_functions.Camel()
	gamma, sigma_d, sigma_dg = 0.08, 0.01, 0.001
	ex2d = Example2d(
		testfn,
		N,
		gamma,
		sigma_d,
		sigma_dg,
	)
	ex2d.build_surrogate_v1()
