#!/usr/bin/env python
"""
Example surrogate modelling in 1d (`examples/example1d`)
========================================================

Class to do surrogate modelling and plotting of a function `f` of one variable, with 
derivative, another function `df`.
"""
import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches

from vkikriging.mylib import (
    covariance_to_stddev,
    enter_debugger_on_uncaught_exception,
    iowrite,
    gek_separate,
    get_aspect,
)
from vkikriging import kriging_v1, kriging_v2, kriging_v3


def plot1d_stddevregion(ax, xi, mu, mse, label=None, color='r'):
    """
	Plot mean, and nice shaded regions representing +- 1,2,3-sigma for 1d plots.

	Args:
	  ax (Axis): Matplotlib axis object in which to plot.
	  xi (ndarray): Reconstruction locations, shape `n_r`.
	  mu (ndarray): Surrogate mean, shape `n_r`.
	  mse (ndarray): Surrogate stddev, shape `n_r`.
	  facecolor (Color): Matplotlib color of the mean-line and stddev regions, 
						 e.g. 'r' for red, etc.
	"""
    for i in [1, 2, 3]:
        verts = list(
            zip(
                np.hstack((xi, xi[::-1])),
                np.hstack((mu + i * mse, (mu - i * mse)[::-1])),
            )
        )
        poly = matplotlib.patches.Polygon(
            verts, facecolor=color, edgecolor='w', linestyle='dashed', alpha=0.1
        )
        ax.add_patch(poly)
    ax.plot(xi, mu, '-', color=color, label=label)


class Example1d:
    """
	Example demonstrating how to use kriging and gek, versions v1,v2 and v3, in 1d.
	For a function `f` (e.g. analytically defined), with gradient `df`.

	Args:
	  f (function): Function to approximate.
	  df (function): Gradient of `f`.
	  xi_samples (ndarray): Locations xi at which to sample `f` and `df`.
	  gamma (float): Correlation length in prior.
	  sigma_d, sigma_dg (float): Standard errors in observations and gradients.
	  xi_min, xi_max (float): Extreme values in independent variable, for reconstruction
							  and plotting.

	Example usage::

	  # 1. Initialize the function to approxiate, samples and Kriging paramters
	  ex1d = Example1d(np.sin, np.cos, np.linspace(0, 1, 8), gamma, sigma_d, sigma_dg,
					   xi_min=-.5, xi_max=1.5)
	  # 2. Build the Kriging model, v1, v2 or v3.  For v1, v2 GEK model is also built.
	  ex1d.build_surrogate_v1()
	  # 3. Plot the model, use any of all of the plot_* functions
	  fig = plt.figure(figsize=(10, 6))
	  ax = fig.add_subplot(111)
	  ex1d.plot_kriging(ax)
	  ex1d.plot_posterior_samples_kriging(ax)
	  ex1d.plot_reference(ax)
	"""

    def __init__(self, f, df, xi_samples, gamma, sigma_d, sigma_dg, xi_min, xi_max):
        ### Define observation locations - points at which data is available.
        # In d-dimensions coordinates should be a matrix of dimension (n x d), in 1-d
        # therefore (n x 1).
        self.d = 1  # Dimension 1d
        self.n_o = len(xi_samples)  # Number of observations
        self.xi_o = np.array([xi_samples]).T

        ### Define reconstruction locations - where we want to approximate the surrogate.
        self.n_r = 501
        self.xi_r = np.array(
            [np.linspace(xi_min, xi_max, self.n_r)]
        ).T  # Uniformly spaced (N x 1)

        ### Put everything together - repeated coordinates are okay.
        self.n = self.n_o + self.n_r
        self.xi = np.vstack((self.xi_o, self.xi_r))  # All coordates (n x 1)

        ### State which of the coordinates will be observed (in this case the first n_o).
        self.observed = np.zeros(self.n, dtype=bool)
        self.observed[: self.n_o] = True

        ### Observe the function.
        # Here we observe at all coordinates - in order to plot the reference solution
        # later.  In reality we would only evaluate the function at the first n_o
        # coordinates representing the observations, and could leave the remaining
        # entries of x equal to zero.  Unobserved values of x and dx are not used by
        # kriging() or gek().
        self.x = np.zeros(self.n)  # Function value
        self.dx = np.zeros((self.n, self.d))  # Function derivative, matrix (n x 1)
        for i, xi_i in enumerate(self.xi):
            self.x[i] = f(xi_i[0])
            self.dx[i, 0] = df(xi_i[0])
        self.x_o = self.x[: self.n_o]  # Observed values only (of truth)
        self.x_r = self.x[self.n_o :]  # Truth at reconstruction locations
        self.dx_o = self.dx[: self.n_o]  # Observed exact gradients
        self.dx_r = self.dx[self.n_o :]  # Exact gradient at reconstruction locations

        ### Specify parameters of the prior
        # Correlation parameter.  The Gaussian covariance function will have a 1-sigma
        # width of gamma.
        self.gamma = gamma
        # Obvervation error.
        self.sigma_d, self.sigma_dg = sigma_d, sigma_dg
        # Sample mean, stddev of observations - used for prior.
        self.mu, self.std = (np.mean(self.x_o), np.std(self.x_o))

    def build_surrogate_v1(self):
        print('Building simple Kriging model (version 1):')
        self.krig = kriging_v1.kriging(
            self.xi, self.x, self.observed, self.sigma_d, self.mu, self.std, self.gamma
        )
        print('Building simple GEK model (version 1):')
        self.gek = kriging_v1.gek(
            self.xi,
            self.x,
            self.dx,
            self.observed,
            self.sigma_d,
            self.sigma_dg,
            self.mu,
            self.std,
            self.gamma,
        )

    def build_surrogate_v2(self):
        print('Building simple Kriging model (version 2):')
        kinit = kriging_v2.Kriging(
            self.xi_o,  # observations only
            self.x_o,
            self.sigma_d,
            self.mu,
            self.std,
            self.gamma,
        )
        muhat, Sigmahat = kinit.predict(self.xi, posterior_cov='full')
        print('Building simple GEK model (version 2):')
        self.krig = {'muhat': muhat, 'Sigmahat': Sigmahat}
        gekinit = kriging_v2.GEK(
            self.xi_o,  # observations only
            self.x_o,
            self.dx_o,
            self.sigma_d,
            self.sigma_dg,
            self.mu,
            self.std,
            self.gamma,
        )
        muhat, Sigmahat = gekinit.predict(self.xi, posterior_cov='full', partial=False)
        self.gek = {'muhat': muhat, 'Sigmahat': Sigmahat}

    def build_surrogate_v3(self):
        print('Building universal Kriging model (version 3):')
        regression_basis = kriging_v3.F_linear
        self.krig = kriging_v3.kriging(
            self.xi,
            self.x,
            self.observed,
            self.sigma_d,
            regression_basis,
            self.std,
            self.gamma,
        )
        self.gek = None

    def plot_reference(self, ax):
        """Plot ground-truth reference and observations in matplotlib axis `ax`."""
        ax.plot(self.xi_r, self.x_r, '-k', label='Ground-truth')
        ax.plot(self.xi_o, self.x_o, 'ok', label='Observations')

    def plot_observed_gradients(self, ax, length=0.1):
        """
		Plot observations of gradients in matplotlib axis `ax` as little bars of length
		`length`.  Attempt to keep lengths of bars constant, which depends on plot aspect
		ratio - therefore use this plotting function last.
		"""
        deltaxi = length / 2 * np.cos(np.arctan(self.dx_o * get_aspect(ax)))[:, 0]
        aspect = get_aspect(ax)
        for i in range(self.n_o):
            ax.plot(
                (self.xi_o[i, 0] - deltaxi[i], self.xi_o[i, 0] + deltaxi[i]),
                (
                    self.x_o[i] - deltaxi[i] * self.dx_o[i, 0],
                    self.x_o[i] + deltaxi[i] * self.dx_o[i, 0],
                ),
                '-k',
                linewidth=3,
                label='Observed gradients' if i == 0 else None,
            )

    def plot_kriging(self, ax):
        """Plot Kriging response with 3 sigmas in matplotlib axis `ax`."""
        n_o = self.n_o
        mse = covariance_to_stddev(self.krig['Sigmahat'])[n_o:]
        plot1d_stddevregion(
            ax,
            self.xi_r[:, 0],
            self.krig['muhat'][n_o:],
            mse,
            label='Kriging',
            color='r',
        )

    def plot_gek(self, ax):
        """Plot GEK response with 3 sigmas in matplotlib axis `ax`."""
        n_o = self.n_o
        mu, dmu = gek_separate(self.gek['muhat'], self.d)
        mse, dmse = gek_separate(covariance_to_stddev(self.gek['Sigmahat']), self.d)
        plot1d_stddevregion(
            ax, self.xi_r[:, 0], mu[n_o:], mse[n_o:], label='GEK', color='b'
        )

    def plot_posterior_samples_kriging(self, ax, nsamples=10):
        n_o = self.n_o
        mvn = stats.multivariate_normal(
            self.krig['muhat'][n_o:],
            self.krig['Sigmahat'][n_o:, n_o:],
            allow_singular=True,
        )
        for i in range(nsamples):
            ax.plot(self.xi_r, mvn.rvs(), '-r', linewidth=.5)

    def plot_posterior_samples_gek(self, ax, nsamples=10):
        n_o = self.n_o
        mvn = stats.multivariate_normal(
            self.gek['muhat'], self.gek['Sigmahat'], allow_singular=True
        )
        for i in range(nsamples):
            xsample, dxsample = gek_separate(mvn.rvs(), self.d)
            ax.plot(self.xi_r, xsample[n_o:], '-b', linewidth=.5)


if __name__ == '__main__':
    enter_debugger_on_uncaught_exception()

    def sintest(xi):
        return np.sin(3.2 * np.pi * xi) + 2.0 * xi

    def sintest_gradient(xi):
        return 3.2 * np.pi * np.cos(3.2 * np.pi * xi) + 2.0

    gamma, sigma_d, sigma_dg = 0.08, 0.01, 0.001
    ex1d = Example1d(
        sintest,
        sintest_gradient,
        np.linspace(0, 1, 8),
        gamma,
        sigma_d,
        sigma_dg,
        xi_min=-.5,
        xi_max=1.5,
    )
    ex1d.build_surrogate_v2()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title(
        f'Correlation length $\gamma={gamma}$, data-noise $\sigma_d={sigma_d}$'
    )
    ex1d.plot_kriging(ax)
    ex1d.plot_gek(ax)
    ex1d.plot_posterior_samples_kriging(ax)
    ex1d.plot_posterior_samples_gek(ax)
    ex1d.plot_reference(ax)
    ex1d.plot_observed_gradients(ax)
    ax.legend(loc='best')
    iowrite('example1d.pdf')
    fig.savefig('example1d.pdf')
