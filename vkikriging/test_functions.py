"""
Test functions for (global) optimization problems
=================================================

Following test-functions implemented: Parabola, NoisyParabola, Rosenbrock, Fourmin,
Schwefel, Camel.  Each has `f`, `df`, `ddf` function members for value, derivative and
Hessian; `xmin`, `xmax` describing domain (in each dimension); `xopt` location of
optimum, `fopt` value of optimum.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
try:
	import algopy
except ImportError:
	print('No module "algopy".	Some derivative info is not available.')
	algopy = None


class Parabola:
	"""
	Parabola, in d-dimensions.	Minimum of 0 at 0 (given +ve def A).

	  f(x) = 0.5 * x^T A x
	"""

	def __init__(self, d=2):
		self.d = d
		self.A = np.identity(d)
		self.xmin, self.xmax = -np.ones(d), np.ones(d)
		self.xopt = np.zeros(d)
		self.fopt = 0

	def f(self, xx):
		return 0.5 * np.dot(xx, np.dot(self.A, xx))

	def df(self, xx):
		return np.dot(self.A, xx)

	def ddf(self, xx):
		return self.A


class NoisyParabola:
	"""Parabola with noise, d-dimensions."""

	def __init__(self, d=2):
		self.d = d
		self.xmin, self.xmax = -np.ones(d), np.ones(d)
		self.xopt = np.zeros(d)
		self.fopt = 0

	def f(self, xx):
		return np.sum(xx ** 2 - 0.2 * np.cos(31. * xx)) + self.d * 0.2

	def df(self, xx):
		return 2 * xx + 0.2 * 31. * np.sin(31. * xx)

	def ddf(self, xx):
		return 2. * np.identity(self.d) + np.diag(0.2 * 31. ** 2 * np.cos(31. * xx))


class Rosenbrock:
	"""Rosenbrock's problem 2d only."""

	def __init__(self):
		self.d = 2
		self.xmin, self.xmax = [-2, -1], [2, 3]
		self.xopt = [1., 1.]
		self.fopt = 0.

	def f(self, xx):
		x, y = xx[0], xx[1]
		return 100.0 * (y - x * x) * (y - x * x) + (1. - x) * (1. - x)

	def df(self, xx):
		x, y = xx[0], xx[1]
		return np.array([-400. * (y - x * x) * x - 2. * (1. - x), 200. * (y - x * x)])

	def ddf(self, xx):
		x, y = xx[0], xx[1]
		return np.array([[-400. * (y - 3. * x ** 2) + 2, -400 * x], [-400 * x, 200]])


class Fourmin:
	"""Don't know the real name of this function, d-dimensions."""

	def __init__(self, d=2):
		self.d = d
		self.xmin, self.xmax = -6 * np.ones(d), 6 * np.ones(d)
		self.xopt = -4.4537713 * np.ones(d)
		self.fopt = -d * 1.30819

	def f(self, xx):
		return 0.01 * np.sum((xx + 0.5) ** 4 - 30. * xx ** 2 - 20. * xx)

	def df(self, xx):
		x = algopy.UTPM.init_jacobian(xx)
		y = self.f(x)
		return algopy.UTPM.extract_jacobian(y)

	def ddf(self, xx):
		x = algopy.UTPM.init_tensor(2, xx)
		y = self.f(x)
		return algopy.UTPM.extract_tensor(2, y)


class Schwefel:
	"""Schwefel test function, d-dimensions."""

	def __init__(self, d=2):
		self.d = d
		self.xmin, self.xmax = -500 * np.ones(d), 500 * np.ones(d)
		self.xopt = 420.9687 * np.ones(d)
		self.fopt = -418.9829 * d

	def f(self, xx):
		a = np.sin(np.sqrt(np.abs(xx)))
		b = a * 0.
		for i in range(self.d):	# Apparently algopy doesn't do vector * vector...
			b[i] = -xx[i] * a[i]
		s = np.sum(b)
		return 418.9829 * self.d + s

	def df(self, xx):
		"""WARNING: derivative at 0 doesn't exist, algopy will return nan there."""
		x = algopy.UTPM.init_jacobian(xx)
		y = self.f(x)
		return algopy.UTPM.extract_jacobian(y)

	def ddf(self, xx):
		x = algopy.UTPM.init_tensor(2, xx)
		y = self.f(x)
		return algopy.UTPM.extract_tensor(2, y)


class Camel:
	"""Six-hump camel-back test function in 2-d."""

	def __init__(self):
		self.d = 2
		self.xmin, self.xmax = [-3, -2], [3, 2]
		self.xopt = [-0.0898, 0.7126]  # and (0.0898, -0.7126)
		self.fopt = -1.0316

	def f(self, xx):
		x, y = xx[0], xx[1]
		return (
			(4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2
			+ x * y
			+ (-4 + 4 * y ** 2) * y ** 2
		)

	def df(self, xx):
		x = algopy.UTPM.init_jacobian(xx)
		y = self.f(x)
		return algopy.UTPM.extract_jacobian(y)

	def ddf(self, xx):
		x = algopy.UTPM.init_tensor(2, xx)
		y = self.f(x)
		return algopy.UTPM.extract_tensor(2, y)


def plot_testfunction(testfn):
	"""Contour plot of 2d test-function `f`."""
	res = 51
	x = np.linspace(testfn.xmin[0], testfn.xmax[0], res)
	y = np.linspace(testfn.xmin[1], testfn.xmax[1], res)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros((res, res))
	for i, xi in enumerate(x):
		for j, yj in enumerate(y):
			Z[i, j] = testfn.f(np.array([xi, yj]))

	fig = plt.figure(figsize=(12, 12))
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev=None, azim=20 + 180)
	ax.plot_surface(
		X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True
	)
	zmin, zmax = np.min(Z), np.max(Z)
	cset = ax.contour(X, Y, Z, zdir='z', offset=zmin)
	ax.plot([testfn.xopt[0]], [testfn.xopt[1]], 'ok')
	ax.set_zlim(zmin, zmax)
	fig.savefig('test_function.pdf')
	plt.show()

	
def plot_derivativetest(testfn):
	"""
	DEBUGGING CODE: Verify derivatives and Hessians implemented above
	by comparing the full function with it's 3-term Taylor expansion.
	These are plotted along a line, and should agree with each other
	close to x0, the point at which the expansion is made.
	"""
	res = 501
	x0 = np.array([200,2])
	xx = np.linspace(testfn.xmin[0], testfn.xmax[1], res)
	ff, aa = np.zeros(res), np.zeros(res)  # Taylor series

	f = testfn.f(x0)  # Change these to test a different fn.
	df = testfn.df(x0)
	ddf = testfn.ddf(x0)

	for i, x in enumerate(xx):
		xp = np.array([x, x0[1]])
		ff[i] = testfn.f(xp)
		aa[i] = f + np.dot(df, xp - x0) + 0.5 * np.dot(xp - x0, np.dot(ddf, xp - x0))

	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)
	ax.plot(xx, ff, '-k', label='f')
	ax.plot(xx, aa, '-r', label='Taylor expansion')
	ax.plot(x0[0], f, 'ok')
	ax.legend(loc='best')
	plt.show()


if __name__ == '__main__':

	plot_testfunction(Schwefel(2))

	plot_derivativetest(Schwefel(2))
