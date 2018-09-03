"""
Module of basic, generic functions
==================================

Specifically for Kriging/GEK.
"""
import numpy as np
import copy, sys, time
from operator import sub


def forward_sub_single(A, b):
	"""Solve Ax = b for A lower triangular for single RHS b."""
	if A.shape[0] != b.shape[0]:
		raise ValueError("A and b must be compatible.")
	x = np.zeros(A.shape[1])
	for j, row in enumerate(A):
		pivot = row[j]
		if np.abs(pivot) < 1.e-12:
			raise ValueError("Matrix has a zero diagonal element.")
		x[j] = (b[j] - np.dot(row, x)) / pivot
	return x


def backward_sub_single(A, b):
	"""Solve Ax = b for A upper triangular for single RHS b."""
	if A.shape[0] != b.shape[0]:
		raise ValueError("A and b must be compatible.")
	x = np.zeros(A.shape[1])
	rev = list(range(A.shape[0]))
	rev.reverse()
	for j in rev:
		row = A[j]
		pivot = row[j]
		if np.abs(pivot) < 1.e-12:
			raise ValueError("Matrix has a zero diagonal element.")
		x[j] = (b[j] - np.dot(row, x)) / pivot
	return x


def tri_sub_generic(A, b, sub_single_fn):
	"""Helper fn for forward_sub() and backward_sub(), enabling multiple
	RHSs in both cases."""
	if b.ndim == 1:
		return sub_single_fn(A, b)
	else:
		outp = np.zeros(b.shape)
		for i in range(b.shape[1]):
			outp[:, i] = sub_single_fn(A, b[:, i])
		return outp


def forward_sub(A, b):
	"""Solve Ax = b for A lower triangular, possible multiple RHSs."""
	return tri_sub_generic(A, b, forward_sub_single)


def backward_sub(A, b):
	"""Solve Ax = b for A upper triangular, possible multiple RHSs."""
	return tri_sub_generic(A, b, backward_sub_single)


def gek_composite(x, dx):
	"""
	For GEK create composite vector of values and gradients - return
	1d vector xc.  This function defines the order of entries in composite
	vectors, and we must be consistent.	 The reverse of this is gek_separate().
	"""
	assert x.shape[0] == dx.shape[0], 'Dimension of derivative info wrong.'
	n, d = dx.shape
	xc = copy.copy(x)  # Extended sample values
	xc = np.hstack((xc, dx.reshape(n * d)))
	return xc


def gek_separate(xc, d):
	"""
	Map composite vector returned by GEK into individual value and
	derivate vectors.  This is useful for postprocessing output of
	gek() for plotting etc.	 Return x (n), dx (n x d).
	"""
	assert xc.ndim == 1 and xc.size % (d + 1) == 0, 'Dimension of input wrong.'
	n = xc.size // (d + 1)
	x = xc[:n]	# Values
	dx = xc[n:].reshape((n, d))	 # Gradients
	return x, dx


def covariance_to_stddev(Sigma):
	"""
	Obtain sample-wise standard deviation for given covariance matrix
	(extract the diagonal and take sqrt).  Useful when plotting error bars.
	  Sigma - Square, +ve-def, covariance matrix, e.g. P or Sigmahat.
	Return array (n).
	"""
	### np.abs() is there just for robustness - rounding error can result in
	### diagonal values in Sigma of e.g -1e-14.
	return np.sqrt(np.abs(np.diag(Sigma)))


def sample_process(mu, Sigma, nsamples=1):
	"""
	Generate random samples from a multivariate normal with a given
	mean mu and covariance matrix Sigma.  E.g. A sample from the
	posterior process: S = sample_process(muhat, Sigmahat)
	Return array (nsamples x n).
	"""
	return np.random.multivariate_normal(mu, Sigma, nsamples)


class Timing:
	"""
	Very rough timing reporting with minimal code intrusiuon - call
	__init__() once at start, and monitor() to report each time
	increment.	Turn stdout reporting off with verbose=False, times
	are still measured and stored in self.t1 (list).
	"""
	def __init__(self, verbose=True):
		self.t1 = [time.time()]
		self.verbose = verbose

	def reinit(self):
		self.t1.append(time.time())

	def monitor(self, message):
		t2 = time.time()
		if self.verbose:
			print('%40s: %10.2f s' % (message, t2 - self.t1[-1]))
		self.t1.append(t2)
		sys.stdout.flush()

		
def enter_debugger_on_uncaught_exception():
	"""Automatically entering the debugger on any uncaught exception."""
	def info(type1, value, tb):
		if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type1 == SyntaxError:
			# we are in interactive mode or we don't have a tty-like device, or error is
			# a SyntaxError (can not be debugged), so we call the default hook
			sys.__excepthook__(type1, value, tb)
		else:
			import traceback, ipdb
			# we are NOT in interactive mode, print the exception...
			traceback.print_exception(type1, value, tb)
			print()
			# ...then start the debugger in post-mortem mode.
			ipdb.pm()
	sys.excepthook = info

	
def iowrite(filename):
	"""Print file-writing information to stdout (in blue!)."""
	FILEIO = '\033[94m'
	ENDC = '\033[0m'
	sys.stdout.write(f'{FILEIO}Writing <{filename}>{ENDC}\n')


def get_aspect(ax):
	"""Compute the aspect-ratio of matplotlib Axis `ax`."""
	figW, figH = ax.get_figure().get_size_inches()	# Total figure size
	_, _, w, h = ax.get_position().bounds  # Axis size on figure 
	disp_ratio = (figH * h) / (figW * w)  # Ratio of display units
	data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())	# Ratio of data units
	return disp_ratio / data_ratio

