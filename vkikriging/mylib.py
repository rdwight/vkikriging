"""
Module of basic, generic functions (`mylib`)
============================================

Specifically for Kriging/GEK.
"""
import numpy as np
import copy, sys, time
from operator import sub


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

