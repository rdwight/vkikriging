"""
(Pseudo-)random sampling schemes on the unit hypercube
======================================================

All implemented in d-dimensions.
"""
import numpy as np

from . import sampling_sobol


def random_uniform(nsamples, d):
    """Random coordinates on unit hypercube."""
    return np.random.random((nsamples, d))


def latin_hypercube(nsamples, d):
    """
    Latin Hypercube sampling using uniform distribution on the interval
    [0.0, 1.0].  Returns nsamples x d array of sample coordinates.
    """
    s = np.zeros((nsamples, d))
    for i in range(d):
        id = list(range(nsamples))
        np.random.shuffle(id)
        for j in range(nsamples):
            s[j][i] = (np.random.random() + id[j]) / nsamples
    return s


def _halton_1d(idx, base):
    """Generate the idx-th entry in the halton sequence with given (prime) base."""
    out, f = 0., 1. / base
    i = idx
    while i > 0:
        out = out + f * (i % base)
        i = np.floor(i / base)
        f = f / base
    return out


def halton(nsamples, d):
    """Halton sequence on [0,1]^d of length nsamples."""
    primes = np.array(
        [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            163,
            167,
            173,
            179,
            181,
            191,
            193,
            197,
            199,
        ]
    )
    assert d <= len(primes)
    s = np.zeros((nsamples, d))
    for i in range(d):
        p = primes[i]
        for j in range(nsamples):
            s[j, i] = _halton_1d(j, p)
    return s


def sobol(nsamples, d):
    """Sobol sequence on [0,1]^d, skipping first 1000 samples."""
    return sampling_sobol.sobol(nsamples, d, skip=1000)
