"""
Module for random field generation.
"""

import numpy as np
import pyfftw.interfaces.numpy_fft as fftw
from spatial_statistics import spectral_variance


def JMcW(model, seed=1):
    """
    Initial condition from J. McWilliams' 1984 JFM paper.
    """
    fk = model.wv != 0
    ckappa = np.zeros_like(model.wv2)
    ckappa[fk] = np.sqrt(
        model.wv2[fk] * (1. + (model.wv2[fk] / 36.) ** 2)) ** -1
    nhx, nhy = model.wv2.shape
    model.rng_init = np.random.default_rng(seed=seed)
    Pi_hat = np.reshape(model.rng_init.normal(size=nhx * nhy)
                        + 1j * model.rng_init.normal(size=nhx * nhy),
                        model.wv.shape) * ckappa
    Pi = fftw.irfft2(Pi_hat[:, :])
    Pi = Pi - Pi.mean()
    Pi_hat = fftw.rfft2(Pi)
    KEaux = 0.5 * spectral_variance(model, model.wv * Pi_hat)
    pik = (Pi_hat / np.sqrt(KEaux))
    zik = -model.wv2 * pik
    return zik
