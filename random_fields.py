"""
Module for random field generation.
"""

import cupy as cp
from spatial_statistics import spectral_variance
import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)
fft_lib = scipy.fft


def JMcW(model, seed=1):
    """
    Initial condition from J. McWilliams' 1984 JFM paper.
    """
    fk = model.wv != 0
    ckappa = cp.zeros_like(model.wv2, dtype=model.real_dtype)
    ckappa[fk] = cp.sqrt(
        model.wv2[fk] * (1. + (model.wv2[fk] / 36.) ** 2)) ** -1
    nhx, nhy = model.wv2.shape
    rng_init = cp.random.default_rng(seed=seed)
    Pi_hat = cp.reshape(rng_init.standard_normal(
        size=nhx * nhy, dtype=model.real_dtype)
        + 1j * rng_init.standard_normal(
        size=nhx * nhy, dtype=model.real_dtype),
        model.wv.shape) * ckappa
    Pi = fft_lib.irfft2(Pi_hat[:, :])
    Pi = Pi - Pi.mean()
    Pi_hat = fft_lib.rfft2(Pi)
    KEaux = 0.5 * spectral_variance(model, model.wv * Pi_hat)
    pik = (Pi_hat / cp.sqrt(KEaux))
    zik = -model.wv2 * pik
    return zik.astype(model.complex_dtype)
