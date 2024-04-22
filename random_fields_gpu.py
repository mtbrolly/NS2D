"""
Module for random field generation.
"""

import cupy as cp
from spatial_statistics_gpu import spectral_variance
import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)
fft_lib = scipy.fft


def JMcW(model, seed=1):
    """
    Initial condition from J. McWilliams' 1984 JFM paper.
    """
    fk = model.wv != 0
    ckappa = cp.zeros_like(model.wv2)
    ckappa[fk] = cp.sqrt(
        model.wv2[fk] * (1. + (model.wv2[fk] / 36.) ** 2)) ** -1
    nhx, nhy = model.wv2.shape
    model.rng_init = cp.random.default_rng(seed=seed)
    Pi_hat = cp.reshape(model.rng_init.standard_normal(size=nhx * nhy)
                        + 1j * model.rng_init.standard_normal(size=nhx * nhy),
                        model.wv.shape) * ckappa
    Pi = fft_lib.irfft2(Pi_hat[:, :])
    Pi = Pi - Pi.mean()
    Pi_hat = fft_lib.rfft2(Pi)
    KEaux = 0.5 * spectral_variance(model, model.wv * Pi_hat)
    pik = (Pi_hat / cp.sqrt(KEaux))
    zik = -model.wv2 * pik
    return zik


def Gaussian_init(model, mean, std):
    z = cp.exp(-(((model.x - mean[0]) / std[0]) ** 2
                 + ((model.y - mean[1]) / std[1]) ** 2))
    zk = fft_lib.rfft2(z)
    zk /= cp.sqrt(0.5 * spectral_variance(model, zk * model.wv2i ** 0.5))
    return zk
