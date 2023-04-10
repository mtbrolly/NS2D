"""
A module for computing instantaneous spatial statistics of the model.
"""

import numpy as np
from numpy import pi


def spec_var(model, fk):
    """
    Compute spectral variance from real fft'ed field, fk.

    Note since we use real fft we have to nominally double the contribution
    from most modes with the exception of those in the first and last column of
    fk.
    """

    var_dens = 2. * np.abs(fk) ** 2 / model.nxny ** 2
    var_dens[..., 0] /= 2
    var_dens[..., -1] /= 2
    return var_dens.sum(axis=(-1, -2))


def calc_ke(model):
    """
    Calculate mean energy per unit area using psik.
    """
    return 0.5 * spec_var(model, model.wv * model.psik)


def calc_Ens(model):
    """
    Calculate mean enstrophy per unit area using psik.
    """
    return 0.5 * spec_var(model, model.wv2 * model.psik)


def calc_cfl(model):
    """
    Calculate Courant number.
    """
    return np.abs(np.hstack([model.u, model.v])).max() * model.dt / model.dx


def calc_kurt_u(model):
    """
    Calculate the spatial kurtosis of u.
    """
    return (np.mean(model.u ** 4) / (np.mean(model.u ** 2) ** 2))


def calc_kurt_z(model):
    """
    Calculate the spatial kurtosis of z.
    """
    return (np.mean(model.z ** 4) / (np.mean(model.z ** 2) ** 2))


def calc_kurt_psi(model):
    """
    Calculate the spatial kurtosis of psi.
    """
    return (np.mean(model.psi ** 4) / (np.mean(model.psi ** 2) ** 2))


def Ensspec(model):
    """
    Compute 2D enstrophy spectrum from zk.
    """
    return np.abs(model.zk) ** 2 / model.nxny ** 2


def KEspec(model):
    """
    Compute 2D energy spectrum from psik.
    """
    return model.wv2 * np.abs(model.psik) ** 2 / model.nxny ** 2


def KEspec_z(model):
    """
    Compute 2D energy spectrum from psik.
    """
    return model.wv2i * np.abs(model.zk) ** 2 / model.nxny ** 2


def calc_ispec(model, spec2D):
    """
    Compute an isotropic spectrum from a 2D spectrum.
    """
    if model.kk.max() > model.ll.max():
        kmax = model.ll.max()
    else:
        kmax = model.kk.max()
    dkr = np.sqrt(model.dk ** 2 + model.dl ** 2)
    kr = np.arange(dkr / 2., kmax + dkr, dkr)
    spec_iso = np.zeros(kr.size)

    for i in range(kr.size):
        fkr = (model.wv >= kr[i] - dkr / 2) & (model.wv <= kr[i] + dkr / 2)
        dtk = pi / (fkr.sum() - 1)
        spec_iso[i] = spec2D[fkr].sum() * kr[i] * dtk

    return kr, spec_iso


def calc_centroid_k(kr, spec_iso, quantity):
    """
    Calculate a (e.g. energy or enstrophy) centroid wavenumber given the
    relevant isotropic spectrum and e.g. mean energy/enstrophy.
    """
    return np.sum(kr * spec_iso) / quantity
