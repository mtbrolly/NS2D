"""
A module for computing spatial statistics of a `Model` instance.
"""

import cupy as cp
if __name__ == "__main__":
    from model import Model


def spectral_variance(model, fk):
    """Calculate spectral variance from real fft'ed field, fk.

    Note since we use real fft we have to nominally double the contribution
    from most modes with the exception of those in the first and last column of
    fk.
    """

    var_dens = 2. * cp.abs(fk) ** 2
    var_dens[..., 0] /= 2
    var_dens[..., -1] /= 2
    return var_dens.sum(axis=(-1, -2)) / model.n_x ** 4


def energy(model):
    """Calculate mean energy per unit area using psik.
    """
    return 0.5 * spectral_variance(model, model.wv * model.psik)


def energy_via_vorticity(model):
    """Calculate mean energy per unit area using zk.
    """
    return 0.5 * spectral_variance(model, model.zk * model.wv2i ** 0.5)


def enstrophy(model):
    """Calculate mean enstrophy per unit area using zk.
    """
    return 0.5 * spectral_variance(model, model.zk)


def cfl(model):
    """Calculate Courant-Friedrichs-Lewy number.
    """
    return cp.abs(
        cp.hstack([model.u, model.v])).max() * model.timestepper.dt / model.dx


def energy_spectrum(model):
    """Calculate 2D energy spectrum from psik.
    """
    return model.wv2 * cp.abs(model.psik) ** 2 / model.n_x ** 4


def enstrophy_spectrum(model):
    """Calculate 2D enstrophy spectrum from zk.
    """
    return cp.abs(model.zk) ** 2 / model.n_x ** 4


def isotropic_spectrum(model, spectrum):
    """Calculate an isotropic spectrum from a 2D spectrum.
    """
    kmax = model.kx.max()
    dkr = cp.sqrt(2.)
    kr = cp.arange(dkr.get() / 2., kmax.get() + dkr.get(), dkr.get())
    iso_spec = cp.zeros(kr.size)

    for i in range(kr.size):
        fkr = (model.wv >= kr[i] - dkr / 2) & (model.wv <= kr[i] + dkr / 2)
        dtk = cp.pi / (fkr.sum() - 1)
        iso_spec[i] = spectrum[fkr].sum() * kr[i] * dtk

    return kr, iso_spec


def isotropic_energy_spectrum(model):
    """Calculate isotropic energy spectrum.
    """
    return isotropic_spectrum(model, energy_spectrum(model))


def isotropic_enstrophy_spectrum(model):
    """Calculate isotropic enstrophy spectrum.
    """
    return isotropic_spectrum(model, enstrophy_spectrum(model))


def energy_centroid(model):
    """Calculate energy centroid.
    """
    kr, iso_spec = isotropic_energy_spectrum(model)
    return cp.sum(kr * iso_spec) / energy(model)


def enstrophy_centroid(model):
    """Calculate enstrophy centroid.
    """
    kr, iso_spec = isotropic_enstrophy_spectrum(model)
    return cp.sum(kr * iso_spec) / enstrophy(model)


def eddy_turnover_time(model):
    """Calculate eddy turnover time.
    """
    return 2 * cp.pi / cp.sqrt(enstrophy(model))


def velocity_kurtosis(model):
    """Calculate the kurtosis of the velocity field (u component only).
    """
    return cp.mean(model.u ** 4) / cp.var(model.u) ** 2


def vorticity_kurtosis(model):
    """Calculate the vorticity of the vorticity field.
    """
    return cp.mean(model.z ** 4) / cp.var(model.z) ** 2


def reynolds_number(model):
    """Calculate Reynold's number.
    """
    return cp.sqrt(cp.mean(model.u ** 2 + model.v ** 2)) / (
        energy_centroid(model) * model.mechanisms['viscosity'].coefficient)


def energy_dissipation_due_to_viscosity(model):
    """Calculate rate of energy dissipation per unit time due to viscosity.
    """
    return 2 * model.mechanisms['viscosity'].coefficient * enstrophy(model)


def time_series(data_dir, n_x, twrite, n_snapshots):
    """Calculate and save time series of some specified statistics."""
    m = Model(n_x)
    E = []
    Z = []
    for i in range(1, n_snapshots + 1):
        m.zk = cp.load(data_dir + f"zk_{i * twrite}.npy")
        m._update_fields()
        E.append(energy(m))
        Z.append(enstrophy(m))
    cp.save(data_dir + "E.npy", E)
    cp.save(data_dir + "Z.npy", Z)
