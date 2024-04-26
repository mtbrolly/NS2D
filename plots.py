"""Plotting functions for standard plots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import spatial_statistics
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
from model import Model
scipy.fft.set_global_backend(cufft)
fft_lib = scipy.fft
plt.ioff()


def plot_vorticity_field(model, halfrange=None, filename='figures/tmp_z.png',
                         cmap='RdBu'):
    """Plot the vorticity field.
    """
    fig, ax = plt.subplots()
    ax.pcolormesh(model.x.get(), model.y.get(), model.z.get(),
                  norm=mpl.colors.CenteredNorm(halfrange=halfrange),
                  cmap=cmap)
    ax.set_xlim(0., 2. * cp.pi)
    ax.set_ylim(0., 2. * cp.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_stream_function(model, halfrange=None, filename='figures/tmp_psi.png',
                         cmap='RdBu'):
    """Plot the stream function.
    """
    fig, ax = plt.subplots()
    ax.pcolormesh(model.x.get(), model.y.get(), model.psi.get(),
                  norm=mpl.colors.CenteredNorm(halfrange=halfrange),
                  cmap=cmap)
    ax.set_xlim(0., 2. * cp.pi)
    ax.set_ylim(0., 2. * cp.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_vorticity_field_upscale(model, halfrange=None, upscale_factor=4,
                                 filename='figures/tmp_z.png', cmap='RdBu'):
    """Plot the vorticity field using zero-padding in Fourier space to
    upscale by a specified factor in physical space.
    """
    m_up = Model(n_x=model.n_x * upscale_factor, precision=model.precision)
    filter = low_pass_spatial_filter(m_up, model.n_x)
    m_up.zk = cp.zeros_like(m_up.wv, dtype=model.complex_dtype)
    m_up.zk[filter] = model.zk.flatten() / (model.n_x / m_up.n_x) ** 2
    m_up._update_fields()

    fig, ax = plt.subplots()
    ax.pcolormesh(m_up.x.get(), m_up.y.get(), m_up.z.get(),
                  norm=mpl.colors.CenteredNorm(halfrange=halfrange),
                  cmap=cmap)
    ax.set_xlim(0., 2. * cp.pi)
    ax.set_ylim(0., 2. * cp.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_speed_field(model, filename='figures/tmp_speed.png'):
    """Plot speed.
    """
    fig, ax = plt.subplots()
    ax.pcolormesh(model.x, model.y, (model.u ** 2 + model.v ** 2) ** 0.5,
                  vmin=0., cmap='Greys_r')
    ax.set_xlim(0., 2. * cp.pi)
    ax.set_ylim(0., 2. * cp.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_isotropic_energy_spectrum(model, filename='figures/tmp_E.png',
                                   ymin=None, ymax=None):
    """Plot the isotropic energy spectrum.
    """
    kr, spec_iso = spatial_statistics.isotropic_energy_spectrum(model)
    fig, ax = plt.subplots()
    ax.loglog(kr.get(), spec_iso.get(), 'k')
    ax.loglog(kr.get(), kr.get() ** -(5 / 3), 'g--')
    ax.loglog(kr.get(), kr.get() ** -3, 'b--')
    ax.loglog(kr.get(), kr.get() ** -2, 'r--')
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_isotropic_enstrophy_spectrum(model, filename='figures/tmp_Z.png',
                                      ymin=None, ymax=None):
    """Plot the isotropic enstrophy spectrum.
    """
    kr, spec_iso = spatial_statistics.isotropic_enstrophy_spectrum(model)
    fig, ax = plt.subplots()
    ax.loglog(kr.get(), spec_iso.get(), 'k')
    ax.loglog(kr.get(), kr.get() ** +(1 / 3), 'g--')
    ax.loglog(kr.get(), kr.get() ** -1, 'b--')
    ax.loglog(kr.get(), kr.get() ** 0., 'r--')
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$Z(k)$")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_time_series(t, quantity, ymin=None, ylabel=None,
                     filename='figures/tmp_E'):
    fig, ax = plt.subplots()
    ax.plot(t, quantity, 'k')
    ax.set_ylim(ymin, None)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_zonally_averaged_velocity(model, filename='figures/tmp_ubar.png'):
    fig, ax = plt.subplots()
    ax.plot(model.u.mean(axis=1), model.y, 'k')
    ax.set_xlabel(r"$u(y)$")
    ax.set_ylabel(r"$y$")
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def low_pass_spatial_filter(model, new_n_x):
    """ Construct a low pass filter.
    """
    kmax = new_n_x // 2
    return (model.kx <= kmax) * (model.ky <= kmax) * (-model.ky < kmax)
