"""Plotting functions for standard plots.

TODO:
    - Add a function for producing videos.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spatial_statistics_gpu
from scipy.interpolate import RegularGridInterpolator
import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)
fft_lib = scipy.fft
plt.ioff()


def plot_vorticity_field(model, halfrange=None, filename='figures/tmp_z.png'):
    """Plot the vorticity field.
    """
    fig, ax = plt.subplots()
    ax.pcolormesh(model.x.get(), model.y.get(), model.z.get(),
                  norm=mpl.colors.CenteredNorm(halfrange=halfrange),
                  cmap='RdBu')
    ax.set_xlim(0., 2. * np.pi)
    ax.set_ylim(0., 2. * np.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_vorticity_field_upscale(model, halfrange=None, upscale_factor=4,
                                 filename='figures/tmp_z.png'):
    """Plot the vorticity field.
    """
    x_l = model.x[0]
    y_l = model.y[:, 0]
    x_lp = np.concatenate(
        (x_l[0:1] - (x_l[1] - x_l[0]), x_l, x_l[-1:] + (x_l[1] - x_l[0])))
    y_lp = np.concatenate(
        (y_l[0:1] - (y_l[1] - y_l[0]), y_l, y_l[-1:] + (y_l[1] - y_l[0])))
    z_lp = np.pad(model.z, pad_width=1, mode='wrap')
    interp = RegularGridInterpolator((x_lp, y_lp), z_lp, method='linear')
    L = 2 * np.pi
    n_x = upscale_factor * model.n_x
    x, y = np.meshgrid(
        L * np.arange(0.5, n_x) / n_x,
        L * np.arange(0.5, n_x) / n_x)
    z = interp(
        np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
        ).reshape(x.shape)
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, z,
                  norm=mpl.colors.CenteredNorm(halfrange=halfrange),
                  cmap='RdBu')
    ax.set_xlim(0., 2. * np.pi)
    ax.set_ylim(0., 2. * np.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_vorticity_field_upscalef(model, halfrange=None, upscale_factor=4,
                                  filename='figures/tmp_z.png'):
    """Plot the vorticity field.
    """
    pad = upscale_factor
    m_x = int(pad * model.n_x)
    m_k = m_x // 2 + 1
    padder = np.ones(m_x, dtype=bool)
    padder[int(model.n_x / 2):
           int(model.n_x * (pad - 0.5)):] = False
    zk_padded = np.zeros((m_x, m_k), dtype='complex128')
    zk_padded[padder, :model.n_kx] = (
        model.zk)[:, :] * pad ** 2
    z_up = fft_lib.irfft2(zk_padded)

    L = 2 * np.pi
    n_x = upscale_factor * model.n_x
    x, y = np.meshgrid(
        L * np.arange(0.5, n_x) / n_x,
        L * np.arange(0.5, n_x) / n_x)
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, z_up,
                  norm=mpl.colors.CenteredNorm(halfrange=halfrange),
                  cmap='RdBu')
    ax.set_xlim(0., 2. * np.pi)
    ax.set_ylim(0., 2. * np.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_speed_field(model, filename='figures/tmp_speed.png'):
    """Plot the vorticity field.
    """
    fig, ax = plt.subplots()
    ax.pcolormesh(model.x, model.y, (model.u ** 2 + model.v ** 2) ** 0.5,
                  vmin=0., cmap='Greys_r')
    ax.set_xlim(0., 2. * np.pi)
    ax.set_ylim(0., 2. * np.pi)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_isotropic_energy_spectrum(model, filename='figures/tmp_E.png',
                                   ymin=None):
    """Plot the isotropic energy spectrum.
    """
    kr, spec_iso = spatial_statistics_gpu.isotropic_energy_spectrum(model)
    fig, ax = plt.subplots()
    ax.loglog(kr.get(), spec_iso.get(), 'k')
    ax.loglog(kr.get(), kr.get() ** -(5 / 3), 'g--')
    ax.loglog(kr.get(), kr.get() ** -3, 'b--')
    ax.loglog(kr.get(), kr.get() ** -2, 'r--')
    ax.set_ylim(ymin, None)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_isotropic_enstrophy_spectrum(model, filename='figures/tmp_Z.png',
                                      ymin=None):
    """Plot the isotropic enstrophy spectrum.
    """
    kr, spec_iso = spatial_statistics_gpu.isotropic_enstrophy_spectrum(model)
    fig, ax = plt.subplots()
    ax.loglog(kr.get(), spec_iso.get(), 'k')
    ax.loglog(kr.get(), kr.get() ** +(1 / 3), 'g--')
    ax.loglog(kr.get(), kr.get() ** -1, 'b--')
    ax.loglog(kr.get(), kr.get() ** 0., 'r--')
    ax.set_ylim(ymin, None)
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
