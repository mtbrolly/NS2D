"""Plotting functions for standard plots.

TODO:
    - Add a function for producing videos.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spatial_statistics
plt.ioff()


def plot_vorticity_field(model, halfrange=None, filename='figures/tmp_z.png'):
    """Plot the vorticity field.
    """
    fig, ax = plt.subplots()
    ax.pcolormesh(model.x, model.y, model.z.T,
                  norm=mpl.colors.CenteredNorm(halfrange=halfrange),
                  cmap='RdBu')
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
    kr, spec_iso = spatial_statistics.isotropic_energy_spectrum(model)
    fig, ax = plt.subplots()
    ax.loglog(kr, spec_iso, 'k')
    ax.set_ylim(ymin, None)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()


def plot_isotropic_enstrophy_spectrum(model, filename='figures/tmp_Z.png',
                                      ymin=None):
    """Plot the isotropic enstrophy spectrum.
    """
    kr, spec_iso = spatial_statistics.isotropic_enstrophy_spectrum(model)
    fig, ax = plt.subplots()
    ax.loglog(kr, spec_iso, 'k')
    ax.set_ylim(ymin, None)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    fig.tight_layout()
    plt.savefig(filename, dpi=576)
    plt.close()
