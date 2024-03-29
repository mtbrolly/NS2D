"""
Base module for defining model.
"""

import pyfftw
import numpy as np
from spatial_statistics import cfl
from pathlib import Path
fftw = pyfftw.interfaces.numpy_fft
pyfftw.config.NUM_THREADS = 8
pyfftw.interfaces.cache.enable()


class Model:
    """Defines a 2-D turbulence model.
    """

    def __init__(
        self,
        n_x,
        mechanisms=None,
        timestepper=None,
        data_dir=None
    ):
        self.n_x = n_x
        self.mechanisms = mechanisms
        self.timestepper = timestepper
        self.data_dir = data_dir
        self._construct_grids()
        self._create_data_dir()

    def _construct_grids(self):
        """Constructs spatial grids in real and Fourier space. Domain is
        [0, 2pi]^2 with grid points at the centres of a uniform rectangular
        mesh.
        """
        L = 2 * np.pi
        self.x, self.y = np.meshgrid(
            L * np.arange(0.5, self.n_x) / self.n_x,
            L * np.arange(0.5, self.n_x) / self.n_x)
        self.dx = L / self.n_x

        self.n_kx = self.n_x // 2 + 1
        self.kx = np.arange(0., self.n_kx)
        self.ky = np.append(np.arange(0., self.n_x / 2),
                            np.arange(-self.n_x / 2, 0.))

        self.kx, self.ky = np.meshgrid(self.kx, self.ky)

        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky

        self.wv2 = self.kx ** 2 + self.ky ** 2
        self.wv = np.sqrt(self.wv2)

        self.wv2i = np.zeros_like(self.wv2)
        self.wv2i[self.wv2 != 0.] = self.wv2[self.wv2 != 0.] ** -1

    def _evolve_one_step(self):
        """Evolve model one time step.
        """
        self.psik = -self.wv2i * self.zk

        for mechanism in self.mechanisms:
            if mechanism.solution_mode == 'approximate':
                mechanism()
        self.timestepper.step()
        for mechanism in self.mechanisms:
            if mechanism.solution_mode == 'discrete':
                mechanism()
        for mechanism in self.mechanisms:
            if mechanism.solution_mode == 'exact':
                mechanism()

    def _update_fields(self):
        """Update all fields to value at current timestep.
        """
        self.psik = -self.wv2i * self.zk
        self.uk = -self.iky * self.psik
        self.vk = self.ikx * self.psik
        self.z = fftw.irfft2(self.zk)
        self.psi = fftw.irfft2(self.psik)
        self.u = fftw.irfft2(self.uk)
        self.v = fftw.irfft2(self.vk)

    def _check_cfl(self):
        """
        Assert that the CFL number is less than unity.
        """
        if self.timestepper.tn % 10 == 0:
            self._update_fields()
            self.cfl = cfl(self)
            assert self.cfl < 1., "CFL condition violated."

    def run(self):
        """Run model until final time.
        """
        self._update_fields()
        while (self.timestepper.t < self.timestepper.T):
            self._check_cfl()
            self._evolve_one_step()
            if self.timestepper.tn % 1000 == 0:
                print(f"Time: {self.timestepper.t:.2f}")
                if self.data_dir:
                    self._save_data()
            if self.timestepper.tn == self.timestepper.Tn:
                break
        self._update_fields()

    def _create_data_dir(self):
        if not Path(self.data_dir).exists():
            Path(self.data_dir).mkdir(parents=True)

    def _save_data(self):
        """
        Save model field data.
        """
        np.save(self.data_dir + f"zk_{self.timestepper.tn:.0f}.npy", self.zk)
