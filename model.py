"""
Base module for defining model.
"""

import cupy as cp
from spatial_statistics import cfl
from pathlib import Path
import cupyx.scipy.fft as cufft
import scipy.fft
import pickle
scipy.fft.set_global_backend(cufft)
fft_lib = scipy.fft


class Model:
    """Defines a 2-D turbulence model.
    """

    def __init__(
        self,
        n_x,
        mechanisms=None,
        timestepper=None,
        data_dir=None,
        data_interval=100,
        precision='double'
    ):
        self.n_x = n_x
        self.precision = precision
        self.mechanisms = mechanisms
        self.timestepper = timestepper
        self.data_dir = data_dir
        self.data_interval = data_interval
        if precision == 'double':
            self.real_dtype = 'float64'
            self.complex_dtype = 'complex128'
        elif precision == 'single':
            self.real_dtype = 'float32'
            self.complex_dtype = 'complex64'
        elif precision == 'half':
            self.real_dtype = 'float16'
            self.complex_dtype = 'complex32'
        else:
            return
        self._construct_grids()
        if self.data_dir:
            self._create_data_dir()

    def _construct_grids(self):
        """Constructs spatial grids in real and Fourier space. Domain is
        [0, 2pi]^2 with grid points at the centres of a uniform rectangular
        mesh.
        """
        L = 2 * cp.pi
        self.x, self.y = cp.meshgrid(
            L * cp.arange(0.5, self.n_x, dtype=self.real_dtype) / self.n_x,
            L * cp.arange(0.5, self.n_x, dtype=self.real_dtype) / self.n_x)
        self.dx = L / self.n_x

        self.n_kx = self.n_x // 2 + 1
        self.kx = cp.arange(0., self.n_kx, dtype=self.real_dtype)
        self.ky = cp.append(
            cp.arange(0., self.n_x / 2, dtype=self.real_dtype),
            cp.arange(-self.n_x / 2, 0., dtype=self.real_dtype))

        self.kx, self.ky = cp.meshgrid(self.kx, self.ky)

        self.ikx = 1j * self.kx
        self.iky = 1j * self.ky

        self.wv2 = self.kx ** 2 + self.ky ** 2
        self.wv = cp.sqrt(self.wv2)

        self.wv2i = cp.zeros_like(self.wv2, dtype=self.real_dtype)
        self.wv2i[self.wv2 != 0.] = self.wv2[self.wv2 != 0.] ** -1

    def _evolve_one_step(self):
        """Evolve model one time step.
        """
        self.psik = -self.wv2i * self.zk
        approximate_mode_mechanisms = False
        for _, mechanism in self.mechanisms.items():
            if mechanism.solution_mode == 'approximate':
                mechanism()
                approximate_mode_mechanisms = True
        if not approximate_mode_mechanisms:
            self.rhs = 0.
        self.timestepper.step()
        for _, mechanism in self.mechanisms.items():
            if mechanism.solution_mode == 'discrete':
                mechanism()
        for _, mechanism in self.mechanisms.items():
            if mechanism.solution_mode == 'exact':
                mechanism()

    def _update_fields(self):
        """Update all fields to value at current timestep.
        """
        self.psik = -self.wv2i * self.zk
        self.uk = -self.iky * self.psik
        self.vk = self.ikx * self.psik
        self.z = fft_lib.irfft2(self.zk)
        self.psi = fft_lib.irfft2(self.psik)
        self.u = fft_lib.irfft2(self.uk)
        self.v = fft_lib.irfft2(self.vk)

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
        self._save_data()
        while (self.timestepper.tn < self.timestepper.Tn):
            self._check_cfl()
            self._evolve_one_step()
            if self.timestepper.tn % self.data_interval == 0:
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
        cp.save(self.data_dir + f"zk_{self.timestepper.tn:.0f}.npy", self.zk)

    def save_model(self):
        """Save Model instance.
        """
        self._update_fields()
        for (k, v) in self.__dict__.items():
            if isinstance(v, cp.ndarray):
                v = v.get()
        for _, object in self.mechanisms.items():
            for (k, v) in object.__dict__.items():
                if isinstance(v, cp.ndarray):
                    v = v.get()
        if 'forcing' in self.mechanisms:
            del self.mechanisms['forcing'].rng
        with open(self.data_dir + r"m.pkl", "wb") as file:
            pickle.dump(self, file)
