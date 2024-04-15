"""
Mechanisms to be implemented in a `Model` instance.

TODO:
    - Generalise stochastic forcing and improve docstrings.
"""

import pyfftw
import multiprocessing
import numpy as np
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.interfaces.cache.enable()
fftw = pyfftw.interfaces.numpy_fft


class SpectralFilter:
    """Exponential spectral filter for applying highly scale-selective but
    non-physical dissipation at small scales.
    """
    solution_mode = 'exact'

    def __init__(self, model):
        filterfac = 23.6
        cphi = 0.65 * np.pi
        wvx = np.sqrt(
            (model.kx * model.dx) ** 2. + (model.k_y * model.dx) ** 2.)
        exp_filter = np.exp(-filterfac * (wvx - cphi) ** 4.)
        exp_filter[wvx <= cphi] = 1.
        self.exp_filter = exp_filter
        self.model = model

    def __call__(self):
        self.model.zk *= self.exp_filter


class Advection:
    """Advection implemented without dealiasing.
    """
    solution_mode = 'approximate'

    def __init__(self, model):
        self.model = model

    def __call__(self):
        self.model.uk = -self.model.iky * self.model.psik
        self.model.vk = self.model.ikx * self.model.psik
        self.model.u = fftw.irfft2(self.model.uk)
        self.model.v = fftw.irfft2(self.model.vk)
        self.model.z_x = fftw.irfft2(self.model.ikx * self.model.zk)
        self.model.z_y = fftw.irfft2(self.model.iky * self.model.zk)
        self.model.nlk = (fftw.rfft2(self.model.u * self.model.z_x)
                          + fftw.rfft2(self.model.v * self.model.z_y))
        self.model.rhs = -self.model.nlk


class AdvectionWithTruncation:
    """Advection implemented with 2/3 rule for approximate dealiasing.
    """
    solution_mode = 'approximate'

    def __init__(self, model):
        self.model = model
        self.mask = (model.kx < model.n_x / 3.) & (
            np.abs(model.ky) < model.n_x / 3.)

    def __call__(self):
        self.model.uk = -self.model.iky * self.model.psik
        self.model.vk = self.model.ikx * self.model.psik
        self.model.u = fftw.irfft2(self.model.uk * self.mask)
        self.model.v = fftw.irfft2(self.model.vk * self.mask)
        self.model.z_x = fftw.irfft2(
            self.model.ikx * self.model.zk * self.mask)
        self.model.z_y = fftw.irfft2(
            self.model.iky * self.model.zk * self.mask)
        self.model.nlk = (fftw.rfft2(self.model.u * self.model.z_x)
                          + fftw.rfft2(self.model.v * self.model.z_y))
        self.model.rhs = -self.model.nlk


class DealiasedAdvection:
    """Advection dealiased using the 3/2 rule.
    """
    solution_mode = 'approximate'

    def __init__(self, model):
        self.model = model

        self.pad = 3. / 2.
        self.m_x = int(self.pad * self.model.n_x)
        self.m_k = int(self.pad * self.model.n_kx)
        self.padder = np.ones(self.m_x, dtype=bool)
        self.padder[int(self.model.n_x / 2):
                    int(self.model.n_x * (self.pad - 0.5)):] = False

    def __call__(self):
        self.nlk = np.zeros(self.model.zk.shape, dtype='complex128')
        self.model.uk = -self.model.iky * self.model.psik
        self.model.vk = self.model.ikx * self.model.psik

        # Create padded arrays
        self.uk_padded = np.zeros((self.m_x, self.m_k), dtype='complex128')
        self.vk_padded = np.zeros((self.m_x, self.m_k), dtype='complex128')
        self.z_xk_padded = np.zeros((self.m_x, self.m_k), dtype='complex128')
        self.z_yk_padded = np.zeros((self.m_x, self.m_k), dtype='complex128')

        # Enter known coefficients, leaving padded entries equal to zero
        self.uk_padded[self.padder, :self.model.n_kx] = self.model.uk[:, :]
        self.vk_padded[self.padder, :self.model.n_kx] = self.model.vk[:, :]
        self.z_xk_padded[self.padder, :self.model.n_kx] = (
            self.model.ikx * self.model.zk)[:, :]
        self.z_yk_padded[self.padder, :self.model.n_kx] = (
            self.model.iky * self.model.zk)[:, :]

        # Inverse transform padded arrays
        self.u_padded = fftw.irfft2(self.uk_padded)
        self.v_padded = fftw.irfft2(self.vk_padded)
        self.z_x_padded = fftw.irfft2(self.z_xk_padded)
        self.z_y_padded = fftw.irfft2(self.z_yk_padded)

        # Calculate Jacobian term
        self.nlk[:, :] = fftw.rfft2(
            (self.u_padded * self.z_x_padded + self.v_padded * self.z_y_padded
             ))[self.padder, :self.model.n_kx] * self.pad ** 2

        self.model.rhs = -self.nlk


class Diffusion:
    """Diffusion with `order` to be specified. Order refers to the power of the
    Laplacian. `order=1.` gives standard Newtonian viscosity; `order>2.` gives
    hyperviscosity; `order=0.` gives linear drag; `order=-1.` gives
    large-scale friction, etc.
    """
    solution_mode = 'exact'

    def __init__(self, model, order=1., coefficient=None):
        self.model = model
        self.order = order
        self.coefficient = coefficient
        if self.order >= 0.:
            self.n_order_visc = np.exp(
                -self.coefficient * self.model.timestepper.dt
                * self.model.wv2 ** self.order)
        else:
            self.n_order_visc = np.exp(
                -self.coefficient * self.model.timestepper.dt
                * self.model.wv2i ** (-self.order))

    def __call__(self):
        self.model.zk *= self.n_order_visc


class Beta:
    """Beta plane.
    """
    solution_mode = 'approximate'

    def __init__(self, model, beta):
        self.model = model
        self.beta = beta

    def __call__(self):
        self.model.rhs -= self.beta * (self.model.ikx * self.model.psik)


class StochasticRingForcing:
    """White-in-time stochastic forcing. Details to follow.
    """
    solution_mode = 'discrete'

    def __init__(self, model, k_f, dk_f, energy_input_rate, seed=1):
        self.model = model
        self.k_f = k_f
        self.dk_f = dk_f
        self.energy_input_rate = energy_input_rate
        self.rng = np.random.default_rng(seed=seed)
        self.band_filter = ((self.model.wv >= self.k_f - self.dk_f)
                            & (self.model.wv <= self.k_f + self.dk_f))
        self.fk_vars = (self.energy_input_rate * self.band_filter
                        * self.model.n_x ** 4 * self.model.wv * 0.5
                        / np.sum(self.band_filter * self.model.wv2i ** 0.5))

    def __call__(self):
        self.fk = np.reshape(self.rng.normal(size=self.model.wv.size)
                             + 1j * self.rng.normal(size=self.model.wv.size),
                             self.model.wv.shape) * self.fk_vars ** 0.5
        self.model.zk += self.fk * self.model.timestepper.dt ** 0.5
