"""
A model class for simulating homogeneous two-dimensional turbulence.

Solves 2-D Navier-Stokes using a standard pseudospectral method with AB3
timestepping. Linear terms are solved exactly.

Dissipation mechanisms implemented:
    - linear friction;
    - hypoviscosity (or "large-scale friction");
    - Newtonian viscosity;
    - Nth-order hyperviscosity;
    - a low-pass, exponential, spectral filter.

Also:
    - a white-in-time, Gaussian-in-space stochastic forcing on a band of
      wavenumbers;
    - beta.


Martin Brolly, 2022.
"""

import numpy as np
from numpy import pi
import diagnostic_tools
import pyfftw.interfaces.numpy_fft as fftw
from pathlib import Path


class Model():
    def __init__(
        self,
        L=2 * pi,
        W=None,
        nx=512,
        ny=None,

        dt=5e-4,
        Tend=40,
        twrite=1000.,

        dissipation='filter',
        filterfac=23.6,
        nu=None,
        nu_2=None,
        nu_n=None,
        viscosity_order=None,

        beta=0.,

        k_f=None,
        f_a=None,
        E_input_rate=None,
        seed=1,

        d_a=None,
        lsf_a=None,

        dealias=False,

        scalar=False,
        mol_diff=None,

        data_dir=None,
    ):

        if ny is None:
            ny = nx
        if W is None:
            W = L

        self.nx = nx
        self.ny = ny
        self.L = L
        self.W = W

        self.beta = beta

        self.k_f = k_f
        if E_input_rate:
            self.f_a = E_input_rate * nx ** -4. * k_f ** 2. * 6e20
        else:
            self.f_a = f_a
        self.seed = seed

        self.d_a = d_a
        self.lsf_a = lsf_a

        self.dt = dt
        self.Tend = Tend
        self.Tendn = int(self.Tend / self.dt)
        self.twrite = twrite
        self.data_dir = data_dir

        self.dissipation = dissipation
        self.filterfac = filterfac
        self.nu = nu
        self.nu_2 = nu_2
        self.nu_n = nu_n
        self.viscosity_order = viscosity_order

        self.dealias = dealias

        self.scalar = scalar
        self.mol_diff = mol_diff

        self._initialise_grid()
        self._initialise_filter()
        self._initialise_frictions()
        self._initialise_time()
        if data_dir:
            self._create_data_dir()

        if self.k_f:
            self._initialise_forcing()

    def _initialise_time(self):
        """
        Initialise time and timestep at zero.
        """
        self.t = 0
        self.tn = 0

    def _initialise_grid(self):
        """
        Define spatial and spectral grids and related constants, as well as
        padding tools if dealiasing is in use.
        """
        self.x, self.y = np.meshgrid(
            np.arange(0.5, self.nx, 1.) / self.nx * self.L,
            np.arange(0.5, self.ny, 1.) / self.ny * self.W)

        self.dk = 2. * pi / self.L
        self.dl = 2. * pi / self.W

        self.nl = self.ny
        self.nk = int(self.nx / 2 + 1)
        self.ll = self.dl * np.append(np.arange(0., self.nx / 2),
                                      np.arange(-self.nx / 2, 0.))
        self.kk = self.dk * np.arange(0., self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)  # noqa: E741
        self.ik = 1j * self.k
        self.il = 1j * self.l

        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # Constant for spectral normalizations
        self.nxny = self.nx * self.ny

        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt(self.wv2)

        iwv2 = self.wv2 != 0.
        self.wv2i = np.zeros_like(self.wv2)
        self.wv2i[iwv2] = self.wv2[iwv2] ** -1

        # Spectral dealiasing setup
        if self.dealias:
            self.pad = 3. / 2.
            self.mx = int(self.pad * self.nx)
            self.mk = int(self.pad * self.nk)
            self.padder = np.ones(self.mx, dtype=bool)
            self.padder[int(self.nx / 2):
                        int(self.nx * (self.pad - 0.5)):] = False

    def _initialise_filter(self):
        """
        Define low-pass, exponential, spectral filter for small scale
        dissipation.
        """
        cphi = 0.65 * pi
        wvx = np.sqrt((self.k * self.dx) ** 2. + (self.l * self.dy) ** 2.)
        exp_filter = np.exp(-self.filterfac * (wvx - cphi) ** 4.)
        exp_filter[wvx <= cphi] = 1.
        self.exp_filter = exp_filter

    def _initialise_frictions(self):
        if self.dissipation == 'viscosity':
            self.visc = np.exp(-self.nu * self.dt * self.wv2)
        if self.dissipation == 'n-order':
            assert (
                self.nu_n and self.viscosity_order), 'nu_n and viscosity_order must defined for hyperviscosity.'  # noqa: E501
            self.n_order_visc = np.exp(
                -self.nu_n * self.dt * self.wv2 ** self.viscosity_order)
        if self.mol_diff:
            self.scalar_visc = np.exp(-self.mol_diff * self.dt * self.wv2)
        if self.d_a:
            self.drag = np.exp(-self.d_a * self.dt)
        if self.lsf_a:
            self.lsf = np.exp(-self.lsf_a * self.dt * self.wv2i)

    def _initialise_forcing(self):
        """
        Set up random forcing.
        """
        self.f_rng = np.random.default_rng(seed=self.seed)
        F = ((self.wv > self.k_f - 2.) & (self.wv < self.k_f + 2.)) * self.f_a
        self.fk_vars = F / ((self.wv + (self.wv == 0)) * pi) / 2

    def _generate_forcing(self):
        """
        Generate a (new) realisation of random forcing.
        """
        self.fk = np.reshape(self.f_rng.normal(size=self.wv.size)
                             + 1j * self.f_rng.normal(size=self.wv.size),
                             self.wv.shape) * np.sqrt(self.fk_vars)

    def _check_cfl(self):
        """
        Assert that the CFL number is less than unity.
        """
        if self.tn % self.twrite == 0:
            self._calc_derived_fields()
            self.cfl = diagnostic_tools.calc_cfl(self)
            assert self.cfl < 1., "CFL condition violated."

    def _create_data_dir(self):
        if not Path(self.data_dir).exists():
            Path(self.data_dir).mkdir(parents=True)

    def _save_data(self):
        """
        Save model field data.
        """
        if self.tn % self.twrite == 0:
            np.save(self.data_dir + f"zk_{self.tn:.0f}.npy", self.zk)

    def _calc_z(self):
        """
        Compute z from zk.
        """
        self.z = fftw.irfft2(self.zk)

    def _calc_zk(self):
        """
        Compute zk from z.
        """
        self.zk = fftw.rfft2(self.z)

    def _calc_psi(self):
        """
        Compute psi from zk.
        """
        self.psik = -self.wv2i * self.zk
        self.psi = fftw.irfft2(self.psik)

    def _calc_dealiased_advection(self):
        """
        Calculate dealiased advection term from zk.
        """
        self.nlk = np.zeros(self.zk.shape, dtype='complex128')
        self.uk = -self.il * self.psik
        self.vk = self.ik * self.psik

        # Create padded arrays
        self.uk_padded = np.zeros((self.mx, self.mk), dtype='complex128')
        self.vk_padded = np.zeros((self.mx, self.mk), dtype='complex128')
        self.z_xk_padded = np.zeros((self.mx, self.mk), dtype='complex128')
        self.z_yk_padded = np.zeros((self.mx, self.mk), dtype='complex128')

        # Enter known coefficients, leaving padded entries equal to zero
        self.uk_padded[self.padder, :self.nk] = self.uk[:, :]
        self.vk_padded[self.padder, :self.nk] = self.vk[:, :]
        self.z_xk_padded[self.padder, :self.nk] = (self.ik * self.zk)[:, :]
        self.z_yk_padded[self.padder, :self.nk] = (self.il * self.zk)[:, :]

        # Inverse transform padded arrays
        self.u_padded = fftw.irfft2(self.uk_padded)
        self.v_padded = fftw.irfft2(self.vk_padded)
        self.z_x_padded = fftw.irfft2(self.z_xk_padded)
        self.z_y_padded = fftw.irfft2(self.z_yk_padded)

        # Calculate Jacobian term
        self.nlk[:, :] = fftw.rfft2((self.u_padded * self.z_x_padded
                                     + self.v_padded
                                     * self.z_y_padded)
                                    )[self.padder, :self.nk] * self.pad ** 2
        return self.nlk

    def _calc_tendency(self):
        """
        Calculates "RHS" of barotropic vorticity equation.
        """
        self.psik = -self.wv2i * self.zk

        if self.dealias:
            self.RHS = -self._calc_dealiased_advection()
        else:
            self.uk = -self.il * self.psik
            self.vk = self.ik * self.psik

            self.u = fftw.irfft2(self.uk)
            self.v = fftw.irfft2(self.vk)

            self.z_x = fftw.irfft2(self.ik * self.zk)
            self.z_y = fftw.irfft2(self.il * self.zk)

            # Advection of vorticity
            self.nlk = (fftw.rfft2(self.u * self.z_x)
                        + fftw.rfft2(self.v * self.z_y))
            self.RHS = -self.nlk

            # Advection of scalar
            if self.scalar:
                self.C_x = fftw.irfft2(self.ik * self.Ck)
                self.C_y = fftw.irfft2(self.il * self.Ck)
                self.RHS_scalar = -(fftw.rfft2(self.u * self.C_x)
                                    + fftw.rfft2(self.v * self.C_y))

        # Beta effect
        if self.beta:
            self.RHS -= self.beta * (self.ik * self.psik)

        # Random forcing
        if self.f_a:
            self._generate_forcing()
            self.RHS += self.fk

        # # Linear drag
        # if self.d_a:
        #     self.drag = np.exp(-self.d_a * self.dt)

        # # Large scale friction
        # if self.lsf_a:
        #     self.lsf = np.exp(-self.lsf_a * self.dt * self.wv2i)

        # Viscosity
        # if self.dissipation == 'viscosity':
        #     self.visc = np.exp(-self.nu * self.dt * self.wv2)

        # # Nth-order (hyper)viscosity
        # if self.dissipation == 'n-order':
        #     assert self.nu_n, 'nu_n must defined for hyperviscosity.'
        #     self.n_order_visc = np.exp(
        #         -self.nu_n * self.dt * self.wv2 ** self.viscosity_order)

    def _step_forward(self):
        """
        Evolve zk one step according to barotropic vorticity equation.
        """
        self._calc_tendency()

        # Timestepping: Adams--Bashforth 3rd order.
        if self.tn == 0:
            self.zk = (self.zk + self.dt * self.RHS)
            if self.dissipation == 'filter':
                self.zk *= self.exp_filter
            elif self.dissipation == 'viscosity':
                self.zk *= self.visc
            elif self.dissipation == 'n-order':
                self.zk *= self.n_order_visc
            if self.d_a:
                self.zk *= self.drag
            if self.lsf_a:
                self.zk *= self.lsf
            if self.scalar:
                self.Ck = (self.Ck + self.dt * self.RHS_scalar)
                self.Ck *= self.exp_filter
            if self.mol_diff:
                self.Ck *= self.mol_visc

        elif self.tn == 1:
            self.zk = self.zk + (self.dt / 2) * \
                (3 * self.RHS - self.RHS_m1)
            if self.dissipation == 'filter':
                self.zk *= self.exp_filter
            if self.dissipation == 'viscosity':
                self.zk *= self.visc
            if self.dissipation == 'n-order':
                self.zk *= self.n_order_visc
            if self.d_a:
                self.zk *= self.drag
            if self.lsf_a:
                self.zk *= self.lsf
            if self.scalar:
                self.Ck = self.Ck + (self.dt / 2) * (3 * self.RHS_scalar
                                                     - self.RHS_scalar_m1)
                self.Ck *= self.exp_filter
            if self.mol_diff:
                self.Ck *= self.mol_visc

        else:
            self.zk = self.zk + (self.dt / 12) * (
                23 * self.RHS - 16 * self.RHS_m1 + 5 * self.RHS_m2)
            if self.dissipation == 'filter':
                self.zk *= self.exp_filter
            if self.dissipation == 'viscosity':
                self.zk *= self.visc
            if self.dissipation == 'n-order':
                self.zk *= self.n_order_visc
            if self.d_a:
                self.zk *= self.drag
            if self.lsf_a:
                self.zk *= self.lsf
            if self.scalar:
                self.Ck = self.Ck + (self.dt / 12) * (
                    23 * self.RHS_scalar - 16 * self.RHS_scalar_m1
                    + 5 * self.RHS_scalar_m2)
                self.Ck *= self.exp_filter
            if self.mol_diff:
                self.Ck *= self.mol_visc

        # Record preceding tendencies
        if self.tn > 0:
            self.RHS_m2 = self.RHS_m1.copy()
            if self.scalar:
                self.RHS_scalar_m2 = self.RHS_scalar_m1.copy()
        self.RHS_m1 = self.RHS.copy()
        if self.scalar:
            self.RHS_scalar_m1 = self.RHS_scalar.copy()

        self._check_cfl()
        assert (~np.isnan(self.zk)).all(), 'Vorticity is nan.'
        if self.data_dir:
            self._save_data()

        self.tn += 1
        self.t += self.dt

    def _calc_derived_fields(self):
        """
        Typically only zk is explicitly updated during timestepping; this
        updates other fields based on the current zk.
        """
        self.psik = -self.wv2i * self.zk
        self.uk = -self.il * self.psik
        self.vk = self.ik * self.psik

        self.z = fftw.irfft2(self.zk)
        self.psi = fftw.irfft2(self.psik)
        self.u = fftw.irfft2(self.uk)
        self.v = fftw.irfft2(self.vk)

        if self.scalar:
            self.C = fftw.irfft2(self.Ck)

    def run(self):
        """
        Run model uninterrupted until final time.
        """
        while (self.t < self.Tend):
            self._step_forward()
            if self.tn == self.Tendn:
                break
        self._calc_derived_fields()

    def run_with_snapshots(self, tsnapint=1.):
        """
        Run model with interruptions at set intervals to allow for other
        calculations to be done, e.g. plotting, evolving particles.
        """
        tsnapints = np.ceil(tsnapint / self.dt)

        while (self.t < self.Tend):
            self._step_forward()
            if (self.tn % tsnapints) == 0:
                self._calc_z()
                yield self.t
            if self.tn > self.Tendn:
                break
        self._calc_derived_fields()
