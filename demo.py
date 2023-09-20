from model import Model
from timesteppers import AB3
from mechanisms import (DealiasedAdvection,
                        Diffusion, StochasticRingForcing)
from random_fields import JMcW
import plots
import spatial_statistics
from time import time

m = Model(256, data_dir="data/stationary/")
m.timestepper = AB3(m, 1e-3, 1000.)
m.mechanisms = (
    DealiasedAdvection(m),
    StochasticRingForcing(m, 4, 1, 1e7),
    Diffusion(m, order=-1., coefficient=1.5),
    Diffusion(m, coefficient=1.5e-3)
    )

m.zk = JMcW(m)  # np.zeros_like(m.kx)

t0 = time()
m.run()
t1 = time()
t_run = t1 - t0
print(f"Time taken: {t_run} seconds")
print(f"CFL: {m.cfl:.4f}")
ETT = spatial_statistics.eddy_turnover_time(m)
print(f"Eddy turnover time: {ETT:.2f}")

m._update_fields()
fig_folder = "figures/stationary/"
suffix = f"_{m.timestepper.t:.0f}"
plots.plot_vorticity_field(
    m, filename=fig_folder + "z" + suffix + ".png")
plots.plot_isotropic_energy_spectrum(
    m, filename=fig_folder + "E" + suffix + ".png", ymin=None)
plots.plot_isotropic_enstrophy_spectrum(
    m, filename=fig_folder + "Z" + suffix + ".png", ymin=None)
