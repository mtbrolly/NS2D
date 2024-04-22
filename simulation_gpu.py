import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft

from model_gpu import Model
from timesteppers import AB3
import mechanisms_gpu as mechanisms
import spatial_statistics_gpu as spatial_statistics
from random_fields_gpu import JMcW
import plots_gpu as plots
from pathlib import Path
from time import time
import pickle
import timeit

scipy.fft.set_global_backend(cufft)

run = "gpu_test_1024_tmp"

m = Model(2048, data_dir="data/" + run + "/", data_interval=100)
m.timestepper = AB3(m, 1e-4, 100.)
m.mechanisms = {
    'advection': mechanisms.DealiasedAdvection(m),
    'forcing': mechanisms.StochasticRingForcing(m, 90, 2, 1e-2),
    'friction': mechanisms.Diffusion(m, order=0., coefficient=1e-2),
    'hyperviscosity': mechanisms.Diffusion(m, order=2, coefficient=1e-9),
                }
zk = JMcW(m)
m.zk = cp.asarray(zk)
m._update_fields()

t0 = time()
m.run()  # noqa
t1 = time()
t_run = t1 - t0
print(f"Time taken: {t_run} seconds")

with open(m.data_dir + r"m.pkl", "wb") as file:
    pickle.dump(m, file)

m._update_fields()
print(f"CFL: {m.cfl:.4f}")
print(f"Eddy turnover time: {spatial_statistics.eddy_turnover_time(m):.2f}")
print(f"Energy = {spatial_statistics.energy(m):.4f}")

fig_folder = "figures/" + run + "/"
if not Path(fig_folder).exists():
    Path(fig_folder).mkdir(parents=True)
suffix = f"_{m.timestepper.tn:.0f}"
plots.plot_vorticity_field(
    m, filename=fig_folder + "z" + suffix + ".png", halfrange=None)
# plots.plot_vorticity_field_upscalef(
#     m, filename=fig_folder + "z_up" + suffix + ".png", halfrange=None)
plots.plot_isotropic_energy_spectrum(
    m, filename=fig_folder + "E" + suffix + ".png", ymin=1e-12)
# plots.plot_zonally_averaged_velocity(
#     m, filename=fig_folder + "ubar" + suffix + ".png")
plots.plot_isotropic_enstrophy_spectrum(
    m, filename=fig_folder + "Z" + suffix + ".png", ymin=1e-8)
