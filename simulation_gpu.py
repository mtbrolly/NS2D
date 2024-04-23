import cupy as cp
from model_gpu import Model
from timesteppers import AB3
import mechanisms_gpu as mechanisms
import spatial_statistics_gpu as spatial_statistics
from random_fields_gpu import JMcW
import plots_gpu as plots
from pathlib import Path
from time import time

run = "stationary_1024_gpu"

m = Model(1024, data_dir="data/" + run + "/", data_interval=100,
          precision='single')
m.timestepper = AB3(m, 1e-4, 10.)
m.mechanisms = {
    'advection': mechanisms.DealiasedAdvection(m),
    'forcing': mechanisms.StochasticRingForcing(m, 4, 1, 1e-2),
    'friction': mechanisms.Diffusion(m, order=-1., coefficient=1.5),
    'viscosity': mechanisms.Diffusion(m, order=1, coefficient=8.5e-5),
                }
zk = JMcW(m)
m.zk = cp.asarray(zk)
m._update_fields()

t0 = time()
m.run()
t1 = time()
t_run = t1 - t0
print(f"Time taken: {t_run} seconds")

m.save_model()

m._update_fields()
print(f"CFL: {m.cfl:.4f}")
print(f"Eddy turnover time: {spatial_statistics.eddy_turnover_time(m):.2f}")
print(f"Energy = {spatial_statistics.energy(m):.4f}")
print(f"Reynolds number: {spatial_statistics.reynolds_number(m):.2f}")
print(f"Energy centroid: {spatial_statistics.energy_centroid(m):.2f}")
print(f"Enstrophy centroid: {spatial_statistics.enstrophy_centroid(m):.2f}")


# fig_folder = "figures/" + run + "/"
# if not Path(fig_folder).exists():
#     Path(fig_folder).mkdir(parents=True)
# suffix = f"_{m.timestepper.tn:.0f}"
# plots.plot_vorticity_field(
#     m, filename=fig_folder + "z" + suffix + ".png", halfrange=None)
# plots.plot_vorticity_field_upscalef(
#     m, filename=fig_folder + "z_up" + suffix + ".png", halfrange=None)
# plots.plot_isotropic_energy_spectrum(
#     m, filename=fig_folder + "E" + suffix + ".png", ymin=1e-12)
# # plots.plot_zonally_averaged_velocity(
# #     m, filename=fig_folder + "ubar" + suffix + ".png")
# plots.plot_isotropic_enstrophy_spectrum(
#     m, filename=fig_folder + "Z" + suffix + ".png", ymin=1e-8)
