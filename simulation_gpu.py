import cupy as cp
from model_gpu import Model
from timesteppers import AB3
import mechanisms_gpu as mechanisms
import spatial_statistics_gpu as spatial_statistics
from random_fields_gpu import JMcW
import plots_gpu as plots
from pathlib import Path
from time import time
# import pickle

run = "stationary_2048_gpu_1_double"

m = Model(2048, data_dir="data/" + run + "/", data_interval=100,
          precision='double')  # !!!
m.timestepper = AB3(m, 1e-5, 100.)
m.mechanisms = {
    'advection': mechanisms.DealiasedAdvection(m),
    'forcing': mechanisms.StochasticRingForcing(m, 4, 1, 2.),
    'friction': mechanisms.Diffusion(m, order=-1., coefficient=1.),
    'viscosity': mechanisms.Diffusion(m, order=1, coefficient=1e-4),
}
zk = JMcW(m)
m.zk = cp.asarray(zk)
m._update_fields()

# with open("data/" + run + "/" + r"m.pkl", "rb") as file:
#     m = pickle.load(file)
# m.mechanisms['forcing'].rng = cp.random.default_rng(
#     seed=2)

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
print(f"Enstrophy = {spatial_statistics.enstrophy(m):.4f}")
print(f"Reynolds number: {spatial_statistics.reynolds_number(m):.2f}")
print(f"Energy centroid: {spatial_statistics.energy_centroid(m):.2f}")
print(f"Enstrophy centroid: {spatial_statistics.enstrophy_centroid(m):.2f}")
print("Energy dissipation due to viscosity = "
      + f"{spatial_statistics.energy_dissipation_due_to_viscosity(m).get():.4f}")  # noqa
print("Energy dissipation due to hypoviscosity = "
      + f"{spatial_statistics.energy_dissipation_due_to_hypoviscosity(m).get():.4f}")  # noqa


# m.timestepper.extend(50.)
# m.mechanisms['forcing'].rng = cp.random.default_rng(
#     seed=2)


fig_folder = "figures/" + run + "/"
if not Path(fig_folder).exists():
    Path(fig_folder).mkdir(parents=True)
suffix = f"_{m.timestepper.tn:.0f}"
plots.plot_vorticity_field(
    m, filename=fig_folder + "z" + suffix + ".png", halfrange=None)
plots.plot_vorticity_field_upscalef(
    m, filename=fig_folder + "z_up" + suffix + ".png", halfrange=None)
plots.plot_stream_function(
    m, filename=fig_folder + "psi" + suffix + ".png", halfrange=None)
plots.plot_isotropic_energy_spectrum(
    m, filename=fig_folder + "E" + suffix + ".png", ymin=1e-12)
# plots.plot_zonally_averaged_velocity(
#     m, filename=fig_folder + "ubar" + suffix + ".png")
plots.plot_isotropic_enstrophy_spectrum(
    m, filename=fig_folder + "Z" + suffix + ".png", ymin=1e-8)
