from model import Model
from timesteppers import AB3
import mechanisms
from random_fields import JMcW
from pathlib import Path
import plots
import spatial_statistics
import pickle
from time import time

run = "demo"

# Define model.
m = Model(1024, data_dir="data/" + run + "/", data_interval=100)
m.timestepper = AB3(m, 5e-4, 100.)
m.mechanisms = {
    'advection': mechanisms.DealiasedAdvection(m),
    'forcing': mechanisms.StochasticRingForcing(m, 90, 2, 1e-2),
    'friction': mechanisms.Diffusion(m, order=0., coefficient=1e-2),
    'hyperviscosity': mechanisms.Diffusion(m, order=2, coefficient=1e-9),
                }

# Specify initial condition.
m.zk = JMcW(m)


# Run the model.
t0 = time()
m.run()
t1 = time()
t_run = t1 - t0
print(f"Time taken: {t_run} seconds")


# Print some statistics of the final snapshot.
print(f"CFL: {m.cfl:.4f}")
print(f"Eddy turnover time: {spatial_statistics.eddy_turnover_time(m):.2f}")
print(f"Energy = {spatial_statistics.energy(m):.4f}")


# Save model.
m._update_fields()
with open(m.data_dir + r"m.pkl", "wb") as file:
    pickle.dump(m, file)


# Produce some figures.
fig_folder = "figures/" + run + "/"
if not Path(fig_folder).exists():
    Path(fig_folder).mkdir(parents=True)
suffix = f"_{m.timestepper.tn:.0f}"
plots.plot_vorticity_field(
    m, filename=fig_folder + "z" + suffix + ".png", halfrange=None)
plots.plot_isotropic_energy_spectrum(
    m, filename=fig_folder + "E" + suffix + ".png", ymin=1e-12)
plots.plot_isotropic_enstrophy_spectrum(
    m, filename=fig_folder + "Z" + suffix + ".png", ymin=1e-8)
