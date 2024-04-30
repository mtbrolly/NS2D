"""
A basic demonstration of running a simulation.
"""

import cupy as cp
from model import Model
from timesteppers import AB3
import mechanisms
from random_fields import JMcW
import plots

# Instantiate a model.
m = Model(256, data_dir="data/demo/")

# Specify a timestepping scheme.
m.timestepper = AB3(m, 5e-4, 5.)

# Specify which mechanisms to include.
m.mechanisms = {
    'advection': mechanisms.DealiasedAdvection(m),
    'forcing': mechanisms.StochasticRingForcing(m, 4, 1, 1.),
    'friction': mechanisms.Diffusion(m, order=-1., coefficient=1.),
    'viscosity': mechanisms.Diffusion(m, order=1, coefficient=1e-3),
}

# Specify initial condition.
m.zk = cp.asarray(JMcW(m))

# Run the model.
m.run()

# Save (pickle) the model.
m.save_model()

# Plot some outputs/diagnostics.
plots.plot_vorticity_field(m, filename="z.png")
plots.plot_isotropic_energy_spectrum(m, filename="E.png")
