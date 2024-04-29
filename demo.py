"""
A basic demonstration of running a simulation.
"""

import cupy as cp
from model import Model
from timesteppers import AB3
import mechanisms as mechanisms
from random_fields import JMcW
import plots as plots

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

m.zk = cp.asarray(JMcW(m))

m.run()

m.save_model()

plots.plot_vorticity_field(m, filename="z.png")
plots.plot_isotropic_energy_spectrum(m, filename="E.png")
