"""
Timestepping schemes to be used together with a `Model` instance.
"""


def _AB1_step(model, dt):
    """One step of the first-order Adams-Bashforth scheme, aka forward Euler.
    """
    model.zk = model.zk + model.rhs * dt
    model.rhs_m1 = model.rhs.copy()


def _AB2_step(model, dt):
    """One step of the second-order Adams-Bashforth scheme.
    """
    model.zk = model.zk + (3 * model.rhs - model.rhs_m1) / 2. * dt
    model.rhs_m2 = model.rhs_m1.copy()
    model.rhs_m1 = model.rhs.copy()


def _AB3_step(model, dt):
    """One step of the third-order Adams-Bashforth scheme.
    """
    model.zk = model.zk + (
        23 * model.rhs - 16 * model.rhs_m1 + 5 * model.rhs_m2) / 12. * dt
    model.rhs_m2 = model.rhs_m1.copy()
    model.rhs_m1 = model.rhs.copy()


class AB3():
    """Third-order Adams-Bashforth timestepper with initialising steps.
    """

    def __init__(self, model, dt, T):
        self.model = model
        self.dt = dt
        self.T = T
        self.t = 0
        self.tn = 0
        self.Tn = int(self.T / self.dt)

    def step(self):
        if self.tn == 0:
            _AB1_step(self.model, self.dt)
        elif self.tn == 1:
            _AB2_step(self.model, self.dt)
        else:
            _AB3_step(self.model, self.dt)
        self.tn += 1
        self.t += self.dt

    def extend(self, new_T):
        self.T = new_T
        self.Tn = int(self.T / self.dt)
