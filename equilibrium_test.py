from model import Model
import inits

m = Model(nx=1024, Tend=500., twrite=40, dt=2.5e-4,
          data_dir='data/equilibrium_test/',
          k_f=64, f_a=1e10, lsf_a=1., dealias=True)

m.zk = inits.zero(m)
m.run()
