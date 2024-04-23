import plots_gpu as plots
import spatial_statistics_gpu as spatial_statistics
import cupy as cp
from pathlib import Path
import pickle

run = "stationary_1024_gpu_2"
data_dir = "data/" + run + "/"
fig_folder = "figures/" + run + "/"
if not Path(fig_folder).exists():
    Path(fig_folder).mkdir(parents=True)

with open(data_dir + r"m.pkl", "rb") as file:
    m = pickle.load(file)

Tn = m.timestepper.Tn
dtn = m.data_interval
tn_out = cp.arange(1, Tn // dtn + 1) * dtn

E = []
Z = []
Re = []
kurt_u = []
kurt_z = []

for i in range(len(tn_out)):
    suffix = f"_{tn_out[i]:.0f}"
    try:
        m.zk = cp.load(data_dir + f"zk_{tn_out[i]:.0f}.npy")
    except FileNotFoundError:
        break
    m._update_fields()
    # plots.plot_vorticity_field(
    #     m, filename=fig_folder + "z_{:03d}.png".format(i))
    plots.plot_isotropic_energy_spectrum(
        m, filename=fig_folder + "E_{:03d}.png".format(i), ymin=1e-12)
    plots.plot_isotropic_enstrophy_spectrum(
        m, filename=fig_folder + "Z_{:03d}.png".format(i), ymin=1e-8)
    E.append(spatial_statistics.energy(m).get())
    Z.append(spatial_statistics.enstrophy(m).get())
    Re.append(spatial_statistics.reynolds_number(m).get())
    kurt_u.append(spatial_statistics.velocity_kurtosis(m).get())
    kurt_z.append(spatial_statistics.vorticity_kurtosis(m).get())

tn_out = tn_out.get()

plots.plot_time_series(tn_out[:len(E)] * m.timestepper.dt, E, ylabel=r'$E$',
                       filename=fig_folder + 'E_series.png', ymin=0.)
plots.plot_time_series(tn_out[:len(Z)] * m.timestepper.dt, Z,
                       filename=fig_folder + 'Z_series.png', ymin=0.)
plots.plot_time_series(tn_out[:len(Z)] * m.timestepper.dt, Re,
                       filename=fig_folder + 'Re_series.png', ymin=0.)
plots.plot_time_series(tn_out[:len(kurt_u)] * m.timestepper.dt, kurt_u,
                       ylabel=r'Kurt(u)',
                       filename=fig_folder + 'kurt_u_series.png', ymin=0.)
plots.plot_time_series(tn_out[:len(kurt_z)] * m.timestepper.dt, kurt_z,
                       ylabel=r'Kurt($\zeta$)',
                       filename=fig_folder + 'kurt_z_series.png', ymin=0.)
