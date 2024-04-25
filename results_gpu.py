import plots_gpu as plots
import spatial_statistics_gpu as spatial_statistics
import cupy as cp
from pathlib import Path
import pickle

run = "stationary_2048_gpu_1"
data_dir = "data/" + run + "/"
fig_folder = "figures/" + run + "/"
if not Path(fig_folder).exists():
    Path(fig_folder).mkdir(parents=True)

with open(data_dir + r"m.pkl", "rb") as file:
    m = pickle.load(file)

Tn = m.timestepper.Tn
dtn = m.data_interval * 10  # !!!
tn_out = cp.arange(1, Tn // dtn + 1) * dtn

E = []
Z = []
Re = []
kurt_u = []
kurt_z = []
E_centroid = []
Z_centroid = []
E_diss_visc = []
E_diss_hypo = []

for i in range(len(tn_out)):
    suffix = f"_{tn_out[i]:.0f}"
    try:
        m.zk = cp.load(data_dir + f"zk_{tn_out[i]:.0f}.npy")
    except FileNotFoundError:
        break
    m._update_fields()

    # plots.plot_vorticity_field(
    #     m, filename=fig_folder + "z_{:03d}.png".format(i))
    # plots.plot_isotropic_energy_spectrum(
    #     m, filename=fig_folder + "E_{:03d}.png".format(i), ymin=1e-12, ymax=10)
    # plots.plot_isotropic_enstrophy_spectrum(
    #     m, filename=fig_folder + "Z_{:03d}.png".format(i), ymin=1e-8, ymax=1e2)

    E.append(spatial_statistics.energy(m).get())
    Z.append(spatial_statistics.enstrophy(m).get())
    Re.append(spatial_statistics.reynolds_number(m).get())
    kurt_u.append(spatial_statistics.velocity_kurtosis(m).get())
    kurt_z.append(spatial_statistics.vorticity_kurtosis(m).get())
    E_centroid.append(spatial_statistics.energy_centroid(m).get())
    Z_centroid.append(spatial_statistics.enstrophy_centroid(m).get())
    E_diss_visc.append(
        spatial_statistics.energy_dissipation_due_to_viscosity(m).get())
    E_diss_hypo.append(
        spatial_statistics.energy_dissipation_due_to_hypoviscosity(m).get())

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
plots.plot_time_series(tn_out[:len(E_centroid)] * m.timestepper.dt, E_centroid,
                       ylabel=r'$k_{E}$',
                       filename=fig_folder + 'E_centroid_series.png', ymin=0.)
plots.plot_time_series(tn_out[:len(Z_centroid)] * m.timestepper.dt, Z_centroid,
                       ylabel=r'$k_{Z}$',
                       filename=fig_folder + 'Z_centroid_series.png', ymin=0.)
plots.plot_time_series(tn_out[:len(E_diss_visc)] * m.timestepper.dt,
                       E_diss_visc,
                       ylabel=r'$\epsilon_{\nu}$',
                       filename=fig_folder + 'E_diss_visc_series.png', ymin=0.)
plots.plot_time_series(tn_out[:len(E_diss_visc)] * m.timestepper.dt,
                       E_diss_hypo,
                       ylabel=r'$\epsilon_{\mu}$',
                       filename=fig_folder + 'E_diss_hypo_series.png', ymin=0.)


E_estimate = cp.ones_like(cp.array(E)) * cp.array(E)[0]
E_estimate -= cp.cumsum(cp.array(E_diss_visc)) * m.timestepper.dt * dtn
E_estimate -= cp.cumsum(cp.array(E_diss_hypo)) * m.timestepper.dt * dtn
E_estimate += cp.cumsum(cp.ones_like(cp.array(E))
                        ) * m.timestepper.dt * dtn * m.mechanisms[
                            'forcing'].energy_input_rate

plots.plot_time_series(tn_out[:len(E)] * m.timestepper.dt, E_estimate.get(),
                       ylabel=r'$E$',
                       filename=fig_folder + 'E_estimate_series.png', ymin=0.)
