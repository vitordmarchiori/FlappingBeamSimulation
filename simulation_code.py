import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import csv

plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 12

# ========================
# GLOBAL PARAMETERS
# ========================
L = 1.0
n_elements = 10
n_nodes = n_elements + 1
dx = L / n_elements
DOF_per_node = 2
total_DOFs = DOF_per_node * n_nodes

E = 2e9
rho = 1000
A = 1e-4
I_default = 1e-9

A_alt = 2e-4
amp_alt = 0.1
freq_alt = 5

dt = 0.001
T = 1.0
time = np.arange(0, T, dt)

amplitude = 0.05
I_solar = 170
beta = 0.25
gamma = 0.5

# ========================
# FEM FUNCTIONS AND SIMULATION
# ========================

def beam_element_matrices(E, I, rho, A, L):
    k = (E * I / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])
    m = (rho * A * L / 420) * np.array([
        [156, 22*L, 54, -13*L],
        [22*L, 4*L**2, 13*L, -3*L**2],
        [54, 13*L, 156, -22*L],
        [-13*L, -3*L**2, -22*L, 4*L**2]
    ])
    return k, m

def simulate_beam(EI, freq, A_override=None, amp_override=None, plot_results=False):
    w = 2 * np.pi * freq
    A_eff = A_override if A_override is not None else A
    amp_eff = amp_override if amp_override is not None else amplitude

    K = np.zeros((total_DOFs, total_DOFs))
    M = np.zeros((total_DOFs, total_DOFs))

    for e in range(n_elements):
        k_e, m_e = beam_element_matrices(E, EI, rho, A_eff, dx)
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k_e[i, j]
                M[dofs[i], dofs[j]] += m_e[i, j]

    u = np.zeros(total_DOFs)
    v = np.zeros(total_DOFs)
    a = np.zeros(total_DOFs)

    irradiance = []
    displacements = []
    nodal_displacements = []
    nodal_rotations = []

    for t in time:
        u[2] = amp_eff * np.sin(w * t)
        v[2] = amp_eff * w * np.cos(w * t)
        a[2] = -amp_eff * w**2 * np.sin(w * t)

        K_eff = M / (beta * dt**2) + K
        rhs = M @ ((1/(beta * dt**2)) * u + (1/(beta * dt)) * v + ((1/(2*beta)) - 1) * a)

        for dof in [0, 1]:
            K_eff[dof, :] = 0
            K_eff[:, dof] = 0
            K_eff[dof, dof] = 1
            rhs[dof] = 0

        K_eff[2, :] = 0
        K_eff[:, 2] = 0
        K_eff[2, 2] = 1
        rhs[2] = u[2]

        u_new = np.linalg.solve(K_eff, rhs)
        a_new = (u_new - u) / (beta * dt**2) - v / (beta * dt) - (1/(2*beta) - 1) * a
        v = v + dt * ((1 - gamma) * a + gamma * a_new)
        u = u_new
        a = a_new

        nodal_y = u[::2]
        nodal_theta = u[1::2]
        nodal_displacements.append(nodal_y.copy())
        nodal_rotations.append(nodal_theta.copy())

        x_vals = []
        y_vals = []
        for i in range(n_elements):
            u1 = u[2*i]
            th1 = u[2*i+1]
            u2 = u[2*i+2]
            th2 = u[2*i+3]
            xe = np.linspace(i*dx, (i+1)*dx, 5)
            xi = xe - i*dx
            l = dx
            N1 = 1 - 3*(xi/l)**2 + 2*(xi/l)**3
            N2 = xi - 2*(xi**2)/l + (xi**3)/l**2
            N3 = 3*(xi/l)**2 - 2*(xi/l)**3
            N4 = - (xi**2)/l + (xi**3)/l**2
            ye = N1*u1 + N2*th1 + N3*u2 + N4*th2
            x_vals.extend(xe.tolist())
            y_vals.extend(ye.tolist())
        displacements.append((np.array(x_vals), np.array(y_vals)))

        angles = []
        for i in range(n_elements):
            du = u[2*i+2] - u[2*i]
            angle = np.arctan2(du, dx)
            angles.append(angle)

        I_theta = I_solar * np.cos(angles)
        irradiance.append(I_theta)

    irradiance = np.array(irradiance)
    power = irradiance * A_eff * dx
    energy_per_panel = np.sum(power * dt, axis=0)
    total_energy = np.sum(energy_per_panel)

    displacements = np.array(displacements, dtype=object)
    nodal_displacements = np.array(nodal_displacements)
    nodal_rotations = np.array(nodal_rotations)

    total_area = A_eff * L
    available_energy = I_solar * total_area * T
    efficiency = total_energy / available_energy

    if plot_results:
        x_plot = np.linspace(0, L, n_elements)

        plt.figure(figsize=(10, 5))
        plt.bar(x_plot, energy_per_panel, width=dx*0.8, color='steelblue', edgecolor='black')
        plt.title(f"Energy per Panel - EI={EI:.1e}, f={freq}Hz", fontsize=14)
        plt.ylabel("Energy (J)")
        plt.xlabel("Position along beam (m)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("energy_per_panel.png", dpi=300)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(x_plot, irradiance[-1], color='steelblue', marker='o', linestyle='-', linewidth=2, markerfacecolor='steelblue')
        plt.title("Solar Irradiance at Final Time Step", fontsize=14)
        plt.xlabel("Position along beam (m)")
        plt.ylabel("Irradiance (W/m²)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("irradiance_final_step.png", dpi=300)
        plt.show()

        print(f"🌞 Total harvested energy: {total_energy:.6f} J")
        print(f"⚡ Energy harvesting efficiency: {efficiency * 100:.2f}%")

    pd.DataFrame(nodal_displacements, columns=[f"Node_{i}" for i in range(n_nodes)]).to_csv("nodal_displacements.csv", index=False)
    pd.DataFrame(irradiance, columns=[f"Elem_{i}" for i in range(n_elements)]).to_csv("irradiance.csv", index=False)
    pd.DataFrame(power, columns=[f"Elem_{i}" for i in range(n_elements)]).to_csv("power.csv", index=False)

    return total_energy, energy_per_panel, displacements, irradiance, efficiency, nodal_displacements, nodal_rotations

# ========================
# ANIMATIONS
# ========================

def animate_beam(displacements, nodal_displacements, nodal_rotations, save=False, file_name="beam_animation.mp4"):
    fig, ax = plt.subplots()
    line_interp, = ax.plot([], [], 'b-', linewidth=2, label='Interpolated')
    line_nodes, = ax.plot([], [], 'ro--', markersize=6, markerfacecolor='black', label='Nodes')
    amp_max = np.max([np.max(np.abs(d[1])) for d in displacements])
    ax.set_xlim(0, 1.1*L)
    ax.set_ylim(-1.2 * amp_max, 1.2 * amp_max)
    ax.set_title("Flapping Wing Displacement", fontsize=14)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Amplitude (m)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    def update(frame):
        x_interp, y_interp = displacements[frame]
        theta = nodal_rotations[frame]
        y = nodal_displacements[frame]
        x_nodes = [0.0]
        for i in range(1, n_nodes):
            dx_proj = dx * np.cos(theta[i-1])
            x_nodes.append(x_nodes[-1] + dx_proj)
        x_nodes = np.array(x_nodes)

        line_interp.set_data(x_interp / L, y_interp)
        line_nodes.set_data(x_nodes / L, y)
        return line_interp, line_nodes

    ani = animation.FuncAnimation(fig, update, frames=len(displacements), interval=30, blit=True)
    if save:
        ani.save(file_name, writer='ffmpeg', fps=60)
        print(f"🎥 Animation saved as {file_name}")
    plt.show()

def animate_irradiance(irr_sets, labels, save=False, file_name="irradiance_animation.mp4"):
    fig, ax = plt.subplots()
    x_plot = np.linspace(0, 1.0, n_elements)
    lines = [ax.plot([], [], label=label, linewidth=2)[0] for label in labels]

    ax.set_xlim(0, 1.1*L)
    ax.set_ylim(0.4 * I_solar, 1.1 * I_solar)
    ax.set_title("Solar Irradiance Animation", fontsize=14)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Irradiance (W/m²)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(x_plot, irr_sets[i][frame])
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(irr_sets[0]), interval=30, blit=True)
    if save:
        ani.save(file_name, writer='ffmpeg', fps=60)
        print(f"🎥 Animation saved as {file_name}")
    plt.show()

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    base = simulate_beam(I_default, freq=10, plot_results=True)
    alt_amp = simulate_beam(I_default, freq=freq_alt, amp_override=amp_alt, plot_results=True)
    alt_area = simulate_beam(I_default, freq=10, A_override=A_alt, plot_results=True)

    animate_beam(base[2], base[5], base[6], save=True, file_name="beam_animation.mp4")
    animate_irradiance([base[3], alt_amp[3], alt_area[3]],
                       labels=["Base", "Alt Freq/Amp", "Alt Area"],
                       save=True,
                       file_name="irradiance_animation.mp4")
