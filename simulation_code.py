import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import matplotlib.animation as animation

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

def simulate_beam(EI, freq, plot_results=False):
    w = 2 * np.pi * freq

    K = np.zeros((total_DOFs, total_DOFs))
    M = np.zeros((total_DOFs, total_DOFs))

    for e in range(n_elements):
        k_e, m_e = beam_element_matrices(E, EI, rho, A, dx)
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

    for t in time:
        # Apply forced displacement at node 1
        u[2] = amplitude * np.sin(w * t)
        v[2] = amplitude * w * np.cos(w * t)
        a[2] = -amplitude * w**2 * np.sin(w * t)

        K_eff = M / (beta * dt**2) + K
        rhs = M @ ((1/(beta * dt**2)) * u + (1/(beta * dt)) * v + ((1/(2*beta)) - 1) * a)

        # Boundary conditions: fix node 0 and impose displacement at node 1
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

        displacements.append(u[::2])

        angles = []
        for i in range(n_elements):
            du = u[2*i+2] - u[2*i]
            angle = np.arctan2(du, dx)
            angles.append(angle)

        I_theta = I_solar * np.cos(angles)
        irradiance.append(I_theta)

    irradiance = np.array(irradiance)
    power = irradiance * A * dx
    energy_per_panel = np.sum(power * dt, axis=0)
    total_energy = np.sum(energy_per_panel)
    displacements = np.array(displacements)

    total_area = A * L
    available_energy = I_solar * total_area * T
    efficiency = total_energy / available_energy

    if plot_results:
        x_plot = np.linspace(0, L - dx, n_elements)

        plt.figure(figsize=(10, 4))
        plt.bar(x_plot, energy_per_panel, width=dx*0.8)
        plt.title(f"Energy per Panel - EI={EI:.1e}, f={freq}Hz")
        plt.ylabel("Energy (J)")
        plt.xlabel("Position along beam (m)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x_plot, irradiance[-1], marker='o')
        plt.title("Solar Irradiance at Final Time Step")
        plt.xlabel("Position along beam (m)")
        plt.ylabel("Irradiance (W/m²)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"🌞 Total harvested energy: {total_energy:.6f} J")
        print(f"⚡ Energy harvesting efficiency: {efficiency * 100:.2f}%")

    return total_energy, energy_per_panel, displacements, irradiance, efficiency

# ========================
# ANIMATIONS AND FRAMES
# ========================

def animate_beam(displacements, save=False, file_name="beam_animation.mp4"):
    fig, ax = plt.subplots()
    x_nodes = np.linspace(0, L, n_nodes)
    line, = ax.plot([], [], 'b.-')
    amp_max = np.max(np.abs(displacements))
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2 * amp_max, 1.2 * amp_max)
    ax.set_title("Flapping Wing Animation")

    def update(frame):
        line.set_data(x_nodes, displacements[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(displacements),
                                  interval=30, blit=True)

    if save:
        ani.save(file_name, writer='ffmpeg', fps=60)
        print(f"🎥 Animation saved as {file_name}")

    plt.show()

def animate_irradiance(irradiance, save=False, file_name="irradiance_animation.mp4"):
    fig, ax = plt.subplots()
    x_plot = np.linspace(0, L - dx, n_elements)
    line, = ax.plot([], [], 'r.-')
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1.2 * I_solar)
    ax.set_title("Solar Irradiance Animation")
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Irradiance (W/m²)')
    ax.grid(True)

    def update(frame):
        line.set_data(x_plot, irradiance[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(irradiance),
                                  interval=30, blit=True)

    if save:
        ani.save(file_name, writer='ffmpeg', fps=60)
        print(f"🎥 Animation saved as {file_name}")

    plt.show()

def save_beam_frame(displacements, frame_idx, file_name):
    x_nodes = np.linspace(0, L, n_nodes)
    plt.figure(figsize=(8, 4))
    plt.plot(x_nodes, displacements[frame_idx], 'b.-')
    plt.xlim(0, L)
    amp_max = np.max(np.abs(displacements))
    plt.ylim(-1.2 * amp_max, 1.2 * amp_max)
    plt.title(f"Beam Deformation - Frame {frame_idx}")
    plt.xlabel("Position (m)")
    plt.ylabel("Displacement (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"📸 Beam frame saved as {file_name}")

def save_irradiance_frame(irradiance, frame_idx, file_name):
    x_plot = np.linspace(0, L - dx, n_elements)
    plt.figure(figsize=(8, 4))
    plt.plot(x_plot, irradiance[frame_idx], 'r.-')
    plt.xlim(0, L)
    plt.ylim(0, 1.2 * I_solar)
    plt.title(f"Solar Irradiance - Frame {frame_idx}")
    plt.xlabel("Position (m)")
    plt.ylabel("Irradiance (W/m²)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    print(f"📸 Irradiance frame saved as {file_name}")

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    total_energy, energy_per_panel, displacements, irradiance, efficiency = simulate_beam(I_default, freq=10, plot_results=True)

    animate_beam(displacements, save=True, file_name="beam_animation.mp4")
    animate_irradiance(irradiance, save=True, file_name="irradiance_animation.mp4")

    # Save a frame at the middle of the motion
    frame_idx = len(time) // 2
    save_beam_frame(displacements, frame_idx, "beam_frame.png")
    save_irradiance_frame(irradiance, frame_idx, "irradiance_frame.png")
