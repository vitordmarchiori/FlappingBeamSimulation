# 🛩️ Flapping Beam Simulation with Solar Energy Harvesting – Flapping-Wing UAV

This project simulates, analyzes, and visualizes the structural behavior of a flapping-wing UAV (Unmanned Aerial Vehicle) wing equipped with solar panels along its surface.

It uses a **dynamic Euler-Bernoulli beam model**, discretized via the **1D Finite Element Method (FEM)**, and integrates real-time **solar irradiance calculation** during the flapping motion.

![](https://github.com/vitordmarchiori/FlappingBeamSimulation/blob/main/Animations/banner.gif)

<p align="center">
  <img src="Animations/banner.gif" alt="Flapping Wing UAV Simulation Banner" width="100%">
</p>

---

## 📦 Project Structure

- `simulate_beam()` – Performs the time-domain simulation of beam deflection, rotation, and irradiance capture.
- `animate_beam()` – Animates the flapping deformation with Hermitian interpolation and node tracking.
- `animate_irradiance()` – Animates and compares irradiance distributions across multiple configurations.
- **Automatic CSV/PNG/MP4 generation** for all key outputs.

---

## 🔧 Requirements

- Python 3.8+
- Required Libraries:
  - `numpy`
  - `matplotlib`
  - `pandas`

Optional:
  - `Pillow` or `ffmpeg` (for enhanced animation rendering)

---

## 🚀 Features and Functionality

Upon running the main script:

✅ Performs transient simulation using Newmark-beta integration  
✅ Visualizes beam deflection (with Hermite interpolation) and irradiance over time  
✅ Saves high-resolution `.mp4` animations and `.png` snapshots  
✅ Exports full simulation data to `.csv` for post-processing

---

## 📊 Outputs Generated

| Output | Description |
|--------|-------------|
| `beam_animation.mp4` | Beam flapping with real node motion and interpolation |
| `irradiance_animation.mp4` | Time-evolving irradiance distribution (comparative) |
| `energy_per_panel.png` | Energy harvested by each panel |
| `irradiance_final_step.png` | Irradiance snapshot at final time step |
| `nodal_displacements.csv` | Displacement time-history of each node |
| `irradiance.csv` | Irradiance time-history per element |
| `power.csv` | Instantaneous power received by each panel |

---

## 🎯 Objectives

- Quantify the effect of **beam flexibility and motion** on energy harvesting.
- Evaluate **parametric changes** (amplitude, area, frequency) and their outcomes.
- Enable **visual analysis** of physical and energetic behavior of flapping wings.

---

## 🧠 Possible Extensions

- Implement spatially varying **EI(x)** profiles
- Include **aerodynamic forces and damping**
- Add **solar incidence angle modeling** and sun tracking
- Use **optimization or ML** for performance tuning

---

## 🛠️ Customization

You can easily adjust:

| Parameter         | Description                            |
|------------------|----------------------------------------|
| `amplitude`       | Vertical motion at the driven node     |
| `E`, `I_default`  | Beam stiffness                         |
| `freq`, `amp_alt` | Frequency and amplitude of flapping    |
| `A`, `A_alt`      | Area of each solar panel               |
| `T`, `dt`         | Simulation time and time step          |

These are defined in the **Global Parameters** section of the script.

---

## 📌 Final Notes

- The model is modular and expandable.
- All simulations are fully **offline and deterministic**.
- Graphics are **publication-quality**, saved in high-resolution.
- Built for clarity, analysis, and educational exploration.

