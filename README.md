# 🛩️ Flapping Beam Simulation with Solar Energy Harvesting – Flapping-Wing UAV

This project simulates, analyzes, and visualizes the structural behavior of a flapping-wing UAV (Unmanned Aerial Vehicle) wing equipped with solar panels distributed along its surface.

It uses a **dynamic Euler-Bernoulli beam model** discretized with the **1D Finite Element Method (FEM)**, integrating the **dynamic calculation of solar irradiance** captured by each panel during the flapping motion.

<p align="center">
  <img src="Animations/banner.gif" alt="Flapping Wing UAV Simulation Banner" width="100%">
</p>

---

## 📦 Project Structure

- `simulate_beam()` – Runs the dynamic simulation of the beam for a given stiffness and flapping frequency.
- `animate_beam()` – Animates the structural deformation of the wing during the flapping motion.
- `animate_irradiance()` – Animates the distribution of solar irradiance over the wing panels.
- `save_beam_frame()` – Saves `.png` images of the beam deformation at specific frames.
- `save_irradiance_frame()` – Saves `.png` images of the solar irradiance distribution at specific frames.

---

## 🔧 Requirements

- Python 3.8+
- Required Libraries:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `tabulate`
  - `Pillow` (optional: for GIF creation)
 
---

## 🚀 What to Expect

The script will automatically:
- Simulate the dynamic behavior of the beam
- Generate `.mp4` animations of the flapping motion and irradiance distribution
- Save `.png` snapshots of the beam and solar panels at a chosen frame

---

## 📊 Outputs Generated

| Output | Description |
|:-------|:------------|
| `Animations/beam_animation.mp4` | Animation of the beam flapping motion |
| `Animations/irradiance_animation.mp4` | Animation of the solar irradiance variation |
| `Frames/beam_frame.png` | Snapshot of the beam deformation |
| `Frames/irradiance_frame.png` | Snapshot of the solar irradiance distribution |

---

## 🎯 Engineering Objectives

- Analyze the effect of **structural flexibility** on solar energy harvesting efficiency.
- Optimize **flapping parameters** to maximize solar panel energy capture.
- **Visually understand** the interplay between deformation dynamics and energy harvesting.

---

## 🧠 Future Improvements

- Implement **variable stiffness EI(x)** profiles along the wing/beam.
- Add **aerodynamic forces** (e.g., wind or relative airflow effects).
- Introduce **solar tracking algorithms** for more realistic sun positioning.
- Optimize rigidity profiles and flapping motion using **machine learning techniques**.

---

## 📌 Notes

- This project is modular. You can adjust parameters such as amplitude, beam stiffness (EI), frequency, and total simulation time at the top of the `simulation_code.py` script.
- Animations and frames are automatically saved into organized folders.
