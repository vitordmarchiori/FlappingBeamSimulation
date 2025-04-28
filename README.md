<p align="center">
  <img src="Animations/banner.gif" alt="Flapping Wing UAV Simulation Banner" width="100%">
</p>

# 🛩️ Flapping Beam Simulation with Solar Energy Harvesting – Flapping-Wing UAV

This project simulates, analyzes, and visualizes the structural behavior of a flapping-wing UAV (Unmanned Aerial Vehicle) wing equipped with solar panels distributed along its structure.

It uses a **dynamic Euler-Bernoulli beam model** discretized with **1D Finite Element Method (FEM)**, also integrating the **calculation of solar irradiance** captured by each panel during the flapping motion.

## 📦 Project Structure

- `simular_viga()` – Runs the dynamic simulation of the beam for different stiffnesses and frequencies.
- `animar_viga()` – Animates the deformation of the wing over time.
- `animar_irradiancia()` – Animates the solar irradiance distribution over time across the panels.
- `salvar_frame_viga()` – Saves `.png` images of the beam deformation at specific time frames.
- `salvar_frame_irradiancia()` – Saves `.png` images of the solar irradiance distribution at specific time frames.
- `analise_tabelas()` – Generates comparison tables for energy harvested vs stiffness/frequency.


## 🔧 Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `tabulate`
- To save animations in `.mp4`, you must have `ffmpeg` installed:
  - Install via: `conda install -c conda-forge ffmpeg` or `sudo apt install ffmpeg`


## 🚀 How to Run

1. Execute the main script.
2. It will automatically:
   - Simulate the beam under default parameters.
   - Generate analysis tables (`.csv`).
   - Display energy and irradiance plots.
   - Save `.mp4` animations of deformation and irradiance.
   - Save `.png` snapshots for comparison at specific time frames.


## 📊 Generated Outputs

- `analise_rigidez.csv` – Energy captured for different beam stiffness values.
- `analise_frequencia.csv` – Energy captured for different flapping frequencies.
- `viga_animacao.mp4` – Animation of the flapping wing (structural deformation).
- `irradiancia_animacao.mp4` – Animation of the solar irradiance variation.
- `frame_viga.png` – Snapshot of the wing deformation at a specific time frame.
- `frame_irradiancia.png` – Snapshot of the irradiance distribution at a specific time frame.
- 

## 🎯 Engineering Objectives

- Analyze the impact of **structural flexibility** (variable EI) on solar energy harvesting efficiency.
- Optimize **rigidity and flapping profiles** to maximize energy capture.
- **Visually understand** how the flapping dynamics influence energy harvesting.


## 🧠 Future Improvements

- Implement **variable stiffness EI(x)** along the beam.
- Simulate **interactive aerodynamic loads** (relative airflow, wind).
- Include **3D motion** and **solar rotation tracking**.
- Optimize the **rigidity profile** using machine learning or numerical optimization techniques.

## 📌 Notes
This project is modular. You can easily adjust parameters such as amplitude, stiffness, frequency, and simulation time at the top of the script.

### 🚀 Designed for aero-structural and solar energy integration studies.
### 🔥 Ready to be used for studies, research publications, and advanced aerospace project development!
