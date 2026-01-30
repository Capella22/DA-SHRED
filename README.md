# DA-SHRED: Data Assimilation and Discrepancy Modeling with Shallow Recurrent Decoders

A unified framework for **sparse sensing**, **latent-space PDE modeling**, **data assimilation**, **multi-scale reconstruction**, and **compressed operator discovery**.

This repository accompanies the methodology presented in three related publications and provides end-to-end implementations of:

- **DA-SHRED** — Data assimilation with discrepancy modeling 
- **Cheap2Rich** — Multi-fidelity framework for multiscale physics
- **SENDAI** — Hierarchical sparse-measurement data assimilation for remote sensing

The aim is to bridge *data-driven dynamical modeling*, *operator learning*, and *data assimilation* under a single modular architecture that is **lightweight**, operates in **sparse-measurement settings**, and is applicable to both physical simulations and real-world observational data.

---

## Publications

This repository supports the following publications:

1. **DA-SHRED** (Core Framework)  
   *"Data Assimilation and Discrepancy Modeling with Shallow Recurrent Decoders"*  
   [arXiv:2512.01170](https://arxiv.org/abs/2512.01170)  
   Full code release here

2. **Cheap2Rich** (Multi-Scale Physics / RDE Application)  
   *"Cheap2Rich: A Multi-Fidelity Framework for Data Assimilation and System Identification of Multiscale Physics — Rotating Detonation Engines"*  
   [arXiv:2601.20295](https://arxiv.org/abs/2601.20295)  
   Full code release: [github.com/kro0l1k/Cheap2Rich](https://github.com/kro0l1k/Cheap2Rich)

3. **SENDAI** (NDVI / Remote Sensing Application)  
   *"SENDAI: A Hierarchical Sparse-measurement, Efficient Data Assimilation Framework"*  
   [arXiv:2601.21664](https://arxiv.org/abs/2601.21664)  
   Full code release: [github.com/xswzaqnjimko/SENDAI_framework](https://github.com/xswzaqnjimko/SENDAI_framework)

---

## Key Features

### 1. DA-SHRED: Latent-Space Data Assimilation with Discrepancy Modeling

DA-SHRED extends the **Shallow Recurrent Decoder (SHRED)** architecture to enable data assimilation from simulation to reality. A key characteristic of DA-SHRED is that **the full state of the ground truth system is never observed** — only sparse sensor measurements are available from the target domain.

**SHRED Foundation:**
- Recurrent neural network (LSTM/GRU) learns a latent representation from temporal trajectories of sensor measurements
- Leverages Takens' embedding theorem: trades spatial information at a single time point for a trajectory of sensor measurements across time
- Shallow decoder maps the latent representation to the high-dimensional state space

**Data Assimilation Extensions:**
- Latent-space GAN alignment bridges simulation-to-reality distribution shifts
- No explicit dynamics model required at inference time
- Supports arbitrary sensor maps and missing-data scenarios
- Naturally supports extremely sparse and irregular sensor configurations
- Handles nonlinear and non-Gaussian uncertainty
- Lightweight architecture — trains in minutes on CPU/GPU

**Discrepancy Modeling with SINDy:**
- **Compressed Search SINDy**: Efficient operator-aware feature generation
- **Compressed Advancing SINDy**: Learning discrepancy operators without evaluating PDEs
- Operator discovery directly in latent space
- Recovers explicit governing equations for missing physics

---

### 2. Cheap2Rich: Multi-Fidelity Framework for Multi-Scale Physics

A multi-scale data assimilation architecture that bridges low-fidelity simulations and high-fidelity reality using only sparse sensor measurements.

- **Low-Frequency (LF) Pathway**: Captures dominant dynamics from simplified simulation models via SHRED, adapted to reality through DA-SHRED with explicit spectral filtering
- **High-Frequency (HF) Pathway**: Learns spectrally-sparse corrections from sensor-space residuals to capture unmodeled fine-scale physics
- **Spectral Sparsity Regularization**: Encourages physically interpretable, parsimonious representations of discrepancy dynamics
- **SINDy-Based Physics Discovery**: Recovers explicit governing equations distinguishing front dynamics from injector-driven corrections
- Demonstrated on **Rotating Detonation Engines (RDEs)** with 74.9% RMSE reduction

For complete implementation details, see the full code release at [github.com/kro0l1k/Cheap2Rich](https://github.com/kro0l1k/Cheap2Rich).

---

### 3. SENDAI: Hierarchical Data Assimilation for Remote Sensing

A hierarchical sparse-measurement, efficient data assimilation framework for reconstructing full spatial states from hyper-sparse sensor observations. Bridges domain shifts between simulation (historical periods) and ground truth (target periods)

- **Sequential Frequency Peeling**: Novel strategy decomposing high-frequency corrections into interpretable layers with explicit spectral constraints and frequency exclusion mechanisms
- **Coordinate-Based Implicit Neural Representations (INRs)**: Fourier positional encoding enables learning of high-frequency spatial patterns while producing spatially coherent reconstructions
- **Extreme Sparsity Reconstruction**: Achieves effective full-state reconstruction from only 64 sensors covering 1.56% of the spatial domain
- **Computational Efficiency**: Lightweight architecture enables training and inference on standard CPU hardware within minutes
- Demonstrated on **MODIS NDVI reconstruction** across six globally distributed sites, achieving up to 185% SSIM improvement over traditional baselines

For complete implementation details, see the full code release at [github.com/xswzaqnjimko/SENDAI_framework](https://github.com/xswzaqnjimko/SENDAI_framework).

---

### 4. PDE Demonstrations

Ready-to-run examples for various dynamical and real-world systems:

- 1D Kuramoto–Sivashinsky Equation
- 2D Kuramoto–Sivashinsky Equation
- Gray–Scott Reaction-Diffusion System
- Rotating Detonation Engine dynamics
- 2D NDVI Fields

---

## Repository Structure

```
DA-SHRED/
│
├── 2D models/
│   ├── 2DKS.py                    # DA-SHRED for 2D Kuramoto-Sivashinsky Equation
│   └── gray-scott.py              # DA-SHRED for 2D Gray-Scott Reaction-Diffusion System
│
├── Multiscale DA-SHRED/           # Multi-scale LF+HF decomposition framework
│
├── NDVI application/              # SENDAI implementation for satellite remote sensing
│                                  # (Full code: github.com/xswzaqnjimko/SENDAI_framework)
│
├── RDE Application/               # Cheap2Rich implementation for Rotating Detonation Engines
│                                  # (Full code: github.com/kro0l1k/Cheap2Rich)
│
├── utils/
│   ├── data_loaders.py
│   └── metrics.py
│
├── DASHRED_1DKS_example.py        # DA-SHRED for 1D Kuramoto-Sivashinsky Equation
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/Capella22/DA-SHRED.git
cd DA-SHRED
pip install -r requirements.txt
```

### Run a Basic Example

```bash
python DASHRED_1DKS_example.py
```

### For Cheap2Rich (RDE Application)

For the complete Cheap2Rich implementation including rotating detonation engine experiments, please refer to the full code release:

```bash
git clone https://github.com/kro0l1k/Cheap2Rich.git
```

See the [Cheap2Rich repository](https://github.com/kro0l1k/Cheap2Rich) for detailed instructions on training the multi-scale architecture and reproducing the RDE results.

### For SENDAI (NDVI Application)

For the complete SENDAI implementation including satellite remote sensing experiments, please refer to the full code release:

```bash
git clone https://github.com/xswzaqnjimko/SENDAI_framework.git
```

See the [SENDAI repository](https://github.com/xswzaqnjimko/SENDAI_framework) for detailed instructions on data acquisition, training the hierarchical architecture, and reproducing the NDVI reconstruction results.

---

## Citation

Relevant papers:

### DA-SHRED (Core Framework)
```bibtex
@article{bao2025data,
  title={Data Assimilation and Discrepancy Modeling with Shallow Recurrent Decoders},
  author={Bao, Yuxuan and Kutz, J. Nathan},
  journal={arXiv preprint arXiv:2512.01170},
  year={2025}
}
```

### Cheap2Rich (Multi-Scale Physics / RDE)
```bibtex
@article{bao2025cheap2rich,
  title={Cheap2Rich: A Multi-Fidelity Framework for Data Assimilation and System Identification of Multiscale Physics -- Rotating Detonation Engines},
  author={Bao, Yuxuan and Zajac, Jan and Powers, Megan and Raman, Venkat and Kutz, J. Nathan},
  journal={arXiv preprint arXiv:2601.20295},
  year={2026}
}
```

### SENDAI (Remote Sensing / NDVI)
```bibtex
@article{zhang2025sendai,
  title={SENDAI: A Hierarchical Sparse-measurement, Efficient Data Assimilation Framework},
  author={Zhang, Xingyue and Bao, Yuxuan and Gao, Mars Liyao and Kutz, J. Nathan},
  journal={arXiv preprint arXiv:2601.21664},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Related Repositories

- **Cheap2Rich (Full Implementation)**: [github.com/kro0l1k/Cheap2Rich](https://github.com/kro0l1k/Cheap2Rich)
- **SENDAI (Full Implementation)**: [github.com/xswzaqnjimko/SENDAI_framework](https://github.com/xswzaqnjimko/SENDAI_framework)
