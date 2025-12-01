# DA-SHRED: Data Assimilation and Discrepancy Modeling with Shallow Recurrent Decoders

A unified framework for **sparse sensing**, **latent-space PDE
modeling**, **data assimilation**, and **compressed operator
discovery**.

This repository accompanies the methodology presented in\
**"Data Assimilation and Discrepancy Modeling with Shallow Recurrent
Decoders"** and provides end-to-end implementations of:

-   **SHRED** --- Shallow Recurrent Decoder
-   **DA-SHRED** --- latent-space data assimilation
-   **Discrepancy Modeling** --- SINDy-like regression in the latent space for missing physics recovery
-   **Example workflows** --- Burgers, damped 2DKS, damped Gray--Scott, damped 1D RDE and other PDE systems

The aim is to bridge *data-driven dynamical modeling*, *operator learning*, and *data assimilation* under a single modular architecture.

------------------------------------------------------------------------

## 🌟 Key Features

### **1. SHRED: Shallow Recurrent Decoder**

A hybrid neural architecture for reconstructing full spatio-temporal PDE
fields from **sparse sensors only**.

-   GRU encoder over sliding windows of sensor traces
-   Latent state capturing local flow regime
-   Shallow MLP decoder for fast full-field reconstruction
-   Naturally supports extremely sparse / irregular sensor maps
-   Extremely lightweight --- trains in minutes on CPU/GPU

------------------------------------------------------------------------

### **2. DA-SHRED: Latent-Space Variational Assimilation**

Once trained, SHRED becomes a **decoder prior**, enabling posterior
estimation directly in the latent space.

-   Assimilation = gradient-based updates to latent trajectories
-   No explicit dynamics model required
-   Supports arbitrary sensor maps and missing-data cases
-   Handles nonlinear / non-Gaussian uncertainty

------------------------------------------------------------------------

### **3. Discrepancy Modeling + SINDy-like regression**

Implements **Algorithm 1: Compressed search SINDy and Algorithm 2: Compressed advancing SINDy**, enabling:

-   Efficient, operator-aware feature generation
-   Learning discrepancy operators **without** evaluating PDEs
-   Scales to hundreds--thousands of operator candidates
-   Operator discovery directly in latent space

------------------------------------------------------------------------

### **4. PDE Demonstrations Included**

Ready-to-run examples:

------------------------------------------------------------------------

## 📁 Repository Structure

    da-shred-sindy/
    │
    ├── models/
    │   ├── shred.py                # SHRED model (GRU encoder + shallow decoder)
    │   ├── sindy_utils.py          # SINDy library + enhanced STLSQ
    │   ├── compressed_reassembly.py# Algorithm 1 implementation
    │   ├── dashred.py              # Latent-space data assimilation
    │
    ├── scripts/
    │   ├── train_shred_with_sindy.py   # Train SHRED (+ optional SINDy regularization)
    │   ├── train_da_shred.py           # Assimilation + latent SINDy discovery
    │   ├── demo_burgers.py             # Full Burgers example
    │   ├── demo_gray_scott.py          # Damped Gray–Scott example
    │
    ├── utils/
    │   ├── data_loaders.py
    │   ├── metrics.py
    │
    ├── data/                    # Auto-populated by demos
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🚀 Getting Started

### **Install**


### **Run a toy example**



### **Train SHRED on your own dataset**


### **Run DA-SHRED data assimilation**



------------------------------------------------------------------------

## 📚 Paper

``` bibtex
@article{bao2025shred,
  title={Data Assimilation and Discrepancy Modeling with Shallow Recurrent Decoders},
  author={...},
  journal={...},
  year={2025}
}
```

------------------------------------------------------------------------
