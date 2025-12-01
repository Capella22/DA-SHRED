# DA-SHRED: Data Assimilation and Discrepancy Modeling with Shallow Recurrent Decoders

A unified framework for **sparse sensing**, **latent-space PDE
modeling**, **data assimilation**, and **compressed operator
discovery**.

This repository accompanies the methodology presented in\
**"Data Assimilation and Discrepancy Modeling with Shallow Recurrent
Decoders"** and provides end-to-end implementations of:

-   **SHRED** --- Shallow Recurrent Decoder\
-   **DA-SHRED** --- latent-space data assimilation\
-   **Discrepancy Modeling** --- SINDy-like latent regression for
    recovering missing physics\
-   **PDE workflows** --- Burgers, damped Gray--Scott, and more

The aim is to bridge *data-driven dynamical modeling*, *operator
learning*, and *practical data assimilation* under a single modular
deep-learning architecture.

------------------------------------------------------------------------

## 🌟 Key Features

### **1. SHRED: Shallow Recurrent Decoder**

A hybrid neural architecture for reconstructing full spatio-temporal PDE
fields from **sparse sensors only**.

-   GRU encoder over sliding windows of sensor traces\
-   Latent state capturing local flow regime\
-   Shallow MLP decoder for fast full-field reconstruction\
-   Naturally supports extremely sparse / irregular sensor maps\
-   Extremely lightweight --- trains in minutes on CPU/GPU

> "...the shallow recurrent decoder integrates a recurrent encoder with
> a minimal-capacity MLP decoder, enabling stable field reconstruction
> while avoiding overfitting to localized sensor noise or transient
> artifacts."

------------------------------------------------------------------------

### **2. SINDy-SHRED: Learning Latent Dynamics**

SHRED can be regularized with a **latent-space SINDy loss**, enforcing
parsimonious latent dynamics.

-   Polynomial + trigonometric library construction\
-   Enhanced STLSQ with threshold scheduling\
-   Extracts sparse, interpretable latent ODEs\
-   Works even for complex or partially known PDE physics

> "...physical dynamics often admit multiple, equally compatible
> spatial--temporal correlation structures... SINDy regularization
> constrains the latent space to a minimal physically consistent
> manifold."

------------------------------------------------------------------------

### **3. DA-SHRED: Latent-Space Variational Assimilation**

Once trained, SHRED becomes a **decoder prior**, enabling posterior
estimation directly in the latent space.

-   Assimilation = gradient-based updates to latent trajectories\
-   No explicit dynamics model required\
-   Supports arbitrary sensor maps and missing-data cases\
-   Handles nonlinear / non-Gaussian uncertainty

> "...the decoder serves as a learned surrogate mapping from latent
> states to full fields, while assimilation consists of optimizing
> latent trajectories to match observed data."

------------------------------------------------------------------------

### **4. Discrepancy Modeling + Compressed Reassembly**

Implements **Algorithm 1: Compressed Reassembly**, enabling:

-   Efficient, operator-aware feature generation\
-   Learning discrepancy operators **without** evaluating PDEs\
-   Scales to hundreds--thousands of operator candidates\
-   Operator discovery directly in latent space

Features include:

-   Perturb → decode → encode pipeline\
-   Vectorized operator batching\
-   Mean-encoded operator signatures forming SINDy Θ-library columns

> "...baseline reconstructed fields are perturbed by candidate
> discrepancy operators, then re-embedded through the encoder, producing
> operator signatures consistent with the learned representation."

------------------------------------------------------------------------

### **5. PDE Demonstrations Included**

Ready-to-run examples:

-   **1D Viscous Burgers**\
-   **Damped Gray--Scott Reaction--Diffusion (U--V system)**\
-   Modular scripts for training, assimilation, SINDy extraction\
-   All demos run quickly on CPU

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

``` bash
pip install -r requirements.txt
```

### **Run a toy example**

``` bash
python scripts/demo_burgers.py
```

### **Train SHRED on your own dataset**

``` bash
python scripts/train_shred_with_sindy.py     --sim my_sim.npy     --sensor_map my_sensor_map.npy     --window 10     --epochs 50     --latent_dim 64     --sindy_reg 1.0     --sindy_lambda 0.1
```

### **Run DA-SHRED data assimilation**

``` bash
python scripts/train_da_shred.py     --shred_ckpt shred_ckpt.pth     --sensor_map my_sensor_map.npy     --sensors my_sensor_seq.npy     --sindy_discover
```

------------------------------------------------------------------------

## 📚 Citation

``` bibtex
@article{your2025shred,
  title={Data Assimilation and Discrepancy Modeling with Shallow Recurrent Decoders},
  author={...},
  journal={...},
  year={2025}
}
```

------------------------------------------------------------------------

## 🧠 What This Repository Enables

-   Reconstruction from **very sparse sensors**\
-   **Interpretable** latent-space dynamics\
-   Practical **data assimilation without PDE solvers**\
-   Scalable **operator discrepancy learning**\
-   Unified pipeline for sensing → reconstruction → modeling → discovery

------------------------------------------------------------------------

## 🛠 Future Extensions

-   Neural-operator priors (FNO / DeepONet)\
-   Adaptive sensor placement via Fisher--Tikhonov criteria\
-   Multi-resolution PDE fields\
-   Stochastic latent dynamics\
-   Physics-augmented compressed reassembly (Augmented Θ libraries)
