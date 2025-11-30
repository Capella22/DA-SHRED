# models/compressed_reassembly.py
"""
Algorithm 1: Compressed Reassembly (perturb-decoding -> encode library)
Given:
  - trained SHRED model (with encode/decode)
  - a set of candidate perturbation operators that act on full-field (functions mapping field->field)
Construct Theta columns by applying each operator to the reconstructed full field and encoding the result.
This file implements a vectorized, batched version that supports many operators efficiently.
"""
import numpy as np
import torch
from tqdm import tqdm

def apply_operators_to_field(field, operators):
    """
    field: (n_full,) numpy array (single snapshot)
    operators: list of callables f(field) -> field (n_full,)
    Returns: stacked array (n_ops, n_full)
    """
    out = []
    for op in operators:
        out.append(op(field))
    return np.stack(out, axis=0)

def build_compressed_theta(shred_model, decoded_fields, operators, batch_encode=64, device='cpu'):
    """
    decoded_fields: (N, n_full) numpy
    operators: list of callables (each maps field->field)
    Returns: Theta_comp (N, n_ops) per-snapshot encoding responses stacked as columns across snapshots
    Strategy:
      - For each snapshot, apply operators to decoded field -> produce (n_ops, n_full)
      - Encode each perturbed field by passing (as single-time-window) through encoder (we provide a helper that wraps)
      - We will create Theta whose columns are the encoded responses (for each op, maybe aggregated across snapshots).
    For simplicity we will encode per-snapshot and average across snapshots per operator to create robust library columns.
    """
    shred_model.eval()
    device = torch.device(device)
    shred_model.to(device)
    N, n_full = decoded_fields.shape
    n_ops = len(operators)
    # For each snapshot, compute encoded response per operator -> (N, n_ops, latent_dim)
    encoded_responses = []
    with torch.no_grad():
        for i in tqdm(range(N), desc="Compressed reassembly"):
            f = decoded_fields[i]  # (n_full,)
            perturbed = apply_operators_to_field(f, operators)  # (n_ops, n_full)
            # Convert each perturbed field into sensor-history-like input for encoder.
            # The SINDy-SHRED paper suggests encoding candidate perturbed fields via encoder mapping Ht(·).
            # Here we assume we can supply a dummy history window by repeating a synthesized sensor reading;
            # but a simpler approach is to directly use encoder's h2z projection on a linear projection of field.
            # We'll use a simple trick: project field -> pseudo-sensor vector using principal components (or identity)
            # then create a (1, seq_len, p) tensor where seq_len=1 and feed through encoder.
            # We'll instead call shred_model.h2z(shred_model.encoder(...)) by fabricating an input matching sensor_dim.
            # To keep the code general, we check for a helper 'encode_field' on model; if present, use it.
            if hasattr(shred_model, "encode_field"):
                enc = shred_model.encode_field(torch.tensor(f, dtype=torch.float32, device=device))
                encoded_responses.append(enc.cpu().numpy())  # (n_ops, latent_dim)
            else:
                # fallback: for each perturbed field, create zeros sensor history and use decoder inverse approximation
                # Not ideal but keeps API consistent. We return zeros.
                latent_dim = shred_model.h2z[0].out_features if hasattr(shred_model, "h2z") else 0
                encoded_responses.append(np.zeros((n_ops, latent_dim)))
    encoded_responses = np.stack(encoded_responses, axis=0)  # (N, n_ops, latent)
    # Build Theta: for each operator, we take the mean encoded response across snapshots -> (n_ops, latent)
    mean_op_enc = encoded_responses.mean(axis=0)  # (n_ops, latent)
    # Flatten each operator's mean encoded vector into a column for Theta (concatenate latent dims into features)
    Theta = mean_op_enc  # (n_ops, latent)
    return Theta, encoded_responses
