#!/usr/bin/env python3
"""
Covariance Learning Experiment Script

This script implements Transformer-based models for learning covariance matrix estimation
from multivariate time series data. It reuses the existing Transformer architectures
from QP_func_approx.py and adapts them for covariance learning tasks.

Usage:
    python covariance_experiment.py --model_type SoftmaxTransformer --seq_len 50 --n_variables 5
    python covariance_experiment.py --model_type LinearTransformer --seq_len 100 --n_variables 10
    python covariance_experiment.py --model_type MLP --seq_len 50 --n_variables 5

Author: Assistant
Date: 2025-01-21
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global parameters (can be overridden by command line arguments)
SEQ_LEN = 50
N_VARIABLES = 4
NOISE_LEVEL = 0.1
NUM_TRAIN_SAMPLES = 5000
NUM_VAL_SAMPLES = 1000
NUM_TEST_SAMPLES = 1000
BATCH_SIZE = 64
NUM_LAYERS = 4
NUM_HEADS = 2
HIDDEN_DIM = 4
DROPOUT_RATE = 0.1
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
PATIENCE = 20
RANDOM_SEED = 42
MODEL_TYPE = "BasicLinearTransformer"


class CovarianceDataset(Dataset):
    """Dataset for covariance learning experiments"""

    def __init__(self, num_samples, seq_len, n_variables, noise_level=0.1):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.n_variables = n_variables
        self.noise_level = noise_level

        # Generate data
        self.sequences = []
        self.covariances = []
        self.true_covariances = []

        print(f"Generating {num_samples} covariance samples...")
        for _ in tqdm(range(num_samples), desc="Generating data"):
            # Generate random covariance matrix (theoretical)
            original_cov = self.generate_random_covariance()

            # Generate multivariate sequence and get actual covariance
            sequence, true_cov = self.generate_multivariate_sequence(original_cov)

            # Encode true covariance matrix as vector (label)
            cov_vector = self.encode_covariance(true_cov)

            self.sequences.append(sequence)
            self.covariances.append(cov_vector)
            self.true_covariances.append(true_cov)

    def generate_random_covariance(self):
        """Generate a random positive definite covariance matrix"""
        A = np.random.randn(self.n_variables, self.n_variables)
        cov_matrix = A @ A.T + 0.1 * np.eye(self.n_variables)

        return cov_matrix

    def generate_multivariate_sequence(self, original_cov):
        """Generate multivariate sequence from covariance matrix"""
        mean = np.zeros(self.n_variables)
        X = np.random.multivariate_normal(mean, original_cov, self.seq_len)

        # Add noise
        noise = np.random.normal(0, self.noise_level, X.shape)
        X_noisy = X + noise

        # Calculate actual covariance of the noisy sequence
        true_cov = np.cov(X_noisy.T)

        return X_noisy, true_cov

    def encode_covariance(self, cov_matrix):
        """Encode covariance matrix as vector (upper triangular)"""
        upper_tri_indices = np.triu_indices(self.n_variables)
        cov_vector = cov_matrix[upper_tri_indices]
        return cov_vector

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.float32),
            "covariance": torch.tensor(self.covariances[idx], dtype=torch.float32),
            "true_covariance": torch.tensor(
                self.true_covariances[idx], dtype=torch.float32
            ),
        }


def vector_to_covariance_matrix(cov_vector, n_variables):
    """Convert vector back to covariance matrix"""
    batch_size = cov_vector.size(0)
    cov_matrix = torch.zeros(
        batch_size, n_variables, n_variables, device=cov_vector.device
    )

    # Reconstruct upper triangular matrix
    idx = 0
    for i in range(n_variables):
        for j in range(i, n_variables):
            cov_matrix[:, i, j] = cov_vector[:, idx]
            if i != j:  # Non-diagonal elements
                cov_matrix[:, j, i] = cov_vector[:, idx]
            idx += 1

    return cov_matrix


def covariance_reconstruction_loss(pred_cov_vector, true_cov_matrix, n_variables):
    """Pure Frobenius norm loss for covariance matrix reconstruction"""

    # Convert predicted vector to covariance matrix
    pred_cov_matrix = vector_to_covariance_matrix(pred_cov_vector, n_variables)

    # Frobenius norm loss: calculate per-sample, then average over batch
    frobenius_losses = torch.norm(pred_cov_matrix - true_cov_matrix, "fro", dim=(1, 2))
    frobenius_loss = frobenius_losses.mean()

    return frobenius_loss


class CovarianceTransformer(nn.Module):
    """Transformer for covariance learning (based on QPTransformer)"""

    def __init__(
        self,
        seq_len,
        n_variables,
        hidden_dim=128,
        num_layers=4,
        num_heads=2,
        dropout=0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_variables = n_variables
        self.hidden_dim = hidden_dim

        # Calculate dimensions
        output_dim = n_variables * (n_variables + 1) // 2

        # Input projection: project each time step (n_variables) to hidden_dim
        self.input_projection = nn.Linear(n_variables, hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        batch_size = sequence.size(0)

        # Keep sequence structure: (batch_size, seq_len, n_variables)
        # Project each time step to hidden dimension
        x = self.input_projection(sequence)  # (batch_size, seq_len, hidden_dim)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Global average pooling over sequence length
        x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Output projection
        x = self.output(x)

        return x


class CovarianceLinearAttentionLayer(nn.Module):
    """Linear attention layer implementation for covariance learning"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm architecture
        # Self-attention with linear attention
        norm_x = self.norm1(x)
        attn_out = self.linear_attention(norm_x)
        x = x + self.dropout(attn_out)

        # Feed-forward network
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)

        return x

    def linear_attention(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reshape for multi-head attention
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Linear attention: O = V * (K^T * Q)
        # Compute K^T * Q (without softmax)
        kt = k.transpose(-2, -1)  # [batch_size, num_heads, head_dim, seq_len]
        attn_scores = torch.matmul(q, kt)  # [batch_size, num_heads, seq_len, seq_len]

        # Apply linear attention: O = V * (K^T * Q)
        # Scale by 1/sqrt(d_h) for numerical stability
        scale = 1.0 / (self.head_dim**0.5)
        attn_scores = attn_scores * scale
        out = torch.matmul(attn_scores, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.out_proj(out)


class CovarianceLinearTransformer(nn.Module):
    """Linear Transformer for covariance learning"""

    def __init__(
        self,
        seq_len,
        n_variables,
        hidden_dim=256,
        num_layers=4,
        num_heads=2,
        dropout=0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_variables = n_variables
        self.hidden_dim = hidden_dim

        # Calculate dimensions
        output_dim = n_variables * (n_variables + 1) // 2

        # Input projection: project each time step (n_variables) to hidden_dim
        self.input_projection = nn.Linear(n_variables, hidden_dim)

        # Linear attention layers
        self.layers = nn.ModuleList(
            [
                CovarianceLinearAttentionLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        batch_size = sequence.size(0)

        # Keep sequence structure: (batch_size, seq_len, n_variables)
        # Project each time step to hidden dimension
        x = self.input_projection(sequence)  # (batch_size, seq_len, hidden_dim)

        # Apply linear attention layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling over sequence length
        x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Output projection
        x = self.output(x)

        return x


class BasicLinearTransformer(nn.Module):
    """Basic Linear Transformer for covariance learning - no input projection, hidden_dim = n_variables"""

    def __init__(
        self,
        seq_len,
        n_variables,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.n_variables = n_variables
        self.hidden_dim = n_variables  # Must equal n_variables

        # Calculate dimensions
        output_dim = n_variables * (n_variables + 1) // 2

        # No input projection - directly use n_variables as hidden_dim
        # Linear attention layers
        self.layers = nn.ModuleList(
            [
                CovarianceLinearAttentionLayer(self.hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, sequence):
        batch_size = sequence.size(0)

        # Keep sequence structure: (batch_size, seq_len, n_variables)
        # No input projection - use sequence directly
        x = sequence  # (batch_size, seq_len, n_variables)

        # Apply linear attention layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling over sequence length
        x = x.mean(dim=1)  # (batch_size, n_variables)

        # Output projection
        x = self.output(x)

        return x


class CovarianceMLP(nn.Module):
    """Small MLP baseline for covariance learning"""

    def __init__(self, seq_len, n_variables, hidden_dim=16, dropout=0.1):
        super().__init__()

        input_dim = seq_len * n_variables
        output_dim = n_variables * (n_variables + 1) // 2

        # Small MLP architecture with ~1000 parameters
        self.layers = nn.Sequential(
            # Input layer: 200 → 16
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Hidden layer 1: 16 → 12
            nn.Linear(hidden_dim, hidden_dim - 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output layer: 12 → 10
            nn.Linear(hidden_dim - 4, output_dim),
        )

    def forward(self, sequence):
        batch_size = sequence.size(0)
        x = sequence.reshape(batch_size, -1)
        return self.layers(x)


def create_model(
    model_type, seq_len, n_variables, hidden_dim, num_layers, num_heads, dropout
):
    """Create model based on type"""
    if model_type == "SoftmaxTransformer":
        return CovarianceTransformer(
            seq_len=seq_len,
            n_variables=n_variables,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif model_type == "LinearTransformer":
        return CovarianceLinearTransformer(
            seq_len=seq_len,
            n_variables=n_variables,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif model_type == "BasicLinearTransformer":
        return BasicLinearTransformer(
            seq_len=seq_len,
            n_variables=n_variables,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif model_type == "MLP":
        return CovarianceMLP(
            seq_len=seq_len,
            n_variables=n_variables,
            # hidden_dim=hidden_dim,  # hidden_dim self set to 16 for MLP
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, dataloader, n_variables):
    """Evaluate covariance model"""
    model.eval()

    total_frobenius_loss = 0.0
    total_samples = 0
    frobenius_errors = []

    # Add R² and NMSE calculation
    all_predictions = []  # Store predicted covariance vectors
    all_true_covariances = []  # Store true covariance vectors
    all_nmse = []  # Store NMSE for each sample

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            sequence = batch["sequence"].to(DEVICE)
            true_cov = batch["true_covariance"].to(DEVICE)

            # Predict
            pred_cov_vector = model(sequence)
            pred_cov_matrix = vector_to_covariance_matrix(pred_cov_vector, n_variables)

            # Process each sample in the batch individually
            for i in range(sequence.size(0)):
                # Frobenius error per sample
                pred_sample = pred_cov_matrix[i]  # (n_variables, n_variables)
                true_sample = true_cov[i]  # (n_variables, n_variables)
                frobenius_error = torch.norm(pred_sample - true_sample, "fro")
                frobenius_errors.append(frobenius_error.item())

                total_frobenius_loss += frobenius_error.item()
                total_samples += 1

                # Add R² and NMSE calculation
                pred_vector = (
                    pred_cov_vector[i].cpu().numpy()
                )  # Predicted covariance vector
                # Encode true covariance matrix as vector (upper triangular)
                upper_tri_indices = np.triu_indices(n_variables)
                true_vector = true_sample.cpu().numpy()[
                    upper_tri_indices
                ]  # True covariance vector

                all_predictions.append(pred_vector)
                all_true_covariances.append(true_vector)

                # Calculate NMSE: ||pred - true||² / ||true||²
                residual_norm_squared = np.sum((pred_vector - true_vector) ** 2)
                true_norm_squared = np.sum(true_vector**2)
                nmse = (
                    residual_norm_squared / true_norm_squared
                    if true_norm_squared > 0
                    else 0.0
                )
                all_nmse.append(nmse)

    # Calculate R² = 1 - SS_res / SS_tot
    all_predictions = np.array(all_predictions)  # (N, vector_dim)
    all_true_covariances = np.array(all_true_covariances)  # (N, vector_dim)

    # Calculate overall mean
    y_mean = np.mean(all_true_covariances, axis=0)  # (vector_dim,)

    # Calculate SS_res and SS_tot
    ss_res = np.sum((all_predictions - all_true_covariances) ** 2)
    ss_tot = np.sum((all_true_covariances - y_mean) ** 2)

    # Calculate R²
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "frobenius_loss": total_frobenius_loss / total_samples,
        "frobenius_mean": np.mean(frobenius_errors),
        "frobenius_median": np.median(frobenius_errors),
        "frobenius_95th": np.percentile(frobenius_errors, 95),
        "frobenius_errors": frobenius_errors,  # Add for plotting
        "r_squared": r_squared,  # Add R²
        "all_nmse": np.array(all_nmse),  # Add NMSE array
        "all_predictions": all_predictions,  # Add predictions
        "all_true_covariances": all_true_covariances,  # Add true values
    }


def plot_training_curves(train_losses, val_losses, output_folder):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(output_folder, "training_curves.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_folder, "training_curves.pdf"), bbox_inches="tight")
    plt.close()

    print(f"\nTraining curves saved to {output_folder}:")


def plot_frobenius_distribution(frobenius_errors, output_folder):
    """Plot Frobenius norm distribution histogram"""
    plt.figure(figsize=(8, 6))
    plt.hist(frobenius_errors, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(
        np.mean(frobenius_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(frobenius_errors):.4f}",
    )
    plt.axvline(
        np.median(frobenius_errors),
        color="blue",
        linestyle="--",
        label=f"Median: {np.median(frobenius_errors):.4f}",
    )
    plt.axvline(
        np.percentile(frobenius_errors, 95),
        color="orange",
        linestyle=":",
        label=f"95th percentile: {np.percentile(frobenius_errors, 95):.4f}",
    )
    plt.xlabel("Frobenius Norm Error", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.title("Distribution of Frobenius Norm Errors", fontweight="bold")
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.legend(prop={"weight": "bold"}, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_folder, "frobenius_distribution.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_nmse_distribution(all_nmse, output_folder):
    """Plot NMSE distribution histogram"""
    plt.figure(figsize=(8, 6))
    plt.hist(all_nmse, bins=30, alpha=0.7, color="lightcoral", edgecolor="black")
    plt.axvline(
        np.mean(all_nmse),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_nmse):.4f}",
    )
    plt.axvline(
        np.median(all_nmse),
        color="blue",
        linestyle="--",
        label=f"Median: {np.median(all_nmse):.4f}",
    )
    plt.axvline(
        np.percentile(all_nmse, 95),
        color="orange",
        linestyle=":",
        label=f"95th percentile: {np.percentile(all_nmse, 95):.4f}",
    )
    plt.xlabel("NMSE per Sample", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.title("Distribution of NMSE (Normalized MSE)", fontweight="bold")
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.legend(prop={"weight": "bold"}, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_folder, "nmse_distribution.pdf"), bbox_inches="tight"
    )
    plt.close()


def create_experiment_folder(args):
    """Create experiment folder with organized structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_experiments_dir = "cov_results"
    model_dir = os.path.join(base_experiments_dir, args.model_type.lower())
    dimension_dir = os.path.join(model_dir, f"seq{args.seq_len}_var{args.n_variables}")

    folder_name = f"l{args.num_layers}_h{args.num_heads}_{timestamp}"
    full_path = os.path.join(dimension_dir, folder_name)

    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_results_to_file(results, args, train_losses, val_losses, output_folder):
    """Save results to file"""
    filename = f"{args.model_type.lower()}_covariance_results.txt"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{args.model_type} Covariance Learning Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random Seed: {args.seed}\n\n")

        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Sequence Length: {args.seq_len}\n")
        f.write(f"Number of Variables: {args.n_variables}\n")
        f.write(f"Transformer Layers: {args.num_layers}\n")
        f.write(f"Attention Heads: {args.num_heads}\n")
        f.write(f"Hidden Dimension: {args.hidden_dim}\n")
        f.write(f"Dropout Rate: {args.dropout}\n\n")

        # Dataset Configuration
        f.write("DATASET CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Samples: {args.num_train}\n")
        f.write(f"Validation Samples: {args.num_val}\n")
        f.write(f"Test Samples: {args.num_test}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Noise Level: {args.noise_level}\n\n")

        # Training Configuration
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Early Stopping Patience: {args.patience}\n\n")

        # Training Results
        f.write("TRAINING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\n")
        f.write(f"Best Validation Loss: {min(val_losses):.6f}\n\n")

        # Test Results
        f.write("TEST RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Frobenius Loss: {results['frobenius_loss']:.6f}\n")
        f.write(f"R-squared: {results['r_squared']:.6f}\n")
        f.write(f"Frobenius Mean: {results['frobenius_mean']:.6f}\n")
        f.write(f"Frobenius Median: {results['frobenius_median']:.6f}\n")
        f.write(f"Frobenius 95th Percentile: {results['frobenius_95th']:.6f}\n\n")

        f.write(f"All files saved in: {output_folder}/\n")

    print(f"Results saved to: {filepath}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Covariance Learning Experiment",
        add_help=False,  # Disable default -h/--help to save -h for num_heads
    )

    # Help option
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default=MODEL_TYPE,
        choices=[
            "SoftmaxTransformer",
            "LinearTransformer",
            "BasicLinearTransformer",
            "MLP",
        ],
        help="Model type to use",
    )
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument(
        "--n_variables", type=int, default=N_VARIABLES, help="Number of variables"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=HIDDEN_DIM, help="Hidden dimension"
    )
    parser.add_argument(
        "--num_layers",
        "-l",
        type=int,
        default=NUM_LAYERS,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num_heads",
        "-h",
        type=int,
        default=NUM_HEADS,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--dropout", type=float, default=DROPOUT_RATE, help="Dropout rate"
    )

    # Dataset parameters
    parser.add_argument(
        "--num_train",
        type=int,
        default=NUM_TRAIN_SAMPLES,
        help="Number of training samples",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=NUM_VAL_SAMPLES,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=NUM_TEST_SAMPLES,
        help="Number of test samples",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--noise_level", type=float, default=NOISE_LEVEL, help="Noise level"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=PATIENCE, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")

    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Override global parameters with command line arguments
    SEQ_LEN = args.seq_len
    N_VARIABLES = args.n_variables
    NUM_TRAIN_SAMPLES = args.num_train
    NUM_VAL_SAMPLES = args.num_val
    NUM_TEST_SAMPLES = args.num_test
    BATCH_SIZE = args.batch_size
    NOISE_LEVEL = args.noise_level
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    HIDDEN_DIM = args.hidden_dim
    DROPOUT_RATE = args.dropout
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    PATIENCE = args.patience
    RANDOM_SEED = args.seed
    MODEL_TYPE = args.model_type

    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Covariance Learning Experiment")
    print("=" * 60)
    print(f"Using device: {DEVICE}")

    # Create experiment folder
    output_folder = create_experiment_folder(args)
    print(f"\nExperiment folder created: {output_folder}")

    # Redirect output to both terminal and file
    original_stdout = sys.stdout
    output_file = open(os.path.join(output_folder, "console_output.txt"), "w")

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(original_stdout, output_file)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Model Type: {MODEL_TYPE}")
    print(f"  Sequence Length: {SEQ_LEN}")
    print(f"  Number of Variables: {N_VARIABLES}")
    print(
        f"  Dataset: {NUM_TRAIN_SAMPLES} train, {NUM_VAL_SAMPLES} val, {NUM_TEST_SAMPLES} test"
    )
    print(f"  Model: {NUM_LAYERS} layers, {NUM_HEADS} heads, {HIDDEN_DIM} hidden_dim")
    print(
        f"  Training: {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}"
    )
    print(f"  Seed: {RANDOM_SEED}")

    # Generate datasets
    print("\nGenerating training data...")
    train_dataset = CovarianceDataset(
        NUM_TRAIN_SAMPLES, SEQ_LEN, N_VARIABLES, NOISE_LEVEL
    )

    print("Generating validation data...")
    val_dataset = CovarianceDataset(NUM_VAL_SAMPLES, SEQ_LEN, N_VARIABLES, NOISE_LEVEL)

    print("Generating test data...")
    test_dataset = CovarianceDataset(
        NUM_TEST_SAMPLES, SEQ_LEN, N_VARIABLES, NOISE_LEVEL
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print(f"\nCreating {MODEL_TYPE} model...")
    model = create_model(
        MODEL_TYPE,
        SEQ_LEN,
        N_VARIABLES,
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_HEADS,
        DROPOUT_RATE,
    )
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model device: {next(model.parameters()).device}")

    # Training setup
    criterion = covariance_reconstruction_loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.02
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Training loop
    print("\nTraining model...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False
        )
        for batch in train_pbar:
            sequence = batch["sequence"].to(DEVICE)
            true_cov = batch["true_covariance"].to(DEVICE)

            optimizer.zero_grad()
            pred_cov_vector = model(sequence)
            loss = criterion(pred_cov_vector, true_cov, N_VARIABLES)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequence.size(0)
            train_samples += sequence.size(0)

            train_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                sequence = batch["sequence"].to(DEVICE)
                true_cov = batch["true_covariance"].to(DEVICE)

                pred_cov_vector = model(sequence)
                loss = criterion(pred_cov_vector, true_cov, N_VARIABLES)

                val_loss += loss.item() * sequence.size(0)
                val_samples += sequence.size(0)

        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(
                model.state_dict(), os.path.join(output_folder, "best_model.pth")
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}"
            )

    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_folder, "best_model.pth")))

    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, N_VARIABLES)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Frobenius Loss: {results['frobenius_loss']:.6f}")
    print(f"R-squared: {results['r_squared']:.6f}")
    print(f"Frobenius Mean: {results['frobenius_mean']:.6f}")
    print(f"Frobenius Median: {results['frobenius_median']:.6f}")
    print(f"Frobenius 95th Percentile: {results['frobenius_95th']:.6f}")

    # Save plots
    plot_training_curves(train_losses, val_losses, output_folder)
    plot_frobenius_distribution(results["frobenius_errors"], output_folder)
    plot_nmse_distribution(results["all_nmse"], output_folder)

    # Save results
    save_results_to_file(results, args, train_losses, val_losses, output_folder)

    # Restore stdout and close output file
    sys.stdout = original_stdout
    output_file.close()
    print(
        f"\nConsole output saved to: {os.path.join(output_folder, 'console_output.txt')}"
    )

    print("\nTraining completed! Check the saved plots and results file.")
