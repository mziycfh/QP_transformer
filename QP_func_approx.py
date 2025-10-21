from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import random
from tqdm import tqdm
import argparse
import os
import pickle

# =============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =============================================================================

# Default Parameters

# Problem Parameters
PROBLEM_DIM = 5  # variable dimension (n)
CONSTRAINT_NUM = 3  # number of constraints (m)

# Dataset Parameters
NUM_TRAIN_SAMPLES = 50000  # training samples
NUM_VAL_SAMPLES = 10000  # validation samples
NUM_TEST_SAMPLES = 10000  # test samples
BATCH_SIZE = 128

KAPPA_LOW = None
KAPPA_HIGH = None

# Model Architecture Parameters
NUM_LAYERS = 4  # number of transformer layers
NUM_HEADS = 2  # number of attention heads
HIDDEN_DIM = 256  # hidden dimension
DROPOUT_RATE = 0.1  # dropout rate

# Training Parameters
NUM_EPOCHS = 500  # maximum number of epochs
LEARNING_RATE = 1e-4  # learning rate
PATIENCE = 30  # early stopping patience
RANDOM_SEED = 42  # random seed for reproducibility

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Tensor = torch.Tensor
torch.set_printoptions(precision=5, linewidth=140, sci_mode=False)


# ------------------------------------------------------------
#  QP Problem Data Generator
# ------------------------------------------------------------
class QPDataset(Dataset):
    """
    Dataset for generating general QP problems with known solutions.
    Generates: min 0.5 * x^T Q x + c^T x subject to A x <= b
    """

    def __init__(
        self,
        num_samples: int,
        n: int,  # variable dimension
        m: int,  # number of constraints
        seed: int = 42,
        # kappa_low / kappa_high: log-uniform range for column scaling factor κ, which controls the eigenvalue spread (condition number) of Q and thus the hardness of the QP.
        kappa_low: float | None = None,
        kappa_high: float | None = None,
    ):
        self.num_samples = num_samples
        self.n = n  # variable dimension
        self.m = m  # number of constraints
        self.seed = seed
        self.kappa_low = kappa_low
        self.kappa_high = kappa_high

        if kappa_low is None or kappa_high is None:
            dataset_path = f"QP_datasets/n{n}_m{m}_s{num_samples}_seed{seed}.pkl"
        else:
            # Format floats without trailing zeros and dots; keep dot in filename
            low_str = (f"{float(kappa_low):.2f}").rstrip("0").rstrip(".")
            high_str = (f"{float(kappa_high):.2f}").rstrip("0").rstrip(".")
            tag = f"kappa{low_str}-{high_str}"
            dataset_path = f"QP_datasets/{tag}_n{n}_m{m}_s{num_samples}_seed{seed}.pkl"

        if os.path.exists(dataset_path):
            print(f"Loading QP dataset: {dataset_path}")
            self._load_dataset(dataset_path)
        else:
            print(f"Generating QP dataset (will save to: {dataset_path})")
            self._generate_dataset()
            self._save_dataset(dataset_path)

    def _load_dataset(self, path: str):
        """Load QP dataset from file"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.problems = data["problems"]
            self.solutions = data["solutions"]
            self.x_inits = data["x_inits"]

    def _save_dataset(self, path: str):
        """Save QP dataset to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "problems": self.problems,
            "solutions": self.solutions,
            "x_inits": self.x_inits,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"QP dataset saved to: {path}")

    def _generate_dataset(self):
        """Original QP dataset generation logic"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.problems = []
        self.solutions = []
        self.x_inits = []

        for _ in tqdm(range(self.num_samples), desc="Generating QP problems"):
            while True:
                # Generate random QP problem
                if self.kappa_low is None or self.kappa_high is None:
                    Q = torch.randn(self.n, self.n)
                    Q = Q @ Q.T + 0.1 * torch.eye(
                        self.n, device=Q.device, dtype=Q.dtype
                    )
                else:
                    Q = torch.randn(self.n, self.n)
                    fro_base = torch.linalg.norm(Q @ Q.T, ord="fro")
                    logk = (
                        torch.empty(1)
                        .uniform_(
                            float(np.log(self.kappa_low)),
                            float(np.log(self.kappa_high)),
                        )
                        .item()
                    )
                    kappa = float(np.exp(logk))
                    t = torch.linspace(
                        0.0, 1.0, steps=self.n, device=Q.device, dtype=Q.dtype
                    )
                    scales = torch.exp(t * np.log(kappa))
                    Q = Q * scales
                    fro_scaled = torch.linalg.norm(Q @ Q.T, ord="fro")
                    alpha = fro_base / (fro_scaled + 1e-12)
                    Q = alpha * (Q @ Q.T) + 0.1 * torch.eye(
                        self.n, device=Q.device, dtype=Q.dtype
                    )

                c = torch.randn(self.n)
                A = torch.randn(self.m, self.n)  # m constraints, n variables
                b = torch.rand(self.m) + 1.0  # Ensure feasible region exists

                try:
                    # Solve exactly with CVXPY
                    x_opt = self._solve_general_qp(Q, c, A, b)

                    # Find a feasible initial point
                    x_init = self._find_feasible_point(A, b)

                    break  # Solution found, exit retry loop
                except RuntimeError:
                    continue  # No solution, regenerate constraints

            self.problems.append((Q, c, A, b))
            self.solutions.append(x_opt)
            self.x_inits.append(x_init)

    def _solve_general_qp(self, Q: Tensor, c: Tensor, A: Tensor, b: Tensor) -> Tensor:
        """Solve general QP exactly"""
        import os
        import sys

        Q_np, c_np, A_np, b_np = map(lambda t: t.detach().cpu().numpy(), (Q, c, A, b))
        n = Q_np.shape[0]
        x = cp.Variable(n)
        obj = 0.5 * cp.quad_form(x, Q_np) + c_np @ x
        constraints = [A_np @ x <= b_np]
        prob = cp.Problem(cp.Minimize(obj), constraints)

        # Suppress OSQP output
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                prob.solve(solver="OSQP", verbose=False)
            finally:
                sys.stdout = old_stdout

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY status: {prob.status}")
        return torch.tensor(x.value, dtype=Q.dtype)

    def _find_feasible_point(self, A: Tensor, b: Tensor) -> Tensor:
        """Find a feasible point for the constraint set A x <= b by random sampling"""
        A_np = A.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()

        # Try up to 50000 random points to find a feasible one
        max_attempts = 50000

        for _ in range(max_attempts):
            # Generate random point in a reasonable range
            x_candidate = np.random.uniform(-5.0, 5.0, self.n)

            # Check feasibility: A @ x <= b
            violations = A_np @ x_candidate - b_np
            if np.all(violations <= 1e-8):  # All constraints satisfied
                return torch.tensor(x_candidate, dtype=torch.float32)

        # If no feasible point found after max_attempts, raise exception
        raise RuntimeError(
            f"Could not find feasible point after {max_attempts} attempts"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        Q, c, A, b = self.problems[idx]
        x_opt = self.solutions[idx]
        x_init = self.x_inits[idx]  # Use pre-computed feasible point

        # Create token sequence: [Q_rows, c, A_rows, b_padded, x_init]
        # x_init = torch.zeros(self.n)  # Start from zero

        # Pad b vector to match dimension n (from m to n)
        b_padded = torch.zeros(self.n)
        b_padded[: self.m] = b  # Fill first m elements with actual b values

        tokens = torch.vstack([Q, c, A, b_padded, x_init])

        return {
            "tokens": tokens,
            "Q": Q,
            "c": c,
            "A": A,
            "b": b,
            "solution": x_opt,
            "problem_id": idx,
        }


# ------------------------------------------------------------
#  Trainable Transformer for QP Solving
# ------------------------------------------------------------
class QPTransformer(nn.Module):
    """
    Pure Transformer for solving general QP problems.
    Architecture: Standard Encoder-Only Transformer.
    """

    def __init__(
        self,
        n: int,  # variable dimension
        m: int,  # number of constraints
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.hidden_dim = hidden_dim

        # Calculate token sequence length: n(Q rows) + 1(c) + m(A rows) + 1(b_padded) + 1(x_init) = n + m + 3
        self.seq_len = n + m + 3

        # Input embedding
        self.token_embedding = nn.Linear(n, hidden_dim)
        self.position_embedding = nn.Embedding(self.seq_len, hidden_dim)

        # Layer normalization for embeddings
        self.embedding_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n),  # Output n-dimensional solution vector
        )

        # Move entire model to GPU after all components are created
        self.to(DEVICE)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Forward pass: direct solution prediction
        Args:
            tokens: (batch_size, seq_len, n) token sequence
        Returns:
            solution: (batch_size, n) predicted solution
        """
        batch_size = tokens.size(0)

        # Embed tokens
        embedded = self.token_embedding(tokens)  # (batch, seq_len, hidden_dim)

        # Add positional encoding
        positions = torch.arange(tokens.size(1), device=tokens.device)
        pos_emb = self.position_embedding(positions)
        embedded = embedded + pos_emb.unsqueeze(0)

        # Layer normalization
        embedded = self.embedding_norm(embedded)

        # Transformer processing
        x = self.transformer(embedded)  # (batch, seq_len, hidden_dim)

        # Extract solution token (last token)
        solution_token = x[:, -1, :]  # (batch, hidden_dim)

        # Output projection
        solution = self.output_projection(solution_token)  # (batch, n)

        return solution


class LinearQPTransformer(nn.Module):
    """
    Linear Transformer for QP problem solving.
    Uses linear attention instead of standard softmax attention.
    """

    def __init__(
        self,
        n: int,  # variable dimension
        m: int,  # number of constraints
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n  # variable dimension
        self.m = m  # number of constraints
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Calculate token sequence length: n(Q rows) + 1(c) + m(A rows) + 1(b_padded) + 1(x_init) = n + m + 3
        self.seq_len = n + m + 3

        # Input embedding
        self.token_embedding = nn.Linear(n, hidden_dim)
        self.position_embedding = nn.Embedding(self.seq_len, hidden_dim)

        # Layer normalization for embeddings
        self.embedding_norm = nn.LayerNorm(hidden_dim)

        # Linear attention layers
        self.layers = nn.ModuleList(
            [
                LinearAttentionLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n),  # Output n-dimensional solution vector
        )

        # Move entire model to GPU after all components are created
        self.to(DEVICE)

    def forward(self, tokens):
        batch_size = tokens.shape[0]

        # Create positional indices
        positions = (
            torch.arange(self.seq_len, device=DEVICE)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embed tokens and add positional encoding
        embedded = self.token_embedding(tokens) + self.position_embedding(positions)
        embedded = self.embedding_norm(embedded)

        # Apply linear attention layers
        x = embedded
        for layer in self.layers:
            x = layer(x)

        # Extract solution token (last token)
        solution_token = x[:, -1, :]  # [batch_size, hidden_dim]

        # Project to solution space
        return self.output_projection(solution_token)


class LinearAttentionLayer(nn.Module):
    """Linear attention layer implementation"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

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


class MLP(nn.Module):
    def __init__(self, n: int, m: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n = n
        self.m = m

        seq_len = n + m + 3
        input_dim = seq_len * n

        hidden_dims = [
            hidden_dim * 6,
            hidden_dim * 6,
            hidden_dim * 5,
            hidden_dim * 5,
            hidden_dim * 4,
            hidden_dim * 4,
        ]

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.projections = nn.ModuleList()

        in_dim = input_dim
        for out_dim in hidden_dims:
            self.layer_norms.append(nn.LayerNorm(in_dim))
            self.activations.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(in_dim, out_dim))
            if in_dim == out_dim:
                self.projections.append(nn.Identity())
            else:
                self.projections.append(nn.Linear(in_dim, out_dim, bias=False))
            in_dim = out_dim

        self.output = nn.Linear(hidden_dims[-1], n)

    def forward(self, tokens):
        batch_size = tokens.size(0)
        x = tokens.reshape(batch_size, -1)

        for ln, act, do, layer, proj in zip(
            self.layer_norms,
            self.activations,
            self.dropouts,
            self.layers,
            self.projections,
        ):
            residual = x
            x = ln(x)
            x = layer(x)
            x = act(x)
            x = do(x)
            x = proj(residual) + x

        return self.output(x)


# ------------------------------------------------------------
#  Training and Evaluation Functions
# ------------------------------------------------------------
def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    patience: int = PATIENCE,
) -> tuple[List[float], List[float]]:
    """Train the QP Transformer model with validation and early stopping"""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.02
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss_so_far = float("inf")
    patience_counter = 0
    best_model_state = None

    epoch_pbar = tqdm(range(num_epochs), desc="Training", leave=True)

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        total_train_samples = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False
        )
        for batch in train_pbar:
            tokens = batch["tokens"].to(DEVICE)
            solutions = batch["solution"].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(tokens)

            # Compute loss
            loss = criterion(predictions, solutions)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * predictions.size(0)
            total_train_samples += predictions.size(0)

            # Update training progress bar
            train_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        epoch_train_loss_avg = epoch_train_loss / total_train_samples
        train_losses.append(epoch_train_loss_avg)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        total_val_samples = 0

        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False
        )
        with torch.no_grad():
            for batch in val_pbar:
                tokens = batch["tokens"].to(DEVICE)
                solutions = batch["solution"].to(DEVICE)

                predictions = model(tokens)
                loss = criterion(predictions, solutions)

                epoch_val_loss += loss.item() * predictions.size(0)
                total_val_samples += predictions.size(0)

                # Update validation progress bar
                val_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        epoch_val_loss_avg = epoch_val_loss / total_val_samples
        val_losses.append(epoch_val_loss_avg)

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss_avg)

        # Early stopping check
        if epoch_val_loss_avg < best_val_loss_so_far:
            best_val_loss_so_far = epoch_val_loss_avg
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{epoch_train_loss_avg:.6f}",
                    "Val Loss": f"{epoch_val_loss_avg:.6f}",
                    "Status": "Best Model Saved",
                }
            )
        else:
            patience_counter += 1
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{epoch_train_loss_avg:.6f}",
                    "Val Loss": f"{epoch_val_loss_avg:.6f}",
                    "Patience": f"{patience_counter}/{patience}",
                }
            )

        # Early stopping
        if patience_counter >= patience:
            epoch_pbar.set_description(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


def evaluate_model(model, dataloader: DataLoader) -> dict:
    """Evaluate model performance and collect data for visualization"""
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0
    objective_errors = []
    constraint_violations = []

    # R-squared calculation variables
    ss_res = 0.0  # Sum of squared residuals
    ss_tot = 0.0  # Total sum of squares
    y_sum = None  # Sum of all y_i for mean calculation

    # Data for visualization
    all_predictions = []
    all_solutions = []
    all_objective_errors = []
    all_constraint_violations = []
    all_nmse = []
    all_Q = []
    all_c = []
    all_A = []
    all_b = []

    with torch.no_grad():
        eval_pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in eval_pbar:
            tokens = batch["tokens"].to(DEVICE)
            solutions = batch["solution"].to(DEVICE)
            Q = batch["Q"].to(DEVICE)
            c = batch["c"].to(DEVICE)
            A = batch["A"].to(DEVICE)
            b = batch["b"].to(DEVICE)

            predictions = model(tokens)

            # MSE loss - accumulate total squared error and sample count
            batch_loss = criterion(predictions, solutions)
            total_loss += batch_loss.item() * predictions.size(
                0
            )  # Multiply by batch size
            total_samples += predictions.size(0)

            # R-squared calculation - accumulate squared errors
            # SS_res = sum_i ||y_i - y_hat_i||^2
            residuals = predictions - solutions
            ss_res += torch.sum(residuals**2).item()

            # Accumulate y_i for mean calculation
            if y_sum is None:
                y_sum = torch.sum(solutions, dim=0)
            else:
                y_sum += torch.sum(solutions, dim=0)

            # Process each sample in the batch
            for i in range(predictions.size(0)):
                pred = predictions[i].cpu().numpy()
                sol = solutions[i].cpu().numpy()
                Q_i = Q[i].cpu().numpy()
                c_i = c[i].cpu().numpy()
                A_i = A[i].cpu().numpy()
                b_i = b[i].cpu().numpy()

                # Objective function error
                pred_obj = 0.5 * pred.T @ Q_i @ pred + c_i.T @ pred
                true_obj = 0.5 * sol.T @ Q_i @ sol + c_i.T @ sol
                obj_error = abs(pred_obj - true_obj)
                objective_errors.append(obj_error)

                # Constraint violation (A x <= b)
                Ax = A_i @ pred  # A x
                violations = np.maximum(0, Ax - b_i)  # max(0, A x - b)
                constraint_violation_mean = np.mean(violations)
                constraint_violations.append(constraint_violation_mean)

                # Per-sample NMSE: ||y_i - y_hat_i||^2 / ||y_i||^2
                residual_norm_squared = np.sum((sol - pred) ** 2)
                y_norm_squared = np.sum(sol**2)
                nmse = (
                    residual_norm_squared / y_norm_squared
                    if y_norm_squared > 0
                    else 0.0
                )

                # Store for visualization
                all_predictions.append(pred)
                all_solutions.append(sol)
                all_objective_errors.append(obj_error)
                all_constraint_violations.append(constraint_violation_mean)
                all_nmse.append(nmse)
                all_Q.append(Q_i)
                all_c.append(c_i)
                all_A.append(A_i)
                all_b.append(b_i)

            # Update evaluation progress bar
            eval_pbar.set_postfix({"Loss": f"{batch_loss.item():.6f}"})

    # Calculate R-squared
    # Calculate mean of all y_i
    y_mean = y_sum / total_samples  # Shape: (n_vars,)

    # Calculate SS_tot = sum_i ||y_i - y_mean||^2
    ss_tot = 0.0
    for sol in all_solutions:
        sol_tensor = torch.tensor(sol, device=DEVICE)
        ss_tot += torch.sum((sol_tensor - y_mean) ** 2).item()

    # Calculate R-squared: R^2 = 1 - SS_res / SS_tot
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse_loss": total_loss / total_samples,
        "r_squared": r_squared,
        "avg_objective_error": np.mean(objective_errors),
        "avg_constraint_violation": np.mean(constraint_violations),
        "all_predictions": np.array(all_predictions),
        "all_solutions": np.array(all_solutions),
        "all_objective_errors": np.array(all_objective_errors),
        "all_constraint_violations": np.array(all_constraint_violations),
        "all_nmse": np.array(all_nmse),
        "all_Q": all_Q,
        "all_c": all_c,
        "all_A": all_A,
        "all_b": all_b,
    }


def visualize_test_results(evaluation_data: dict, output_folder: str):
    """Generate detailed visualization of test results using pre-computed data"""
    # Extract data from evaluation results
    all_predictions = evaluation_data["all_predictions"]
    all_solutions = evaluation_data["all_solutions"]
    all_objective_errors = evaluation_data["all_objective_errors"]
    all_constraint_violations = evaluation_data["all_constraint_violations"]
    all_nmse = evaluation_data["all_nmse"]

    # 1. NMSE Distribution Histogram
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
    plt.legend(prop={"weight": "bold"}, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_folder, "nmse_distribution.pdf"), bbox_inches="tight"
    )
    plt.show()

    # 3. Objective Error Distribution Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(
        all_objective_errors, bins=30, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    plt.axvline(
        np.mean(all_objective_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_objective_errors):.4f}",
    )
    plt.axvline(
        np.median(all_objective_errors),
        color="blue",
        linestyle="--",
        label=f"Median: {np.median(all_objective_errors):.4f}",
    )
    plt.axvline(
        np.percentile(all_objective_errors, 95),
        color="orange",
        linestyle=":",
        label=f"95th percentile: {np.percentile(all_objective_errors, 95):.4f}",
    )
    plt.xlabel("Objective Function Error", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.title("Distribution of Objective Function Errors", fontweight="bold")
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.legend(prop={"weight": "bold"}, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_folder, "objective_error_distribution.pdf"),
        bbox_inches="tight",
    )
    plt.show()

    # 3. Constraint Violation Distribution Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(
        all_constraint_violations,
        bins=30,
        alpha=0.7,
        color="lightsteelblue",
        edgecolor="black",
    )
    plt.axvline(
        np.mean(all_constraint_violations),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_constraint_violations):.4f}",
    )
    plt.axvline(
        np.median(all_constraint_violations),
        color="blue",
        linestyle="--",
        label=f"Median: {np.median(all_constraint_violations):.4f}",
    )
    plt.axvline(
        np.percentile(all_constraint_violations, 95),
        color="orange",
        linestyle=":",
        label=f"95th percentile: {np.percentile(all_constraint_violations, 95):.4f}",
    )
    plt.xlabel("Max Constraint Violation", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.title("Distribution of Constraint Violations", fontweight="bold")
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.legend(prop={"weight": "bold"}, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_folder, "constraint_violation_distribution.pdf"),
        bbox_inches="tight",
    )
    plt.show()

    print(f"\nSaved plots to {output_folder}:")
    print(f"  - nmse_distribution.pdf")
    print(f"  - objective_error_distribution.pdf")
    print(f"  - constraint_violation_distribution.pdf")

    # Return data for saving to file
    return {
        "total_samples": len(all_predictions),
        "objective_error_range": [
            np.min(all_objective_errors),
            np.max(all_objective_errors),
        ],
        "objective_error_mean": np.mean(all_objective_errors),
        "objective_error_median": np.median(all_objective_errors),
        "objective_error_95th_percentile": np.percentile(all_objective_errors, 95),
        "constraint_violation_range": [
            np.min(all_constraint_violations),
            np.max(all_constraint_violations),
        ],
        "constraint_violation_mean": np.mean(all_constraint_violations),
        "constraint_violation_median": np.median(all_constraint_violations),
        "constraint_violation_95th_percentile": np.percentile(
            all_constraint_violations, 95
        ),
        "zero_violation_samples": np.sum(all_constraint_violations < 1e-6),
        "total_violation_samples": len(all_constraint_violations),
    }


def create_experiment_folder(args):
    """Create experiment folder with organized structure"""
    import os
    from datetime import datetime

    # Create organized folder structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_experiments_dir = "experiments"
    model_dir = os.path.join(base_experiments_dir, args.model_type.lower())
    dimension_dir = os.path.join(
        model_dir, f"n{args.problem_dim}_m{args.constraint_num}"
    )

    # Create final experiment folder name; include kappa tag if provided (test-only)
    if (
        getattr(args, "kappa_low", None) is not None
        and getattr(args, "kappa_high", None) is not None
    ):
        low_str = (f"{float(args.kappa_low):.2f}").rstrip("0").rstrip(".")
        high_str = (f"{float(args.kappa_high):.2f}").rstrip("0").rstrip(".")
        folder_name = f"l{args.num_layers}_h{args.num_heads}_kappa{low_str}-{high_str}_{timestamp}"
    else:
        folder_name = f"l{args.num_layers}_h{args.num_heads}_{timestamp}"
    full_path = os.path.join(dimension_dir, folder_name)

    # Create all necessary directories
    os.makedirs(full_path, exist_ok=True)

    return full_path


def save_results_to_file(
    results, detailed_results, args, train_losses, val_losses, output_folder
):
    """Save experimental results to a text file"""
    from datetime import datetime

    filename = f"{args.model_type.lower()}_results.txt"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{args.model_type} Experimental Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random Seed: {args.seed}\n\n")

        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Problem Dimension (n): {args.problem_dim}\n")
        f.write(f"Constraint Number (m): {args.constraint_num}\n")
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
        f.write(f"Test Kappa range: [{args.kappa_low}, {args.kappa_high}]\n\n")

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
        f.write(f"Best Validation Loss: {min(val_losses):.6f}\n")
        f.write(f"Training Epochs Completed: {len(train_losses)}\n\n")

        # Test Results
        f.write("TEST RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Test Samples: {detailed_results['total_samples']}\n")
        f.write(f"MSE Loss: {results['mse_loss']:.6f}\n")
        f.write(f"R-squared: {results['r_squared']:.6f}\n")
        f.write(f"Objective Error: {results['avg_objective_error']:.6f}\n")
        f.write(f"Constraint Violation: {results['avg_constraint_violation']:.6f}\n")
        f.write(
            f"Objective Error Range: [{detailed_results['objective_error_range'][0]:.4f}, {detailed_results['objective_error_range'][1]:.4f}]\n"
        )
        f.write(
            f"Objective Error Median: {detailed_results['objective_error_median']:.4f}\n"
        )
        f.write(
            f"Objective Error 95th Percentile: {detailed_results['objective_error_95th_percentile']:.4f}\n"
        )
        f.write(
            f"Constraint Violation Range: [{detailed_results['constraint_violation_range'][0]:.4f}, {detailed_results['constraint_violation_range'][1]:.4f}]\n"
        )
        f.write(
            f"Constraint Violation Median: {detailed_results['constraint_violation_median']:.4f}\n"
        )
        f.write(
            f"Constraint Violation 95th Percentile: {detailed_results['constraint_violation_95th_percentile']:.4f}\n"
        )
        f.write(
            f"Samples with Zero Constraint Violation: {detailed_results['zero_violation_samples']}/{detailed_results['total_violation_samples']}\n\n"
        )

        # File Information
        f.write("GENERATED FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"All files saved in: {output_folder}/\n")
        f.write("Plots:\n")
        f.write("  - training_curves.pdf\n")
        f.write("  - nmse_distribution.pdf\n")
        f.write("  - objective_error_distribution.pdf\n")
        f.write("  - constraint_violation_distribution.pdf\n")
        f.write(f"Results: {filepath}\n")

    return filepath


def plot_training_curves(train_losses, val_losses, output_folder):
    """Plot and save training curves"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, "b-", linewidth=2, label="Training Loss")
    plt.plot(val_losses, "r-", linewidth=2, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    # Save plot to file
    plt.savefig(os.path.join(output_folder, "training_curves.pdf"), bbox_inches="tight")
    print(f"\nTraining curves saved to {output_folder}:")
    print(f"  - training_curves.pdf")
    plt.show()


def create_model(model_type, args):
    """Create model based on model type"""
    if model_type == "SoftmaxTransformer":
        return QPTransformer(
            n=args.problem_dim,
            m=args.constraint_num,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    elif model_type == "LinearTransformer":
        return LinearQPTransformer(
            n=args.problem_dim,
            m=args.constraint_num,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    elif model_type == "MLP":
        return MLP(
            n=args.problem_dim,
            m=args.constraint_num,
            hidden_dim=args.hidden_dim,  # Single hidden_dim parameter
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Trainable Transformer for QP Problem Solving",
        add_help=False,  # Disable default -h/--help to save -h for num_heads
    )

    # Help option
    parser.add_argument(
        "--help",
        "-help",
        action="help",
        help="Nah we ain't gonna help you with argparse!",
    )

    # Problem Parameters
    parser.add_argument(
        "--problem_dim",
        "-n",
        type=int,
        default=PROBLEM_DIM,
        help=f"Variable dimension (n) (default: {PROBLEM_DIM})",
    )
    parser.add_argument(
        "--constraint_num",
        "-m",
        type=int,
        default=CONSTRAINT_NUM,
        help=f"Number of constraints (m) (default: {CONSTRAINT_NUM})",
    )

    # Dataset Parameters
    parser.add_argument(
        "--num_train",
        type=int,
        default=NUM_TRAIN_SAMPLES,
        help=f"Number of training samples (default: {NUM_TRAIN_SAMPLES})",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=NUM_VAL_SAMPLES,
        help=f"Number of validation samples (default: {NUM_VAL_SAMPLES})",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=NUM_TEST_SAMPLES,
        help=f"Number of test samples (default: {NUM_TEST_SAMPLES})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument("--kappa_low", type=float, default=KAPPA_LOW)
    parser.add_argument("--kappa_high", type=float, default=KAPPA_HIGH)

    # Model Architecture Parameters
    parser.add_argument(
        "--num_layers",
        "-l",
        type=int,
        default=NUM_LAYERS,
        help=f"Number of transformer layers (default: {NUM_LAYERS})",
    )
    parser.add_argument(
        "--num_heads",
        "-h",
        type=int,
        default=NUM_HEADS,
        help=f"Number of attention heads (default: {NUM_HEADS})",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=HIDDEN_DIM,
        help=f"Hidden dimension (default: {HIDDEN_DIM})",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT_RATE,
        help=f"Dropout rate (default: {DROPOUT_RATE})",
    )

    # Training Parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Maximum number of epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Early stopping patience (default: {PATIENCE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )

    # Model Type
    parser.add_argument(
        "--model_type",
        type=str,
        default="SoftmaxTransformer",
        choices=[
            "SoftmaxTransformer",
            "LinearTransformer",
            "MLP",
        ],
        help="Type of model to use (default: SoftmaxTransformer)",
    )

    return parser.parse_args()


# ------------------------------------------------------------
#  Main Training and Comparison Script
# ------------------------------------------------------------
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Override global parameters with command line arguments
    PROBLEM_DIM = args.problem_dim
    CONSTRAINT_NUM = args.constraint_num
    NUM_TRAIN_SAMPLES = args.num_train
    NUM_VAL_SAMPLES = args.num_val
    NUM_TEST_SAMPLES = args.num_test
    BATCH_SIZE = args.batch_size
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    HIDDEN_DIM = args.hidden_dim
    DROPOUT_RATE = args.dropout
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    PATIENCE = args.patience
    RANDOM_SEED = args.seed
    MODEL_TYPE = args.model_type
    KAPPA_LOW = args.kappa_low
    KAPPA_HIGH = args.kappa_high
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Trainable Transformer for General QP Problem Solving")
    print("=" * 60)
    print(f"Using device: {DEVICE}")

    # Create experiment folder
    output_folder = create_experiment_folder(args)
    print(f"\nExperiment folder created: {output_folder}")

    # Redirect output to both terminal and file
    import sys

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
    print(f"  Problem: {PROBLEM_DIM} variables, {CONSTRAINT_NUM} constraints")
    print(
        f"  Dataset: {NUM_TRAIN_SAMPLES} train, {NUM_VAL_SAMPLES} val, {NUM_TEST_SAMPLES} test"
    )
    print(f"  Test Kappa range: [{KAPPA_LOW}, {KAPPA_HIGH}]")
    print(f"  Model: {NUM_LAYERS} layers, {NUM_HEADS} heads, {HIDDEN_DIM} hidden_dim")
    print(
        f"  Training: {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}"
    )
    print(f"  Seed: {RANDOM_SEED}")

    # Create datasets
    print("Generating training data...")
    train_dataset = QPDataset(
        NUM_TRAIN_SAMPLES,
        PROBLEM_DIM,
        CONSTRAINT_NUM,
        seed=RANDOM_SEED,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    print("Generating validation data...")
    val_dataset = QPDataset(
        NUM_VAL_SAMPLES,
        PROBLEM_DIM,
        CONSTRAINT_NUM,
        seed=RANDOM_SEED + 1,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Generating test data...")
    test_dataset = QPDataset(
        NUM_TEST_SAMPLES,
        PROBLEM_DIM,
        CONSTRAINT_NUM,
        seed=RANDOM_SEED + 2,
        kappa_low=KAPPA_LOW,
        kappa_high=KAPPA_HIGH,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print(f"Creating {MODEL_TYPE} model...")
    model = create_model(MODEL_TYPE, args)
    model = model.to(DEVICE)  # Move model to the same device as data

    # Display model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model device: {next(model.parameters()).device}")

    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE
    )

    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader)

    # Generate detailed test results visualization
    print("\nGenerating detailed test results visualization...")
    detailed_results = visualize_test_results(results, output_folder)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"MSE Loss: {results['mse_loss']:.6f}")
    print(f"R-squared: {results['r_squared']:.6f}")
    print(f"Average Objective Error: {results['avg_objective_error']:.6f}")
    print(f"Average Constraint Violation: {results['avg_constraint_violation']:.6f}")

    # Plot training curves
    plot_training_curves(train_losses, val_losses, output_folder)

    # Save results to file
    print("\nSaving results to file...")
    results_filename = save_results_to_file(
        results, detailed_results, args, train_losses, val_losses, output_folder
    )
    print(f"Results saved to: {results_filename}")

    print("\nTraining completed! Check the saved plots and results file.")

    # Restore stdout and close output file
    sys.stdout = original_stdout
    output_file.close()
    print(
        f"Console output saved to: {os.path.join(output_folder, 'console_output.txt')}"
    )
