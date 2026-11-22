import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularPreprocessor:
    """
    Transforms mixed tabular data into a normalized continuous representation suitable
    for continuous diffusion models, and inverts generated vectors back to original schema.
    """
    def __init__(self, continuous_cols, categorical_cols):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.min_vals = {}
        self.max_vals = {}
        self.cat_categories = {}
        self.feature_slices = {}
        self.total_dim = 0

    def fit(self, df: pd.DataFrame):
        current_idx = 0
        
        # Continuous columns: normalized to [-1, 1]
        for col in self.continuous_cols:
            cmin = float(df[col].min())
            cmax = float(df[col].max())
            if cmax == cmin:
                cmax += 1e-5
            self.min_vals[col] = cmin
            self.max_vals[col] = cmax
            self.feature_slices[col] = ('cont', current_idx, current_idx + 1)
            current_idx += 1

        # Categorical columns: One-Hot encoded
        for col in self.categorical_cols:
            cats = sorted(df[col].unique().tolist())
            self.cat_categories[col] = cats
            n_cats = len(cats)
            self.feature_slices[col] = ('cat', current_idx, current_idx + n_cats, cats)
            current_idx += n_cats

        self.total_dim = current_idx
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col in self.continuous_cols:
            cmin, cmax = self.min_vals[col], self.max_vals[col]
            # scale to [-1, 1]
            scaled = 2.0 * (df[col].values - cmin) / (cmax - cmin) - 1.0
            arrays.append(scaled.reshape(-1, 1))

        for col in self.categorical_cols:
            cats = self.cat_categories[col]
            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            val_to_idx = {c: i for i, c in enumerate(cats)}
            for i, val in enumerate(df[col]):
                if val in val_to_idx:
                    one_hot[i, val_to_idx[val]] = 1.0
                else:
                    one_hot[i, 0] = 1.0
            # Center one-hot roughly around [-0.5, 0.5] or scale
            arrays.append(one_hot)

        return np.hstack(arrays).astype(np.float32)

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        data = {}
        n_samples = arr.shape[0]

        for col in self.continuous_cols:
            _, start, end = self.feature_slices[col]
            col_data = arr[:, start:end].flatten()
            col_data = np.clip(col_data, -1.0, 1.0)
            cmin, cmax = self.min_vals[col], self.max_vals[col]
            unscaled = 0.5 * (col_data + 1.0) * (cmax - cmin) + cmin
            # Cast integer-like features
            if col in ['tenure_months', 'discount_applied', 'app_logins_per_month', 'cs_tickets_raised', 'delivery_delays_count', 'churned']:
                unscaled = np.round(unscaled).astype(int)
                if col in ['discount_applied', 'churned']:
                    unscaled = np.clip(unscaled, 0, 1)
            data[col] = unscaled

        for col in self.categorical_cols:
            _, start, end, cats = self.feature_slices[col]
            sub_arr = arr[:, start:end]
            argmax_indices = np.argmax(sub_arr, axis=1)
            data[col] = [cats[idx] for idx in argmax_indices]

        # Add customer ID
        data['customer_id'] = [f"SYNTH-{10000 + i}" for i in range(n_samples)]
        
        df_out = pd.DataFrame(data)
        # Reorder columns neatly
        cols_order = ['customer_id'] + self.categorical_cols + self.continuous_cols
        return df_out[cols_order]


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TabularDiffusionMLP(nn.Module):
    """
    Denoising MLP network with residual connections and sinusoidal time embeddings.
    Predicts the noise added to tabular features at timestep t.
    """
    def __init__(self, data_dim: int, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.in_proj = nn.Linear(data_dim, hidden_dim)
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.05),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.05),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_emb(t)
        h = self.in_proj(x) + t_embed
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_proj(h)


class TabularDDPM:
    """
    Denoising Diffusion Probabilistic Model manager for Tabular Data.
    Implements standard forward noise schedule and reverse sampling.
    """
    def __init__(self, model: nn.Module, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = 'cpu'):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def forward_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_noise(x_0, t, noise)
        pred_noise = self.model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n_samples: int, data_dim: int, temperature: float = 1.0) -> np.ndarray:
        self.model.eval()
        x = torch.randn(n_samples, data_dim, device=self.device) * temperature

        for i in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t)
            
            beta_t = self.betas[i]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]
            
            # Equation: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_pred)
            mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise)
            
            if i > 0:
                noise = torch.randn_like(x) * temperature
                sigma = torch.sqrt(self.posterior_variance[i])
                x = mean + sigma * noise
            else:
                x = mean

        return x.cpu().numpy()
