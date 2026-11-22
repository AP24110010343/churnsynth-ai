import os
import pickle
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_generator import generate_baseline_subscription_data
from diffusion_model import TabularPreprocessor, TabularDiffusionMLP, TabularDDPM

def main():
    print("=" * 60)
    print("ChurnSynth AI: Training Tabular DDPM Diffusion Model")
    print("=" * 60)

    # 1. Generate / Load Data
    data_path = "baseline_subscription_data.csv"
    if not os.path.exists(data_path):
        print("Generating baseline retail subscription data...")
        df = generate_baseline_subscription_data(n_samples=4000, random_seed=42)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} records. Churn rate: {df['churned'].mean():.2%}")

    continuous_cols = [
        'tenure_months', 'monthly_fee', 'discount_applied', 
        'app_logins_per_month', 'skipped_boxes_ratio', 
        'cs_tickets_raised', 'delivery_delays_count', 
        'engagement_score', 'churned'
    ]
    categorical_cols = ['subscription_tier', 'box_category']

    # 2. Preprocess
    preprocessor = TabularPreprocessor(continuous_cols, categorical_cols)
    preprocessor.fit(df)
    X = preprocessor.transform(df)
    data_dim = preprocessor.total_dim
    print(f"Preprocessed feature vector dimension: {data_dim}")

    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # 3. Model & DDPM Setup
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using computing device: {device}")

    model = TabularDiffusionMLP(data_dim=data_dim, hidden_dim=128, time_dim=64)
    ddpm = TabularDDPM(model, timesteps=100, device=device)

    # DataLoader
    tensor_x = torch.tensor(X, dtype=torch.float32, device=device)
    dataset = TensorDataset(tensor_x)
    batch_size = 256
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    # 4. Training Loop
    epochs = 300
    start_time = time.time()
    print(f"Starting diffusion training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            loss = ddpm.compute_loss(batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        scheduler.step()
        avg_loss = total_loss / len(dataset)

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | MSE Loss: {avg_loss:.5f} | Elapsed: {elapsed:.1f}s")

    # 5. Save Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'data_dim': data_dim,
        'timesteps': 100,
        'continuous_cols': continuous_cols,
        'categorical_cols': categorical_cols
    }
    torch.save(checkpoint, 'churn_diffusion_model.pt')
    print("\nSaved trained model weights to 'churn_diffusion_model.pt'!")

    # 6. Quick Verification Sample
    print("\nTesting reverse diffusion sampling (generating 500 synthetic customers)...")
    sample_vectors = ddpm.sample(n_samples=500, data_dim=data_dim, temperature=0.95)
    df_synthetic = preprocessor.inverse_transform(sample_vectors)
    print("Sample generation successful!")
    print(f"Synthetic Churn Rate: {df_synthetic['churned'].mean():.2%}")
    print("\nPreview of Synthetic Customers:")
    print(df_synthetic.head())

if __name__ == '__main__':
    main()
