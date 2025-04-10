# reproduce.py
import torch
import argparse
import numpy as np
import os
import utils

from ddpm import DDPM, NoiseScheduler  

@torch.no_grad()
def deterministic_sample(model, noise_scheduler, x_T):
    device = x_T.device
    x = x_T.clone()

    for t in reversed(range(len(noise_scheduler))):
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)  # predicted noise
        beta_t = noise_scheduler.betas[t].to(device)
        alpha_t = noise_scheduler.alphas[t].to(device)
        alpha_bar_t = noise_scheduler.alpha_bars[t].to(device)

        # Posterior mean
        coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * eps)

        # z = 0 for deterministic sampling
        # so no noise is added, even if t > 0
        x = mean

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pth",
                        help="Path to the trained DDPM model.")
    parser.add_argument("--lbeta", type=float, default=1e-4, help="Lower bound for beta.")
    parser.add_argument("--ubeta", type=float, default=0.02, help="Upper bound for beta.")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of diffusion steps.")
    parser.add_argument("--n_dim", type=int, default=64, help="Dimensionality of the data.")
    parser.add_argument("--prior_samples_path", type=str, default="data/albatross_prior_samples.npy",
                        help="Path to the .npy file containing prior samples.")
    parser.add_argument("--output_path", type=str, default="albatross_samples_reproduce.npy",
                        help="Output file for the final generated samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Ensure reproducibility
    utils.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the trained model
    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Create noise scheduler with the same hyperparams used in training
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)

    # Load the prior samples x_T
    prior_samples = np.load(args.prior_samples_path)  # shape: [32561, n_dim]
    x_T = torch.from_numpy(prior_samples).float().to(device)

    # Run deterministic reverse diffusion
    print(f"Running deterministic reverse diffusion for {x_T.shape[0]} samples...")
    x_0 = deterministic_sample(model, noise_scheduler, x_T)

    # Move to CPU and save
    x_0_np = x_0.cpu().numpy()
    np.save(args.output_path, x_0_np)
    print(f"Saved deterministic samples to {args.output_path}")


if __name__ == "__main__":
    main()
