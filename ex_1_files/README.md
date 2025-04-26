# VAE Experiments for MNIST

This project implements both Amortized VAE and Latent Optimization approaches for the MNIST dataset, along with various experiments and evaluations.

## File Structure

- `model.py`: Contains the ConvVAE architecture with encoder and decoder networks
- `vae_utils.py`: Utility functions for loss computation, visualization, and evaluation
- `amortized_vae.py`: Implementation of the standard VAE with amortized inference
- `latent_optimization.py`: Implementation of VAE with direct latent optimization
- `main_vae.py`: Main script to run all experiments
- `sup_classification.py`: Supervised classification baseline (Part 1)

## Running the Experiments

1. First ensure you have all required dependencies:
```bash
pip install torch torchvision tqdm matplotlib numpy sklearn
```

2. Run the full experiment suite:
```bash
python main_vae.py
```

This will:
- Train the amortized VAE for 30 epochs
- Train the latent optimization VAE for 30 epochs
- Compare reconstructions from both approaches
- Compute log probabilities for different digits

## Output Structure

Results are saved in two directories:

- `checkpoints/`: Model checkpoints at epochs 1, 5, 10, 20, and 30
  - `vae_epoch_{N}.pt`: Amortized VAE checkpoints
  - `latent_opt_epoch_{N}.pt`: Latent optimization checkpoints

- `results/`: Generated outputs and visualizations
  - `recon_epoch_{N}.png`: Amortized VAE reconstructions
  - `latent_opt_recon_epoch_{N}.png`: Latent optimization reconstructions
  - `samples_epoch_{N}.png`: Samples from prior
  - `comparison_epoch_{N}.png`: Side-by-side comparison
  - `training_curves.png`: Loss curves during training
  - `log_probabilities.npy`: Computed log probabilities for each digit

## Configuration

Key hyperparameters (in respective files):
- Latent dimension: 200
- Sigma_p: 0.4
- Learning rates:
  - Amortized VAE: 0.001
  - Latent optimization: 0.001 (decoder), 0.01 (latent vectors)
- Training epochs: 30
- Batch size: 64
- Dataset size: 20,000 stratified samples
