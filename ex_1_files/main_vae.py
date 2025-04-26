import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

from amortized_vae import AmortizedVAETrainer
from latent_optimization import LatentOptimizer
from vae_utils import compute_log_probability, save_image_grid

def setup_digit_samples(n_per_digit=5):
    """Get specific number of samples for each digit from train and test sets"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    
    train_samples = {i: [] for i in range(10)}
    test_samples = {i: [] for i in range(10)}
    
    # Collect train samples
    for img, label in train_dataset:
        label = int(label)
        if len(train_samples[label]) < n_per_digit:
            train_samples[label].append(img)
        if all(len(samples) >= n_per_digit for samples in train_samples.values()):
            break
    
    # Collect test samples
    for img, label in test_dataset:
        label = int(label)
        if len(test_samples[label]) < n_per_digit:
            test_samples[label].append(img)
        if all(len(samples) >= n_per_digit for samples in test_samples.values()):
            break
    
    # Convert to tensors
    train_samples = {k: torch.stack(v) for k, v in train_samples.items()}
    test_samples = {k: torch.stack(v) for k, v in test_samples.items()}
    
    return train_samples, test_samples

def run_amortized_vae():
    print("\nRunning Amortized VAE Experiments...")
    trainer = AmortizedVAETrainer(latent_dim=200)
    trainer.train(epochs=30)

def run_latent_optimization():
    print("\nRunning Latent Optimization Experiments...")
    optimizer = LatentOptimizer(latent_dim=200)
    optimizer.train(epochs=30)

def compare_reconstructions(epoch_list=[1, 5, 10, 20, 30]):
    print("\nComparing Reconstructions...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test samples
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    
    for epoch in epoch_list:
        # Load amortized VAE checkpoint
        amortized_checkpoint = torch.load(f'ex_1_files/checkpoints/vae_epoch_{epoch}.pt')
        amortized_model = AmortizedVAETrainer(latent_dim=200).model
        amortized_model.load_state_dict(amortized_checkpoint['model_state_dict'])
        amortized_model.eval()
        
        # Load latent optimization checkpoint
        latent_checkpoint = torch.load(f'ex_1_files/checkpoints/latent_opt_epoch_{epoch}.pt')
        latent_model = LatentOptimizer(latent_dim=200).decoder
        latent_model.load_state_dict(latent_checkpoint['decoder_state_dict'])
        latent_model.eval()
        
        # Generate reconstructions
        with torch.no_grad():
            amortized_recon, _, _ = amortized_model(test_images)
            # For latent optimization, we'll use samples from prior
            z = torch.randn(10, 200).to(device)
            latent_recon = latent_model(z)
        
        # Save comparison grid
        grid = torch.cat([test_images, amortized_recon, latent_recon], dim=0)
        save_image_grid(grid, 
                       f'ex_1_files/results/comparison_epoch_{epoch}.png',
                       nrow=10)

def compute_log_probabilities():
    print("\nComputing Log Probabilities...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load final amortized VAE model
    checkpoint = torch.load('ex_1_files/checkpoints/vae_epoch_30.pt')
    model = AmortizedVAETrainer(latent_dim=200).model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Get samples for each digit
    train_samples, test_samples = setup_digit_samples(n_per_digit=5)
    
    # Move samples to device
    train_samples = {k: v.to(device) for k, v in train_samples.items()}
    test_samples = {k: v.to(device) for k, v in test_samples.items()}
    
    # Compute log probabilities
    results = {'train': {}, 'test': {}}
    
    for digit in range(10):
        # Training samples
        log_probs = compute_log_probability(model, train_samples[digit])
        results['train'][digit] = log_probs.cpu().numpy()
        
        # Test samples
        log_probs = compute_log_probability(model, test_samples[digit])
        results['test'][digit] = log_probs.cpu().numpy()
    
    # Save results
    np.save('ex_1_files/results/log_probabilities.npy', results)
    
    # Print average log probabilities per digit
    print("\nAverage Log Probabilities per Digit:")
    print("Digit | Train | Test")
    print("-" * 25)
    for digit in range(10):
        train_avg = np.mean(results['train'][digit])
        test_avg = np.mean(results['test'][digit])
        print(f"{digit:5d} | {train_avg:6.2f} | {test_avg:6.2f}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('ex_1_files/checkpoints', exist_ok=True)
    os.makedirs('ex_1_files/results', exist_ok=True)
    
    # Run experiments
    run_amortized_vae()
    run_latent_optimization()
    compare_reconstructions()
    compute_log_probabilities()
