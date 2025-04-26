import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

from model import ConvVAE
from vae_utils import compute_recon_loss, save_image_grid, plot_reconstructions, plot_training_curves

class LatentOptimizer:
    def __init__(self, latent_dim=200, sigma_p=0.4, decoder_lr=0.001, latent_lr=0.01, device=None):
        self.latent_dim = latent_dim
        self.sigma_p = sigma_p
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model (we'll only use the decoder)
        full_vae = ConvVAE(latent_dim=latent_dim).to(self.device)
        self.decoder = nn.Sequential(
            full_vae.fc_decode,
            lambda x: x.view(-1, 128, 1, 1),
            full_vae.decoder
        ).to(self.device)
        
        # Setup data and initialize latent vectors
        self.setup_data()
        self.initialize_latent_vectors()
        
        # Optimizers
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=decoder_lr)
        self.latent_optimizer = optim.Adam([self.latent_vectors], lr=latent_lr)
        
        # Training history
        self.train_losses = []
        
    def setup_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load MNIST dataset
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                                 download=True, transform=transform)
        
        # Take stratified subset of 20,000 samples
        train_targets = train_dataset.targets
        train_idx, _ = train_test_split(range(len(train_targets)), 
                                      train_size=20000, 
                                      stratify=train_targets)
        
        # Extract data and create dataset
        self.train_data = torch.stack([train_dataset[i][0] for i in train_idx]).to(self.device)
        self.train_loader = DataLoader(
            TensorDataset(torch.arange(len(self.train_data)), self.train_data),
            batch_size=64, 
            shuffle=True
        )
        
    def initialize_latent_vectors(self):
        """Initialize one latent vector per training example"""
        self.latent_vectors = torch.randn(len(self.train_data), self.latent_dim).to(self.device)
        self.latent_vectors.requires_grad = True
    
    def train_epoch(self):
        total_recon_loss = 0
        total_prior_loss = 0
        
        for batch_idx, (indices, data) in enumerate(tqdm(self.train_loader, desc='Training')):
            # Get corresponding latent vectors
            z = self.latent_vectors[indices]
            
            # Forward pass
            self.decoder_optimizer.zero_grad()
            self.latent_optimizer.zero_grad()
            
            recon = self.decoder(z)
            
            # Compute losses
            recon_loss = compute_recon_loss(recon, data, self.sigma_p)
            prior_loss = 0.5 * torch.sum(z**2) / len(data)  # Prior regularization
            
            loss = recon_loss + prior_loss
            
            # Backward pass
            loss.backward()
            
            # Update both decoder and latent vectors
            self.decoder_optimizer.step()
            self.latent_optimizer.step()
            
            # Record losses
            total_recon_loss += recon_loss.item()
            total_prior_loss += prior_loss.item()
            
            # Save reconstructions at specific epochs
            if batch_idx == 0 and self.current_epoch in [0, 4, 9, 19, 29]:
                self.save_epoch_results(data, recon)
        
        return {
            'recon': total_recon_loss / len(self.train_loader),
            'prior': total_prior_loss / len(self.train_loader),
            'total': (total_recon_loss + total_prior_loss) / len(self.train_loader)
        }
    
    def save_epoch_results(self, data, recon):
        # Save reconstructions
        plot_reconstructions(
            data[:10],
            recon[:10],
            self.current_epoch,
            f'ex_1_files/results/latent_opt_recon_epoch_{self.current_epoch+1}.png'
        )
        
        # Save model checkpoint
        torch.save({
            'epoch': self.current_epoch,
            'decoder_state_dict': self.decoder.state_dict(),
            'latent_vectors': self.latent_vectors,
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'latent_optimizer_state_dict': self.latent_optimizer.state_dict(),
        }, f'ex_1_files/checkpoints/latent_opt_epoch_{self.current_epoch+1}.pt')
    
    def sample_from_prior(self, n_samples=10):
        """Sample from prior N(0,I) and generate images"""
        self.decoder.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            samples = self.decoder(z)
        return samples
    
    def train(self, epochs=30):
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training
            losses = self.train_epoch()
            self.train_losses.append(losses['total'])
            
            # Print progress
            print(f'Loss: {losses["total"]:.4f} '
                  f'(Recon: {losses["recon"]:.4f}, Prior: {losses["prior"]:.4f})')
            
            # Sample from prior at specific epochs
            if epoch in [0, 4, 9, 19, 29]:
                samples = self.sample_from_prior(n_samples=10)
                save_image_grid(
                    samples,
                    f'ex_1_files/results/latent_opt_samples_epoch_{epoch+1}.png',
                    nrow=5
                )
        
        # Plot final training curve
        plot_training_curves(
            self.train_losses,
            save_path='ex_1_files/results/latent_opt_training_curve.png'
        )
