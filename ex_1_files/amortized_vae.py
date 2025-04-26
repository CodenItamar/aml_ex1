import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

from model import ConvVAE
from vae_utils import (
    compute_kl_loss,
    compute_recon_loss,
    save_image_grid,
    plot_reconstructions,
    plot_training_curves
)

class AmortizedVAETrainer:
    def __init__(self, latent_dim=200, sigma_p=0.4, lr=0.001, device=None):
        self.latent_dim = latent_dim
        self.sigma_p = sigma_p
        self.lr = lr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and optimizer
        self.model = ConvVAE(latent_dim=latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup data
        self.setup_data()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
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
        train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                     download=True, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(self.train_loader, desc='Training')):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            
            # Compute losses
            recon_loss = compute_recon_loss(recon_batch, data, self.sigma_p)
            kl_loss = compute_kl_loss(mu, logvar)
            loss = recon_loss + kl_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record losses
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # Save first batch reconstructions at specific epochs
            if batch_idx == 0 and self.current_epoch in [0, 4, 9, 19, 29]:
                self.save_epoch_results(data, recon_batch)
        
        return {
            'total': total_loss / len(self.train_loader),
            'recon': total_recon / len(self.train_loader),
            'kl': total_kl / len(self.train_loader)
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        with torch.no_grad():
            for data, _ in tqdm(self.test_loader, desc='Validating'):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                
                recon_loss = compute_recon_loss(recon_batch, data, self.sigma_p)
                kl_loss = compute_kl_loss(mu, logvar)
                loss = recon_loss + kl_loss
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
        
        return {
            'total': total_loss / len(self.test_loader),
            'recon': total_recon / len(self.test_loader),
            'kl': total_kl / len(self.test_loader)
        }
    
    def save_epoch_results(self, data, recon_batch):
        # Save reconstructions
        plot_reconstructions(
            data[:10], 
            recon_batch[:10],
            self.current_epoch,
            f'ex_1_files/results/recon_epoch_{self.current_epoch+1}.png'
        )
        
        # Save model checkpoint
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'ex_1_files/checkpoints/vae_epoch_{self.current_epoch+1}.pt')
    
    def sample_from_prior(self, n_samples=10):
        """Sample from prior N(0,I) and generate images"""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            samples = self.model.decode(z)
        return samples
    
    def train(self, epochs=30):
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses['total'])
            
            # Validation
            val_losses = self.validate()
            self.val_losses.append(val_losses['total'])
            
            # Print progress
            print(f'Train Loss: {train_losses["total"]:.4f} '
                  f'(Recon: {train_losses["recon"]:.4f}, KL: {train_losses["kl"]:.4f})')
            print(f'Valid Loss: {val_losses["total"]:.4f} '
                  f'(Recon: {val_losses["recon"]:.4f}, KL: {val_losses["kl"]:.4f})')
            
            # Sample from prior at specific epochs
            if epoch in [0, 4, 9, 19, 29]:
                samples = self.sample_from_prior(n_samples=10)
                save_image_grid(samples, 
                              f'ex_1_files/results/samples_epoch_{epoch+1}.png',
                              nrow=5)
        
        # Plot final training curves
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            'ex_1_files/results/training_curves.png'
        )
