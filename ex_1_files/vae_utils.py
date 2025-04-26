import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

def compute_kl_loss(mu, logvar):
    """
    Compute KL divergence loss between N(mu, var) and N(0, I)
    KL(N(mu, var) || N(0, I)) = 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
    """
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_loss.mean()

def compute_recon_loss(recon_x, x, sigma_p=0.4):
    """
    Compute reconstruction loss (MSE) scaled by sigma_p^2
    """
    mse = F.mse_loss(recon_x, x, reduction='none')
    return torch.sum(mse, dim=[1,2,3]).mean() / (2 * sigma_p**2)

def compute_log_probability(model, x, M=1000, sigma_p=0.4):
    """
    Compute log p(x) using importance sampling with M samples
    Following equation 9 from the PDF
    """
    with torch.no_grad():
        # Get encoder parameters
        mu, logvar = model.encode(x)
        
        # Sample M values of z
        std = torch.exp(0.5 * logvar)
        z_samples = mu.unsqueeze(1) + std.unsqueeze(1) * torch.randn(mu.size(0), M, mu.size(1), device=mu.device)
        
        # Compute log p(x|z)
        x_expanded = x.unsqueeze(1).expand(-1, M, -1, -1, -1)
        recon_x = model.decode(z_samples.view(-1, z_samples.size(-1)))
        recon_x = recon_x.view(x_expanded.shape)
        log_p_x_z = -torch.sum((x_expanded - recon_x)**2, dim=[2,3,4]) / (2 * sigma_p**2)
        
        # Compute log p(z)
        log_p_z = -0.5 * torch.sum(z_samples**2, dim=-1)
        
        # Compute log q(z|x)
        log_q_z = -0.5 * torch.sum(((z_samples - mu.unsqueeze(1)) / std.unsqueeze(1))**2 + logvar.unsqueeze(1), dim=-1)
        
        # Combine terms and use logsumexp for stability
        log_weights = log_p_x_z + log_p_z - log_q_z
        log_p_x = torch.logsumexp(log_weights, dim=1) - torch.log(torch.tensor(M, device=x.device, dtype=torch.float))
        
        return log_p_x

def save_image_grid(images, path, nrow=10, padding=2, normalize=True):
    """
    Save a grid of images
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(images, path, nrow=nrow, padding=padding, normalize=normalize)

def plot_reconstructions(original, recon, epoch, save_path=None):
    """
    Plot original and reconstructed images side by side
    """
    plt.figure(figsize=(10, 4))
    
    # Original images
    plt.subplot(1, 2, 1)
    plt.imshow(vutils.make_grid(original, normalize=True).permute(1, 2, 0))
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed images
    plt.subplot(1, 2, 2)
    plt.imshow(vutils.make_grid(recon, normalize=True).permute(1, 2, 0))
    plt.title(f'Reconstructed (Epoch {epoch+1})')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """
    Plot training (and optionally validation) curves
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
