import torch
import torch.nn as nn

class AdaptiveDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, latents):
        return self.deconv(latents)