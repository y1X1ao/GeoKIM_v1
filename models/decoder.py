import torch.nn as nn

class MAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)
