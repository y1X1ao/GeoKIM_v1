import torch.nn as nn
import torch
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def fit(self, X_np, epochs=100):
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.forward(X_tensor), X_tensor)
            loss.backward()
            optimizer.step()

    def encode(self, X_np):
        with torch.no_grad():
            return self.encoder(torch.tensor(X_np, dtype=torch.float32)).numpy()
