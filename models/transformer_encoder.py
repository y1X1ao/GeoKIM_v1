# transformer_encoder.py
import torch
import torch.nn as nn

class TabularTransformerEncoder(nn.Module):
    def __init__(self, num_tokens, embed_dim=64, depth=4, heads=4, mlp_ratio=4.0, pooling='mean'):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.pooling = pooling

       
        self.token_embed = nn.Linear(1, embed_dim)

    
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
     
        x = x.unsqueeze(-1)
        x = self.token_embed(x) + self.pos_embed  
        x = self.encoder(x) 
      
        if self.pooling == 'mean':
            x = x.mean(dim=1) 
        elif self.pooling == 'cls':
            x = x[:, 0]  
        elif self.pooling == 'max':
            x, _ = x.max(dim=1)
        elif self.pooling == 'mean+max':
            x_mean = x.mean(dim=1)
            x_max, _ = x.max(dim=1)
            x = torch.cat([x_mean, x_max], dim=-1) 
        elif self.pooling == 'flatten':
            x = x.reshape(x.size(0), -1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        return x