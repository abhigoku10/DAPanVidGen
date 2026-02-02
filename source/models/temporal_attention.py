import torch
import torch.nn as nn


class RecurrentTemporalTransformer(nn.Module):
    def __init__(self,
                 latent_dim: int = 256,
                 heads: int = 4,
                 rnn_hidden_dim: int = 256,
                 rnn_layers: int = 1,
                 rnn_type: str = "gru",
                 dropout: float = 0.1,
                 causal_attention: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.causal_attention = causal_attention

        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(latent_dim, rnn_hidden_dim, rnn_layers, batch_first=False)
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(latent_dim, rnn_hidden_dim, rnn_layers, batch_first=False)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.rnn_proj = nn.Linear(rnn_hidden_dim, latent_dim)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim)
        )

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(latents)          
        rnn_out = self.rnn_proj(rnn_out)            

        attn_input = self.norm1(rnn_out)

        attn_mask = None
        if self.causal_attention:
            T = latents.size(0)
            attn_mask = torch.triu(torch.ones(T, T, device=latents.device), diagonal=1).bool()

        attn_raw, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=attn_mask)
        attn_out = rnn_out + self.dropout(attn_raw) 

        ffn_input = self.norm2(attn_out)
        ffn_out = self.ffn(ffn_input)
        out = attn_out + ffn_out               

        return out
