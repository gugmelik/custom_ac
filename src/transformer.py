import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from src.ckpt import checkpoint_block

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #with torch.profiler.record_function("attn"):
        x = x + self.attn(self.ln1(x))
        #with torch.profiler.record_function("ff"):
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        use_checkpoint: bool = False,
        checkpoint_impl: str = "custom",
    ) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            with torch.profiler.record_function(f"block_{i}"):
                if use_checkpoint:
                    if checkpoint_impl == "custom":
                        x = checkpoint_block(block, x, preserve_rng_state=True)
                    elif checkpoint_impl == "torch":
                        # use_reentrant=False is the recommended implementation in modern PyTorch.
                        x = torch_checkpoint(
                            block,
                            x,
                            preserve_rng_state=True,
                            use_reentrant=False,
                        )
                    else:
                        raise ValueError(f"Unknown checkpoint_impl={checkpoint_impl!r}")
                else:
                    x = block(x)

        x = self.ln_f(x)
        return self.head(x)


class NNTransformerLM(nn.Module):
    """Language model wrapper around nn.TransformerEncoder for benchmarking."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask)
        x = self.ln_f(x)
        return self.head(x)