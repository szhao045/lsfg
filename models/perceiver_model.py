import torch
import torch.nn as nn
from transformers import PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import AbstractPreprocessor

class TrackPreprocessor(AbstractPreprocessor):
    def __init__(self, config):
        super().__init__()
        self.nt_embed = nn.Embedding(5, config.d_model)
        self.cov_embed = nn.Embedding(config.num_bins, config.d_model)
        self.pos_embed = nn.Embedding(config.seq_len, config.d_model)
        
        # Fixed bin edges for coverage
        edges = torch.linspace(config.min_cov, config.max_cov, steps=config.num_bins+1)
        self.register_buffer('bin_edges', edges)
        self.num_bins = config.num_bins
        self.seq_len = config.seq_len

    def forward(self, inputs):
        nt_tokens, logcov = inputs
        
        # Nucleotide embedding
        seq_emb = self.nt_embed(nt_tokens)  # [B, L, D]
        
        # Coverage embedding
        cov_clamped = logcov.clamp(self.bin_edges[0], self.bin_edges[-1])
        bin_ids = torch.bucketize(cov_clamped, self.bin_edges) - 1
        bin_ids = bin_ids.clamp(0, self.num_bins - 1)
        cov_emb = self.cov_embed(bin_ids)  # [B, L, D]
        
        # Positional embedding
        positions = torch.arange(self.seq_len, device=nt_tokens.device)
        pos_emb = self.pos_embed(positions).unsqueeze(0)  # [1, L, D]
        
        # Combine embeddings
        embeddings = seq_emb + cov_emb + pos_emb  # [B, L, D]
        
        return embeddings, None  # Return embeddings and None for mask

class MPRAPreprocessor(AbstractPreprocessor):
    def __init__(self, config):
        super().__init__()
        self.nt_embed = nn.Embedding(5, config.d_model)
        self.act_proj = nn.Linear(1, config.d_model)
        self.pos_embed = nn.Embedding(config.seq_len + 1, config.d_model)
        self.seq_len = config.seq_len

    def forward(self, inputs):
        nt_tokens, activity = inputs
        B = nt_tokens.shape[0]
        
        # Sequence embedding
        seq_emb = self.nt_embed(nt_tokens)  # [B, L, D]
        
        # Activity embedding
        act = activity.view(B, 1)  # [B, 1]
        act_emb = self.act_proj(act)  # [B, D]
        act_emb = act_emb.unsqueeze(1)  # [B, 1, D]
        
        # Combine and add positional embedding
        x = torch.cat([act_emb, seq_emb], dim=1)  # [B, L+1, D]
        positions = torch.arange(self.seq_len + 1, device=nt_tokens.device)
        pos_emb = self.pos_embed(positions)  # [L+1, D]
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # [B, L+1, D]
        
        return x + pos_emb, None  # Return embeddings and None for mask

class MPRADecoder(nn.Module):
    def __init__(self, d_model, d_latent, mpra_len):
        super().__init__()
        self.mpra_len = mpra_len
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent * 4),
            nn.GELU(),
            nn.Linear(d_latent * 4, mpra_len * 5)  # Output logits for each position
        )
        
    def forward(self, latents):
        # latents: [B, d_latent]
        logits = self.decoder(latents)  # [B, mpra_len * 5]
        logits = logits.view(-1, self.mpra_len, 5)  # [B, mpra_len, 5]
        return logits

class Perceiver(nn.Module):
    def __init__(self,
                 d_model: int = 256,
                 d_latent: int = 512,
                 num_latents: int = 128,
                 num_self_attn_per_block: int = 4,
                 num_cross_attn_heads: int = 4,
                 num_self_attn_heads: int = 8,
                 seq_len: int = 1000,
                 mpra_len: int = 150,
                 num_bins: int = 256,
                 min_cov: float = 0.0,
                 max_cov: float = 10.0):
        super().__init__()
        
        # Create config
        config = PerceiverConfig(
            d_model=d_model,
            d_latents=d_latent,
            num_latents=num_latents,
            num_self_attn_per_block=num_self_attn_per_block,
            num_cross_attn_heads=num_cross_attn_heads,
            num_self_attn_heads=num_self_attn_heads,
            hidden_act="gelu",
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1
        )
        
        # Add custom config parameters
        config.seq_len = seq_len
        config.mpra_len = mpra_len
        config.num_bins = num_bins
        config.min_cov = min_cov
        config.max_cov = max_cov
        
        # Create preprocessors with their respective sequence lengths
        self.track_preprocessor = TrackPreprocessor(config)
        mpra_config = PerceiverConfig(**config.to_dict())
        mpra_config.seq_len = mpra_len
        self.mpra_preprocessor = MPRAPreprocessor(mpra_config)
        
        # Create Perceiver model
        self.perceiver = PerceiverModel(config)
        
        # Output heads
        self.track_head = nn.Linear(d_latent, 1)
        self.mpra_head = nn.Linear(d_latent, 1)
        
        # Decoder and postprocessor for MPRA generation
        self.mpra_decoder = MPRADecoder(d_model, d_latent, mpra_len)

    def forward_track(self, track_nt, logcov):
        # Preprocess track data
        embeddings, _ = self.track_preprocessor((track_nt, logcov))
        
        # Process through Perceiver
        outputs = self.perceiver(inputs=embeddings)
        latents = outputs.last_hidden_state
        
        # Pool and predict
        pooled = latents.mean(dim=1)  # [B, d_latent]
        return self.track_head(pooled).squeeze(-1)

    def forward_mpra(self, mpra_nt, activity):
        # Preprocess MPRA data
        embeddings, _ = self.mpra_preprocessor((mpra_nt, activity))
        
        # Process through Perceiver
        outputs = self.perceiver(inputs=embeddings)
        latents = outputs.last_hidden_state
        
        # Pool and predict
        pooled = latents.mean(dim=1)  # [B, d_latent]
        return self.mpra_head(pooled).squeeze(-1)
