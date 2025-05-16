import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

class MPRAPostprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.nt_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
        
    def forward(self, logits):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        # Get most likely nucleotide at each position
        nt_ids = torch.argmax(probs, dim=-1)
        return nt_ids
    
    def decode_sequence(self, nt_ids):
        # Convert numeric IDs to nucleotide sequences
        sequences = []
        for seq in nt_ids:
            seq_str = ''.join([self.nt_map[nt.item()] for nt in seq])
            sequences.append(seq_str)
        return sequences

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
        self.mpra_postprocessor = MPRAPostprocessor()

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

    def generate_mpra_sequence(self, target_activity, num_sequences=1, temperature=1.0):
        """
        Generate MPRA sequences with desired activity.
        
        Args:
            target_activity (float): Target activity value
            num_sequences (int): Number of sequences to generate
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            list: Generated nucleotide sequences
            torch.Tensor: Predicted activities for generated sequences
        """
        device = next(self.parameters()).device
        
        # Create target activity tensor
        target = torch.tensor([[target_activity]] * num_sequences, device=device)
        
        # Create random initial sequence
        initial_seq = torch.randint(0, 5, (num_sequences, self.mpra_preprocessor.seq_len), device=device)
        
        # Process through preprocessor
        embeddings, _ = self.mpra_preprocessor((initial_seq, target))
        
        # Get latents from Perceiver
        outputs = self.perceiver(inputs=embeddings)
        latents = outputs.last_hidden_state.mean(dim=1)  # [B, d_latent]
        
        # Decode to sequence logits
        logits = self.mpra_decoder(latents)  # [B, mpra_len, 5]
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to sequences
        nt_ids = self.mpra_postprocessor(logits)
        
        # Get predicted activities
        pred_activities = self.forward_mpra(nt_ids, target)
        
        # Convert to nucleotide sequences
        sequences = self.mpra_postprocessor.decode_sequence(nt_ids)
        
        return sequences, pred_activities


# ─── Synthetic Dataset ─────────────────────────────────────────────────────────
class SyntheticTrackDataset(Dataset):
    def __init__(self, num_samples, track_len=5000):
        self.N = num_samples
        self.track_len = track_len

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        track_nt = torch.randint(0, 5, (self.track_len,), dtype=torch.long)
        logcov = torch.rand(self.track_len) * 10.0
        target = torch.rand(1) * 10.0
        return track_nt, logcov, target

class SyntheticMPRADataset(Dataset):
    def __init__(self, num_samples, mpra_len=150):
        self.N = num_samples
        self.mpra_len = mpra_len

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        mpra_nt = torch.randint(0, 5, (self.mpra_len,), dtype=torch.long)
        activity = torch.rand(1) * 10.0
        return mpra_nt, activity, activity


if __name__ == "__main__":
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create separate datasets
    track_ds = SyntheticTrackDataset(num_samples=100)
    mpra_ds = SyntheticMPRADataset(num_samples=1000)
    
    # Create separate dataloaders
    track_loader = DataLoader(track_ds, batch_size=32, shuffle=True)
    mpra_loader = DataLoader(mpra_ds, batch_size=32, shuffle=True)
    
    # Initialize model
    model = Perceiver(
        d_model=256,
        d_latent=512, 
        num_latents=128,
        num_self_attn_per_block=4,
        num_cross_attn_heads=4,
        num_self_attn_heads=8,
        seq_len=5000 
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  
    criterion = nn.MSELoss()
    
    # Training loop
    EPOCHS = 10
    for epoch in range(1, EPOCHS+1):
        # Track data training
        model.train()
        track_loss = 0.0
        for track_nt, logcov, target in track_loader:
            track_nt = track_nt.to(device)
            logcov = logcov.to(device)
            target = target.to(device)

            # Forward pass
            pred = model.forward_track(track_nt, logcov)
            loss = criterion(pred, target.squeeze(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            track_loss += loss.item() * track_nt.size(0)

        avg_track_loss = track_loss / len(track_ds)
        
        # MPRA data training
        mpra_loss = 0.0
        for mpra_nt, activity, target in mpra_loader:
            mpra_nt = mpra_nt.to(device)
            activity = activity.to(device)
            target = target.to(device)

            # Forward pass
            pred = model.forward_mpra(mpra_nt, activity)
            loss = criterion(pred, target.squeeze(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mpra_loss += loss.item() * mpra_nt.size(0)

        avg_mpra_loss = mpra_loss / len(mpra_ds)
        
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Track Loss: {avg_track_loss:.4f}")
        print(f"  MPRA Loss: {avg_mpra_loss:.4f}")

    print("\nTraining completed! Testing sequence generation...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Test sequence generation with different target activities
    target_activities = [1.0, 5.0, 9.0]  # Low, medium, and high activity targets
    temperatures = [0.5, 1.0, 1.5]  # Conservative, balanced, and random generation
    
    print("\nGenerating sequences with different target activities and temperatures:")
    print("=" * 80)
    
    for target_activity in target_activities:
        print(f"\nTarget Activity: {target_activity}")
        print("-" * 40)
        
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            print("-" * 20)
            
            # Generate sequences
            sequences, pred_activities = model.generate_mpra_sequence(
                target_activity=target_activity,
                num_sequences=3,  # Generate 3 sequences for each combination
                temperature=temp
            )
            
            # Print results
            for i, (seq, activity) in enumerate(zip(sequences, pred_activities)):
                print(f"Sequence {i+1}:")
                print(f"  Sequence: {seq}")
                print(f"  Predicted activity: {activity.item():.2f}")
                print()        