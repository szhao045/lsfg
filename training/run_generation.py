import sys
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.perceiver_model import Perceiver
from training.generation import generate_mpra_sequence

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Running MPRA Sequence Generation with configuration:")
    print(OmegaConf.to_yaml(cfg.generation))
    print("Model configuration:")
    print(OmegaConf.to_yaml(cfg.model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Perceiver(
        d_model=cfg.model.d_model,
        d_latent=cfg.model.d_latent,
        num_latents=cfg.model.num_latents,
        num_self_attn_per_block=cfg.model.num_self_attn_per_block,
        num_cross_attn_heads=cfg.model.num_cross_attn_heads,
        num_self_attn_heads=cfg.model.num_self_attn_heads,
        seq_len=cfg.model.seq_len,       # Used by TrackPreprocessor
        mpra_len=cfg.model.mpra_len,     # Used by MPRAPreprocessor & Decoder
        num_bins=cfg.model.num_bins,
        min_cov=cfg.model.min_cov,
        max_cov=cfg.model.max_cov
    ).to(device)

    # Load checkpoint
    # Resolve checkpoint_path to an absolute path relative to the original CWD
    checkpoint_path = to_absolute_path(cfg.generation.checkpoint_path)
    print(f"Attempting to load checkpoint from absolute path: {checkpoint_path}") # Logging the absolute path

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please ensure you have trained a model and the path in configs/config.yaml under generation.checkpoint_path is correct.")
        return
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model checkpoint from {checkpoint_path}: {e}")
        print("Ensure the checkpoint is compatible with the current model architecture defined by your model config.")
        return

    model.eval() # Set model to evaluation mode

    print(f"\nGenerating {cfg.generation.num_sequences_to_generate} MPRA sequences with target activity {cfg.generation.target_activity}...")

    generated_sequences, predicted_activities = generate_mpra_sequence(
        model=model,
        target_activity=cfg.generation.target_activity,
        num_sequences=cfg.generation.num_sequences_to_generate,
        temperature=cfg.generation.temperature,
        mpra_len=cfg.model.mpra_len, # mpra_len for generation should match model's mpra_len, TODO: Probably we can handle the variable length with a better collator
        device=device
    )

    print("\n--- Generated Sequences and Predicted Activities ---")
    for i, (seq, activity) in enumerate(zip(generated_sequences, predicted_activities)):
        print(f"Sequence {i+1}: {seq}")
        print(f"  Predicted Activity: {activity.item():.4f}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main() 