import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

# Adjust these imports based on final locations
from models.perceiver_model import Perceiver 
from data.collators import PerceiverTrackDataset, PerceiverMPRADataset
from training.generation import generate_mpra_sequence 


# Basic training loop (can be expanded)
def train_model(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data_nt, data_aux, target) in enumerate(dataloader):
            data_nt, data_aux, target = data_nt.to(device), data_aux.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            predictions = model.forward_track(data_nt, data_aux) 
            
            loss = criterion(predictions, target.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(dataloader):.4f}")

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model using Hydra config
    # The Perceiver class __init__ needs to match these parameters or accept a cfg object
    model = Perceiver(
        d_model=cfg.model.d_model,
        d_latent=cfg.model.d_latent,
        num_latents=cfg.model.num_latents,
        num_self_attn_per_block=cfg.model.num_self_attn_per_block,
        num_cross_attn_heads=cfg.model.num_cross_attn_heads,
        num_self_attn_heads=cfg.model.num_self_attn_heads,
        seq_len=cfg.model.seq_len, # This is track_len for TrackPreprocessor
        mpra_len=cfg.model.mpra_len,
        num_bins=cfg.model.num_bins,
        min_cov=cfg.model.min_cov,
        max_cov=cfg.model.max_cov
    ).to(device)

    # Setup datasets and dataloaders for both modalities
    track_dataset = PerceiverTrackDataset(num_samples=cfg.data.num_track_train_samples, track_len=cfg.model.seq_len)
    mpra_dataset = PerceiverMPRADataset(num_samples=cfg.data.num_mpra_train_samples, mpra_len=cfg.model.mpra_len)

    track_dataloader = DataLoader(
        track_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True
    )
    mpra_dataloader = DataLoader(
        mpra_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    
    criterion = torch.nn.MSELoss() # Using MSE for both as an example
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    print(f"Starting joint training on {device}")
    
    for epoch in range(cfg.training.epochs):
        model.train()
        
        # --- Track Data Training ---
        epoch_track_loss = 0
        for batch_idx, (track_nt, logcov, track_target) in enumerate(track_dataloader):
            track_nt, logcov, track_target = track_nt.to(device), logcov.to(device), track_target.to(device)
            
            optimizer.zero_grad()
            track_predictions = model.forward_track(track_nt, logcov)
            loss_track = criterion(track_predictions, track_target.squeeze(-1) if track_target.ndim > 1 else track_target)
            loss_track.backward()
            optimizer.step()
            
            epoch_track_loss += loss_track.item()
            if batch_idx % cfg.training.log_interval == 0:
                print(f"Epoch {epoch+1}/{cfg.training.epochs} [Track], Batch {batch_idx}/{len(track_dataloader)}, Track Loss: {loss_track.item():.4f}")
        
        avg_epoch_track_loss = epoch_track_loss / len(track_dataloader) if len(track_dataloader) > 0 else 0
        print(f"Epoch {epoch+1} average Track Loss: {avg_epoch_track_loss:.4f}")

        epoch_mpra_loss = 0
        for batch_idx, (mpra_nt, mpra_activity_input, mpra_target_activity) in enumerate(mpra_dataloader):
            mpra_nt = mpra_nt.to(device)
            mpra_activity_input = mpra_activity_input.to(device)
            mpra_target_activity = mpra_target_activity.to(device)

            optimizer.zero_grad()
            mpra_predictions = model.forward_mpra(mpra_nt, mpra_activity_input)
            loss_mpra = criterion(mpra_predictions, mpra_target_activity.squeeze(-1) if mpra_target_activity.ndim > 1 else mpra_target_activity)
            loss_mpra.backward()
            optimizer.step()

            epoch_mpra_loss += loss_mpra.item()
            if batch_idx % cfg.training.log_interval == 0:
                print(f"Epoch {epoch+1}/{cfg.training.epochs} [MPRA], Batch {batch_idx}/{len(mpra_dataloader)}, MPRA Loss: {loss_mpra.item():.4f}")

        avg_epoch_mpra_loss = epoch_mpra_loss / len(mpra_dataloader) if len(mpra_dataloader) > 0 else 0
        print(f"Epoch {epoch+1} average MPRA Loss: {avg_epoch_mpra_loss:.4f}")

        if cfg.training.get("run_generation_test", False) and (epoch + 1) % 5 == 0 : # Every 5 epochs
            print("Running generation test...")
            model.eval() 
            generated_seqs, pred_activities = generate_mpra_sequence(
                model, 
                target_activity=cfg.generation.target_activity, 
                num_sequences=cfg.generation.num_sequences_to_generate, 
                mpra_len=cfg.model.mpra_len,
                temperature=cfg.generation.temperature,
                device=device
            )
            print(f"Generated sequences: {generated_seqs}")
            print(f"Predicted activities for generated sequences: {pred_activities}")
            # model.train() # Already set at the beginning of the epoch loop

    print("Training finished.")
    # Ensure checkpoint directory exists
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True) # Add this line
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'perceiver_joint_final.pth'))
    print(f"Model saved to {os.path.join(checkpoint_dir, 'perceiver_joint_final.pth')}")

if __name__ == "__main__":
    main() 