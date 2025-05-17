import torch
import torch.nn as nn

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
        for seq_ids_single in nt_ids:
            seq_str = ''.join([self.nt_map[nt_id.item()] for nt_id in seq_ids_single])
            sequences.append(seq_str)
        return sequences

def generate_mpra_sequence(model, target_activity, num_sequences=1, temperature=1.0, mpra_len=150, device=None):
    """
    Generate MPRA sequences with desired activity.
    
    Args:
        model: The trained Perceiver model.
        target_activity (float): Target activity value
        num_sequences (int): Number of sequences to generate
        temperature (float): Sampling temperature (higher = more random)
        mpra_len (int): Length of the MPRA sequence to generate.
        device (torch.device, optional): Device to run generation on.
            
    Returns:
        list: Generated nucleotide sequences
        torch.Tensor: Predicted activities for generated sequences
    """
    if device is None:
        device = next(model.parameters()).device
    
    mpra_postprocessor = MPRAPostprocessor() # Instantiate within the function

    # Create target activity tensor
    target = torch.tensor([[target_activity]] * num_sequences, device=device, dtype=torch.float32)
    
    # Create random initial sequence (as a placeholder for preprocessor input)
    # The model's mpra_preprocessor will handle the actual embedding.
    # We pass a dummy sequence here because mpra_preprocessor expects nt_tokens.
    initial_seq_dummy = torch.randint(0, 5, (num_sequences, mpra_len), device=device, dtype=torch.long)
    
    # Preprocess inputs for the Perceiver model (activity only, sequence part is what we generate)
    # The mpra_preprocessor in the Perceiver model expects (nt_tokens, activity)
    # For generation, we are effectively trying to find nt_tokens given activity.
    # The Perceiver's forward_mpra takes (mpra_nt, activity) and processes them.
    # The decoder then takes the latent representation to generate sequence logits.

    # We need the latent representation that corresponds to the target_activity.
    # We can get this by passing a dummy sequence and the target activity to the mpra_preprocessor and then the perceiver.
    
    # Get embeddings for the target activity and a dummy sequence
    embeddings, _ = model.mpra_preprocessor((initial_seq_dummy, target))
    
    # Get latents from Perceiver
    outputs = model.perceiver(inputs=embeddings)
    latents = outputs.last_hidden_state.mean(dim=1)  # [B, d_latent]
    
    # Decode to sequence logits
    # The decoder is part of the main Perceiver model: model.mpra_decoder
    logits = model.mpra_decoder(latents)  # [B, mpra_len, 5]
    
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    
    # Convert to sequences
    nt_ids = mpra_postprocessor(logits)
    
    # Get predicted activities for the generated sequences
    # We need to ensure nt_ids are on the correct device for model.forward_mpra
    pred_activities = model.forward_mpra(nt_ids.to(device), target)
    
    # Convert to nucleotide sequences
    sequences = mpra_postprocessor.decode_sequence(nt_ids)
    
    return sequences, pred_activities 