# Motivation

# Motivation

How can we unify track-based data with functional genomics assays such as Massively Parallel Reporter Assays (MPRA) or Deep Mutagenesis Scans (DMS)?

From works such as Enformer and Borzoi, we know that models trained on track-based data capture rich information about gene regulation. These models use convolutional layers to extract features from multiple cell types over very long stretches of genomic sequence, and they exhibit strong predictive power for cell-type-specific expression and variant effects. To apply them, we typically feed in fixed-length inputs and generate predictions for all tracks formatted during training. 

The next question is: how can we use these pre-trained embeddings to predict the activity of an arbitrary sequence outside its fixed training context? Researchers have begun fine-tuning these embeddings to predict short-sequence activity (e.g., https://www.biorxiv.org/content/10.1101/2024.06.23.600232v1). However, many cell types or states of interest lack labeled functional data. 

Can we therefore train a joint model that leverages both:
1. **Track-based data** (easy to collect, genome-wide, available across many tissues and cell types), and  
2. **Labeled functional genomics data** (large-scale but synthetic, covering only a limited set of cell types)  

to deliver accurate activity predictions even where direct labels are scarce?



## Project Structure

```
.
├── configs/                # Hydra configuration files
│   ├── model/              # Model-specific configurations 
│   └── config.yaml         # Main configuration file (data, training, generation params)
├── data/                   # Data loading and preprocessing
│   └── collators.py        # PyTorch Dataset classes for synthetic track and MPRA data
├── models/                 # Model architecture implementations
│   └── perceiver_model.py  # Core Perceiver model, preprocessors, and decoder
├── training/               # Training, evaluation, and generation scripts
│   ├── train.py            # Main script for training the model on both modalities
│   ├── generation.py       # Logic for generating MPRA sequences
│   └── run_generation.py   # Script to run MPRA sequence generation
├── tests/                  # Unit tests (to be added)
├── .gitignore              # Specifies intentionally untracked files that Git should ignore
└── requirements.txt        # Python dependencies
└── README.md               
```

## Setup

1.  **Create a virtual environment** (conda example):
    ```bash
    conda create -n lsfg python=3.10 
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

All scripts that use Hydra (training, generation) should be run from the **project root directory**.

### 1. Configuration

Key parameters for the model, training, data, and generation are managed via YAML files in the `configs/` directory.
*   `configs/config.yaml`: Main configuration.
    *   `model`: Specifies which model configuration to use (defaults to `perceiver`, linking to `configs/model/perceiver.yaml`).
    *   `data`: Parameters for data generation (e.g., `num_track_train_samples`, `num_mpra_train_samples`).
    *   `training`: Training hyperparameters (e.g., `batch_size`, `epochs`, `learning_rate`, `checkpoint_dir`).
    *   `generation`: Parameters for sequence generation (e.g., `checkpoint_path`, `target_activity`, `num_sequences_to_generate`, `temperature`).
*   `configs/model/perceiver.yaml`: Parameters for the Perceiver model architecture (e.g., `d_model`, `d_latent`, `num_latents`, `seq_len`, `mpra_len`).

You can override any configuration parameter from the command line. See [Hydra documentation](https://hydra.cc/docs/intro/) for more details.

### 2. Training

The model is trained jointly on synthetic track data and MPRA data.

To start training with default parameters specified in `configs/config.yaml`:
```bash
python training/train.py
```

Checkpoints and logs will be saved in a timestamped directory under `outputs/` (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`). The training script saves the final model as `perceiver_joint_final.pth` within this `checkpoints` subdirectory.

**Override parameters:**
You can override any configuration parameter from the command line. For example, to change the number of epochs and learning rate:
```bash
python training/train.py training.epochs=50 training.learning_rate=0.0005
```
To run a sweep over multiple parameters (multirun):
```bash
python training/train.py --multirun training.epochs=10,20 model.d_latent=256,512
```
Results from multiruns will be saved under the `multirun/` directory.

### 3. Generating MPRA Sequences

After training, you can generate MPRA sequences with desired activity levels.

Make sure the `generation.checkpoint_path` in `configs/config.yaml` points to your trained model checkpoint (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/perceiver_joint_final.pth`). Update this path to match your specific training run output.

To generate sequences with default parameters:
```bash
python training/run_generation.py
```

**Override generation parameters:**
For example, to generate 10 sequences with a target activity of 2.5 and a specific temperature, using a specific checkpoint:
```bash
python training/run_generation.py generation.num_sequences_to_generate=10 generation.target_activity=2.5 generation.temperature=0.7 generation.checkpoint_path="outputs/YOUR_TRAINING_RUN_TIMESTAMP/checkpoints/perceiver_joint_final.pth"
```

The script will print the generated sequences and their predicted activities.


## TODO
-   Integrate real biological datasets.
-   Implement more sophisticated data loading and preprocessing.
    -  How do we embed track data in a more sophisticated way? 
    -  How do we incorporate cell type information? 
-   Add comprehensive evaluation metrics and validation loops.
-   Add unit tests.
