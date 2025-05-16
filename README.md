# RNA-seq and MPRA Analysis with PerceiverIO

How do we unify the track based data with functional genomics data like Massively Parallel Reporter Assay (MPRA) or Deep Mutagenesis Scan (DMS)? 

## Project Structure

```
.
├── data/                # Data processing and loading modules
├── models/              # Model architecture implementations
├── training/            # Training and evaluation scripts
├── utils/               # Utility functions
├── configs/             # Configuration files
└── tests/               # Unit tests
```

## Setup

1. Create a virtual environment:
```bash
conda create -n lsfg python=3.10
conda activate lsfg

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Brief test of the overall pipeline:
```bash
python -m models.train_perceiver
```

## Features

- PerceiverIO architecture implementation for multi-modal biological data
- Support for RNA-seq data processing and analysis
- MPRA data integration and analysis
- Training and evaluation pipelines
- Data visualization tools

## Contributing

[Contributing guidelines will be added]

## License

[License information will be added] 