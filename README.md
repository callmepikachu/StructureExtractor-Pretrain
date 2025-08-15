# StructureExtractor-Pretrain

Pre-training StructureExtractor for GDM-Net, a long document graph construction model integrating ContinualGNN and StreamE ideas.

## Project Description

This project implements a pre-training framework for StructureExtractor, a core component of GDM-Net designed for long document graph construction. It combines the strengths of ContinualGNN and StreamE to efficiently build and update knowledge graphs from document streams while mitigating catastrophic forgetting.

### Baseline

The baseline for this project is the GDM-Net framework, which decomposes document-level relation extraction (DocRE) into two stages:
1. **StructureExtractor**: Builds a knowledge graph from the input document.
2. **GMF (Graph-based Matching Framework)**: Performs relation classification on the extracted graph.

This project focuses on pre-training the StructureExtractor component using the ReDocRED dataset. The goal is to create a robust graph construction model that can be fine-tuned for specific DocRE tasks.

### Dataset

The primary dataset used for pre-training is **ReDocRED**, a large-scale human-annotated dataset for document-level relation extraction. ReDocRED extends the DocRED dataset with revised annotations and additional documents, making it suitable for evaluating long document understanding.

Key characteristics of ReDocRED:
- Contains 10,769 documents for training
- Includes 964 documents for development and 994 for testing
- Covers 96 entity types and 49 relation types
- Provides document-level annotations with multiple facts per document

## Project Structure

```
StructureExtractor-Pretrain/
├── README.md                           # Project introduction, installation guide, usage instructions
├── requirements.txt                    # Python dependencies list
├── config/                             # Configuration files directory
│   ├── default_config.yaml             # Default configuration file
│   └── # Other specific configurations (e.g., for_training.yaml, for_inference.yaml)
├── src/                                # Main source code directory
│   ├── __init__.py
│   ├── data/                           # Data processing related code
│   │   ├── __init__.py
│   │   ├── dataset.py                  # ReDocRED dataset class (ReDocREDDataset)
│   │   ├── collator.py                 # Custom data collator (DataCollator)
│   │   └── preprocess.py               # (Optional) Data preprocessing scripts/functions
│   ├── model/                          # Model related code (core is StructureExtractor)
│   │   ├── __init__.py
│   │   ├── extractor.py                # The designed StructureExtractor model (can reuse or fine-tune version from GDM-Net)
│   │   └── # Other model components (if needed, such as simplified GraphWriter for validation)
│   ├── train/                          # Training related code
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Trainer class (PretrainTrainer), encapsulating training loop
│   │   ├── train.py                    # Training script entry point (main function)
│   │   └── train_eval.py               # Integrated training and evaluation script
│   ├── evaluate/                       # Evaluation related code
│   │   ├── __init__.py
│   │   ├── evaluate.py                 # Evaluation functions
│   │   └── eval.py                     # Evaluation script entry point
│   └── utils/                          # Utility functions
│       ├── __init__.py
│       ├── config.py                   # Configuration loading and validation
│       ├── logger.py                   # Logging configuration
│       └── # Other general tools (e.g., metrics calculation)
├── scripts/                            # Utility scripts directory
│   ├── download_redocred.sh            # Script to download ReDocRED data
│   └── run_pipeline.sh                 # Script to run the complete training and evaluation pipeline
├── notebooks/                          # (Optional) Jupyter Notebooks for exploratory data analysis or model experiments
│   └── # ...
├── checkpoints/                        # (Initially empty) Directory for storing trained model checkpoints
│   └── # ...
├── logs/                               # (Initially empty) Directory for storing training logs
│   └── # ...
└── data/                               # (Initially empty or symbolic link) Points to the actual location of ReDocRED dataset
    ├── train.json                      # ReDocRED training set
    ├── dev.json                        # ReDocRED validation set
    └── # ... (test.json etc.)
```

## Installation Guide

1. Clone the project repository:
   ```
   git clone <repository-url>
   cd StructureExtractor-Pretrain
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Download Dataset
```
bash scripts/download_redocred.sh
```

### 2. Run Complete Training and Evaluation Pipeline
```
bash scripts/run_pipeline.sh
```

### 3. Train Model Separately

#### Train with Preprocessed Data (Recommended for Large Datasets)
```
python src/train/train.py --config config/default_config.yaml --train-data data/preprocessed_train --dev-data data/dev_revised.json --output-dir output --use-preprocessed-data
```

#### Train with Raw Data
```
python src/train/train.py --config config/default_config.yaml --train-data data/train_revised.json --dev-data data/dev_revised.json --output-dir output
```

### 4. Evaluate Model

Using the adapted evaluation script (recommended):
```
python src/evaluate/eval_adapted.py --config config/default_config.yaml --checkpoint output/checkpoints/best_model.pt --test-data data/dev.json --output-dir output
```

Or using the original evaluation script:
```
python src/evaluate/eval.py \
    --config config/default_config.yaml \
    --checkpoint output/checkpoints/best_model.pt \
    --test-data data/dev.json \
    --output-dir output
```

### 5. Preprocess Data (Optional, Recommended for Large Datasets)
```
python scripts/preprocess_data.py --config config/default_config.yaml --input-data data/train_revised.json --output-dir data/preprocessed_train
```

## Configuration

The configuration file `config/default_config.yaml` contains all parameters for model, data, training, and evaluation. Main configuration items include:

- `model`: Model architecture parameters
- `data`: Data processing parameters
- `training`: Training hyperparameters
- `evaluation`: Evaluation parameters
- `infrastructure`: Infrastructure configuration (device, logging, etc.)

### Data Loading Optimization

To handle large datasets, we provide several data loading optimization schemes:

1. **Lazy Loading**: Process data chunks on demand to reduce initial memory usage
2. **Memory-Efficient Loader**: Use LRU cache mechanism to balance memory usage and performance
3. **Preprocessing Script**: Pre-process data to avoid repeated calculations during training

Detailed information can be found in the `数据加载优化方案.md` file.

## Technical Features

This project implements the following key technologies:

1. **ContinualGNN**: Continual learning mechanism to prevent catastrophic forgetting
2. **StreamE**: Streaming graph construction and incremental updates
3. **Dual-Perspective Knowledge Consolidation**: Combines data replay and model regularization
4. **Hierarchical-Importance Sampling**: Optimizes memory buffer update strategy
5. **Approximate Influence Detection**: Efficiently detects the impact of document changes on the knowledge graph

## License

This project is open-sourced under the MIT License.