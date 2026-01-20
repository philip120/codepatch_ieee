# CodePatch: Semantic ViT for MATLAB-to-Pseudocode Generation

A vision-transformer-inspired architecture for converting MATLAB code to natural language pseudocode.

## Project Structure

```
├── model/                  # Model components (pipeline steps 1-6)
│   ├── semantic_extractor.py   # Step 1: MATLAB → semantic operations
│   ├── codebert_encoder.py     # Step 2: CodeBERT embeddings (frozen)
│   ├── pixel_embedder.py       # Step 3: Depth + type embeddings (trainable)
│   ├── patch_embedder.py       # Step 4: Group pixels into patches
│   ├── projector.py            # Step 5: Bottleneck MLP (trainable)
│   └── qwen_decoder.py         # Step 6: Qwen text generation (frozen)
│
├── train/                  # Training and inference scripts
│   ├── train_pipeline.py       # Main training script
│   ├── train_with_func.py      # Pipeline testing/debugging
│   ├── inference.py            # Run inference with trained model
│   ├── semantic_adapter.py     # ANTLR-based MATLAB parser
│   ├── load_dataset.py         # Hugging Face dataset loader
│   └── matlab_dataset.py       # PyTorch Dataset wrapper
│
├── grammars-v4/matlab/     # ANTLR grammar for MATLAB parsing
│   ├── MATLAB.g4               # Grammar definition
│   ├── matlabLexer.py          # Compiled lexer (Python)
│   ├── matlabParser.py         # Compiled parser (Python)
│   └── ...
│
└── checkpoints/            # Saved model checkpoints
```

## Setup

### 1. Install Dependencies

```bash
pip install -r train/requirements.txt
```

### 2. ANTLR Grammar (if recompiling)

The MATLAB grammar is pre-compiled in `grammars-v4/matlab/`. If you need to recompile:

```bash
# Install ANTLR
# macOS
brew install antlr

# Ubuntu
sudo apt-get install antlr4

# Or download JAR
wget https://www.antlr.org/download/antlr-4.13.1-complete.jar
```

Compile the grammar:
```bash
cd grammars-v4/matlab

# Using installed antlr
antlr4 -Dlanguage=Python3 MATLAB.g4

# Or using JAR
java -jar /path/to/antlr-4.13.1-complete.jar -Dlanguage=Python3 MATLAB.g4
```

**Important:** The `antlr4-python3-runtime` version must match the ANTLR version used to compile the grammar.

### 3. Google Colab Setup

```python
# Install dependencies
!pip install torch transformers datasets antlr4-python3-runtime==4.13.1

# Clone repository
!git clone https://github.com/your-username/codepatch_ieee.git
%cd codepatch_ieee

# If grammar needs recompiling
!apt-get install -y default-jre
!wget https://www.antlr.org/download/antlr-4.13.1-complete.jar
!cd grammars-v4/matlab && java -jar /content/antlr-4.13.1-complete.jar -Dlanguage=Python3 MATLAB.g4
```

## Usage

### Training

```bash
python -m train.train_pipeline \
    --epochs 10 \
    --lr 1e-4 \
    --patch_size 4 \
    --bottleneck 512 \
    --dropout 0.4
```

### Inference

```bash
# Evaluate on dataset
python -m train.inference --checkpoint checkpoints/best_model.pt --eval --num_samples 10

# Single code snippet
python -m train.inference --checkpoint checkpoints/best_model.pt --code "function y = f(x); y = x*2; end"

# Interactive mode
python -m train.inference --checkpoint checkpoints/best_model.pt --interactive
```


## Architecture Overview

```
MATLAB Code
    │
    ▼ SemanticExtractor (ANTLR parser)
[semantic operations with depth & type]
    │
    ▼ CodeBERTEncoder (frozen)
[CLS embeddings per operation]
    │
    ▼ PixelEmbedder (trainable)
[CLS + depth_emb + type_emb]
    │
    ▼ PatchEmbedder
[group pixels into patches]
    │
    ▼ Projector (trainable, bottleneck MLP)
[project to Qwen embedding space]
    │
    ▼ QwenDecoder (frozen)
[generate pseudocode]
```

## Dataset

Uses `philip120/matlab-nl-pseudocode` from Hugging Face.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| patch_size | 4 | Pixels per patch |
| bottleneck | 512 | MLP bottleneck dimension |
| dropout | 0.4 | Dropout rate |
| lr | 1e-4 | Learning rate |
| weight_decay | 0.05 | AdamW weight decay |
| grad_accum | 4 | Gradient accumulation steps |
