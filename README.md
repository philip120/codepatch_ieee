# CodePatch

Code for "CodePatch: A Comparative Study of ViT-Inspired and Tree-Structural Code Representations for Pseudocode Generation."

## Structure

```
├── model/                  # Model Variant 2: ViT-only
├── model2/                 # Model Variant (Tree/RvNN)
│   ├── recursive_encoder.py    # RvNN (shared by tree variants)
│   └── semantic_extractor.py   # AST → tree structure
├── combined_model/         # Model Variant 3: Combined (Tree + ViT)
├── tree_text_model/        # Model Variant 4: Tree + Text
├── shared/                 # Shared components
│   ├── codebert_encoder.py     # CodeBERT (frozen)
│   ├── qwen_decoder.py         # Qwen3-4B decoder
│   ├── projector.py            # Linear projector (768 → 2560)
│   └── patch_embedder.py       # ViT-style patch grouping
├── train/                  # Training and evaluation
│   ├── train_full.py           # Two-stage training (main entry)
│   ├── train_stage1.py         # Stage 1: decoder fine-tuning
│   ├── train_pipeline.py       # Stage 2: encoder/projector training
│   ├── evaluate.py             # Unified evaluation (all variants)
│   ├── inference.py            # Single-sample inference
│   ├── semantic_adapter.py     # ANTLR-based MATLAB parser
│   ├── load_dataset.py         # HuggingFace dataset loader
│   └── matlab_dataset.py       # PyTorch Dataset wrapper
└── grammars-v4/matlab/     # ANTLR4 MATLAB grammar
```

## Setup

```bash
pip install -r train/requirements.txt
```

Requires `antlr4-python3-runtime==4.13.1`.

## Training

### Full two-stage training (recommended)

```bash
python -m train.train_full --s2_model tree_text
```

Runs Stage 1 (decoder fine-tuning) then Stage 2 (encoder training) back-to-back. Default: 5 Stage 1 epochs, 10 Stage 2 epochs.

### Stage 1 only: Decoder fine-tuning (Text-only baseline)

```bash
python -m train.train_stage1 \
    --epochs 5 --lr 2e-4 --unfreeze_layers 18 \
    --save_dir checkpoints_stage1
```

### Stage 2 only: Encoder/projector training

Requires an existing Stage 1 checkpoint (`--skip_stage1`):

```bash
# Tree+Text (best variant)
python -m train.train_full --skip_stage1 \
    --stage1_checkpoint checkpoints_stage1/best_model.pt \
    --s2_model tree_text --s2_epochs 10

# ViT-only
python -m train.train_full --skip_stage1 \
    --stage1_checkpoint checkpoints_stage1/best_model.pt \
    --s2_model vit --s2_epochs 10 --patch_size 1

# Combined (Tree + ViT)
python -m train.train_full --skip_stage1 \
    --stage1_checkpoint checkpoints_stage1/best_model.pt \
    --s2_model combined --s2_epochs 10 --patch_size 1
```

## Evaluation

```bash
python -m train.evaluate \
    --model_type tree_text \
    --checkpoint checkpoints/tree_text_final.pt \
    --num_samples 500 --max_tokens 512 \
    --output_path results/eval_tree_text.json
```

Model types: `stage1`, `vit`, `combined`, `tree_text`.

## Dataset

MATLAB → pseudocode pairs. Dataset will be made available upon acceptance.
