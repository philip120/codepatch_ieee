# train/inference_patch.py
"""
Inference for Patch-based Model
"""
import sys
import os
import torch
from train.train_patch import PatchModel, DEVICE
from train.semantic_adapter import code_to_nodes

def load_model(checkpoint_path: str = "patch_best.pt", patch_size: int = 4, max_nodes: int = 64):
    print("Loading model...")
    model = PatchModel(patch_size=patch_size, max_nodes=max_nodes).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.projector.load_state_dict(checkpoint['projector_state_dict'])
    model.eval()
    return model

def generate(code: str, model: PatchModel):
    nodes = code_to_nodes(code)
    if not nodes:
        return "Error: No nodes"
        
    print(f"Extracted {len(nodes)} nodes")
    return model.generate(nodes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="patch_best.pt")
    parser.add_argument("--code", type=str)
    parser.add_argument("--patch_size", type=int, default=4)
    args = parser.parse_args()
    
    model = load_model(args.checkpoint, args.patch_size)
    if args.code:
        print(generate(args.code, model))
    else:
        # Interactive
        while True:
            code = input("Enter code (or 'quit'): ")
            if code == 'quit': break
            print(generate(code, model))

