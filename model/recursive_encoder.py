"""
Recursive Encoder Module (RvNN)

This module implements the recursive aggregation logic:
1. Takes a list of vectors (children)
2. Pads them to fixed size (e.g., 8)
3. Passes them through a shared MLP to get a single vector
4. Can be used recursively for trees of any depth.
"""
import torch
import torch.nn as nn

class RecursiveEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1536,  # Input/Output dimension
        max_branching: int = 8, # Max children to aggregate
        hidden_dim: int = 3072, # Internal MLP dimension
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_branching = max_branching
        
        # Input dimension is flattened children
        self.input_dim = max_branching * embed_dim
        
        # Shared MLP for aggregation
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, children_vectors: torch.Tensor) -> torch.Tensor:
        """
        Aggregate a batch of children groups.
        
        Args:
            children_vectors: [batch_size, num_children, embed_dim]
                             (num_children must be <= max_branching)
                             
        Returns:
            aggregated_vector: [batch_size, embed_dim]
        """
        B, N, D = children_vectors.shape
        device = children_vectors.device
        
        # 1. Pad to max_branching if necessary
        if N < self.max_branching:
            padding = torch.zeros(
                B, self.max_branching - N, D, 
                device=device, dtype=children_vectors.dtype
            )
            padded_input = torch.cat([children_vectors, padding], dim=1)
        elif N > self.max_branching:
            # Truncate if too many children (naive strategy)
            padded_input = children_vectors[:, :self.max_branching, :]
        else:
            padded_input = children_vectors
            
        # 2. Flatten: [B, 8, 1536] -> [B, 8*1536]
        flattened = padded_input.reshape(B, -1)
        
        # 3. Pass through MLP
        output = self.mlp(flattened)
        
        return output

    def aggregate_tree(self, node_embeddings: list) -> torch.Tensor:
        """
        Helper for recursive aggregation of a list of embeddings.
        Currently assumes batch_size=1 (inference mode/simple training).
        
        Args:
            node_embeddings: List of tensors [1, D], one per child node
            
        Returns:
            Single tensor [1, D]
        """
        if not node_embeddings:
            # Return zero vector if empty block
            return torch.zeros(1, self.embed_dim, device=self.mlp[0].weight.device)
            
        # Stack children: [1, N, D]
        children_tensor = torch.stack(node_embeddings, dim=1)
        
        # Forward pass
        return self(children_tensor)
