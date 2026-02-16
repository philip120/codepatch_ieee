# Architecture Documentation

---

## 1. Semantic Extractor (`model2/semantic_extractor.py`)
This module handles AST parsing to extract texts, depths, type IDs, and tree roots from MATLAB code.

### Data Structure: TreeNode
* **Composition:** Wraps a `SemanticNode` instance (`self.node`) rather than inheriting from it.
* **index:** Represents the original position in the list.
* **children:** A list of pointers to other `TreeNode` objects.

### Stack-Based Reconstruction (`_build_tree`)
* Iterates through a flat list of nodes.
* Maintains a stack of parent nodes.
* Assigns a node as a child if its depth is strictly greater than the stack top.
* Pops the stack to locate the correct parent if the current node's depth is smaller or equal.
* Assigns the node as a root if the stack is completely empty.

### Feature Extraction Pipeline
* **Semantic Parsing:** Calls `code_to_nodes` (via ANTLR) to identify meaningful logical units like assignments, conditions, and loops.
* **texts (Metadata):** Raw code strings for each operation, prepped for the language model.
* **depths (Metadata):** Integer values acting as structural positional encoding.
* **type_ids (Metadata):** Categorical IDs (e.g., if=1, for=4) mapped from a shared vocabulary.
* **Hierarchy Generation:** Executes the stack-based tree reconstruction logic.

### Semantic Adapter
* **Technology:** Uses ANTLR4 with a MATLAB grammar, backed by a regex `fallback_extract` to prevent pipeline crashes.
* **Semantic Nodes:** Abandons highly granular ASTs in favor of extracting high-level logical nodes.
* **Capture Rules:** Treats assignments, expressions, and jump statements as single cohesive units.
* **Structural Rules:** Uses selection statements (if/else) and iteration statements (loops) to define the hierarchy.
* **Depth Tracking:** Tracks nesting depth during the ANTLR walk (e.g., function=0, nested if=1).
* **Output:** Returns a flat list of `SemanticNode` objects containing text, logic type, and depth.

### Semantic Extractor Logic (Model 2 Specific)
* **Tree Reconstruction:** Reconstructs the logical tree hierarchy from the flat adapter output.
* **TreeNode Wrapper:** Wraps raw nodes in a class with explicit children lists, a requirement for bottom-up processing in Recursive Neural Networks (Tree-LSTMs).
* **Feature Mapping:** Maps raw node types to numerical IDs using `TYPE_TO_ID` for model embeddings.
* **Summary Flow:** Code String → Flat Nodes (Adapter) → Hierarchical Tree (Extractor) → Code Embedding (Encoder).

---

## 2. CodeBERT Encoder (`shared/codebert_encoder.py`)
* **Status:** Frozen.
* **Function:** Converts raw text strings (one per semantic operation) into `[N, 768]` CLS embeddings using the pre-trained CodeBERT model. Operates before the Pixel Embedder.

---

## 3. Pixel Embedder (`shared/pixel_embedder.py`)
Adds learnable structural metadata to each CodeBERT embedding via element-wise addition.

### Trainable Embeddings
* **Depth Embedding:** `nn.Embedding(16, 768)` — encodes AST nesting depth (e.g., top-level=0, nested if=1).
* **Type Embedding:** `nn.Embedding(16, 768)` — encodes semantic node type (e.g., assignment, loop, conditional).
* Both initialized with small normal noise (`std=0.02`) so they don't dominate the CLS signal early in training.

### Forward Pass
```python
pixel = CLS_embedding + depth_embedding(depth_id) + type_embedding(type_id)
```
* **Input:** `[N, 768]` CLS embeddings, `[N]` depth indices, `[N]` type indices.
* **Output:** `[N, 768]` pixel embeddings enriched with structural position and type information.

---

## 4. Patch Embedder (`shared/patch_embedder.py`)
A utility module inspired by the Vision Transformer (ViT) architecture. It groups fine-grained code "pixel" embeddings into larger "patches" to reduce sequence lengths for the decoder.

### The ViT for Code Concept
Instead of cutting a 2D image into squares, this views a MATLAB script as a 1D sequence of semantic pixels. It compresses multiple pixels into a single vector to make the input manageable for an LLM decoder like Qwen.

### Padding for Divisibility
Neural networks require consistent tensor shapes. The module calculates the remainder and zero-pads the sequence end to ensure the length is a perfect multiple of the patch size.

```python
remainder = N % P
if remainder != 0:
    pad_len = P - remainder
    padding = torch.zeros(pad_len, D, ...)
    pixel_embeddings = torch.cat([pixel_embeddings, padding], dim=0)
```

### Dimensional Concatenation (Flattening)
Instead of pooling or averaging (which loses data), it concatenates the pixels to preserve local context.

```python
patches = pixel_embeddings.view(num_patches, P, D)
flat_patches = patches.view(num_patches, P * D)
```

### Motivation
* **Context Window Efficiency:** Reduces sequence lengths significantly (e.g., 400 operations become 100 tokens with a patch size of 4).
* **Local Context Preservation:** Flattening keeps the exact sequence of pixels intact within the patch.
* **Dimension Matching:** Aligns the flattened output dimension to the hidden dimension of the target LLM.

---

## 5. MLP Model 1 (Projector) (`shared/projector.py`)
This module uses a two-stage transformation to compress and translate CodeBERT features into pseudo-tokens the LLM can understand, preventing rote memorization.

### Stage 1: Compression (Bottleneck)
* **Input:** Flattened patch embeddings (e.g., 3072 dimensions).
* **Transformation:** A Linear layer reduces the dimension to 512, forcing the model to discard noise.
* **Regularization:** Uses LayerNorm for stability, GELU for non-linearity, and an aggressive Dropout (0.4) to prevent overfitting.

### Stage 2: Expansion
* **Transformation:** A second Linear layer expands the bottleneck features up to 2560 (matching the Qwen3-4B-Instruct embedding dimension).
* **Final LayerNorm:** Ensures output vectors are scaled appropriately before injection.

### Training Parameters
* **in_dim (3072):** Fixed based on patch size and CodeBERT constraints.
* **bottleneck_dim (512):** Tunable hyperparameter for compression intensity.
* **out_dim (2560):** Read dynamically from `decoder.hidden_size` to match the decoder.
* **Gradient Flow:** Fully trainable; learns to map code features into LLM-compatible embeddings via backpropagation.

---

## 6. Recursive Encoder (Model 2 MLP) (`model2/recursive_encoder.py`)
Processes data bottom-up using recursive neural logic.

### Components
* **Child Aggregator (MLP):** Distills up to 8 children's embeddings into a single summary vector.
    *   **Concatenation-based Aggregation:** Instead of summing or averaging, the model concatenates up to 8 children into a "Super-Vector" of dimension \(8 \times 2560 = 20{,}480\). This allows the model to learn **order-sensitive** logic (e.g., distinguishing between the first and last child in a block).
    *   **Distillation:** A two-layer MLP (Linear → LayerNorm → GELU → Dropout → Linear) compresses this large vector back down to the model dimension (2560), distilling the structural context.
    *   **Padding:** Nodes with fewer than 8 children are zero-padded to maintain a fixed-width input window for the MLP.
* **Combiner (Linear + LayerNorm):** Merges a parent's own semantic meaning (e.g., "This is an if header") with the distilled summary of its children. It concatenates the parent vector and the child summary (\(2 \times 2560\)) and projects them back to 2560, followed by LayerNorm.

### Recursive Logic (`forward_tree`)
* **Base Case:** Leaf nodes return their own embedding.
* **Recursive Step:** Calls `forward_tree` on all children first (post-order traversal).
* **Aggregation & Combination:** Passes encoded children through the aggregator, then merges the result with the parent embedding.
* **Forest Handling:** Performs a final aggregation on all root nodes if a script contains multiple top-level blocks.

---

## 7. Qwen Decoder (`shared/qwen_decoder.py`)
Handles autoregressive generation and specific loss masking for fine-tuning.

### Notation
* **N:** Number of semantic operations extracted from the MATLAB source code.
* **M:** Number of patches after grouping N pixels with patch size P (`M = ceil(N / P)`). With the RvNN global vector prepended, the code occupies `M+1` positions.
* **P:** Number of prompt tokens (from Qwen's tokenizer applied to the task prompt string).
* **T:** Number of target tokens (from Qwen's tokenizer applied to the ground-truth pseudocode).
* **D:** Qwen hidden dimension (2560 for Qwen3-4B-Instruct).

### Step 1: Encode MATLAB Code (Trainable Pipeline)

```
MATLAB code
  → SemanticExtractor: parse into N semantic operations
  → CodeBERT (frozen): [N, 768] CLS embeddings
  → PixelEmbedder: + depth/type → [N, 768]
  → PatchEmbedder: group into patches → [M, patch_size * 768]
  → Projector: bottleneck MLP → [M, D]
  → RvNN: tree aggregation → [1, D]
  → Concatenate: [M+1, D]   ← these are the "pseudo-tokens"
```

### Step 2: Build the Input Sequence (`forward_train`)

The pseudocode target (e.g. `"1. Check if x is positive\n2. Double it"`) is tokenized by Qwen's tokenizer into token IDs, then converted to Qwen's own embeddings via its embedding table. Three segments are concatenated along the sequence dimension:

```
Position:  [0 ... M]            [M+1 ... M+P]         [M+P+1 ... M+P+T]
Content:   code pseudo-tokens    prompt embeddings      target text embeddings
Source:    our projector/RvNN    Qwen embed table       Qwen embed table
Shape:     [1, M+1, D]          [1, P, D]              [1, T, D]

Full input_embeds: [1, M+1+P+T, D]
```

### Step 3: Build the Labels (Masking)

```
Position:  [0 ... M]   [M+1 ... M+P]   [M+P+1 ... M+P+T]
Labels:    [-100 ...]   [-100 ...]       [token_id_1, token_id_2, ..., token_id_T]
```

`-100` is PyTorch's ignore index for `CrossEntropyLoss`. Only the T pseudocode token positions contribute to the loss.

```python
patch_labels  = torch.full((1, num_patches), -100, ...)
prompt_labels = torch.full((1, num_prompt), -100, ...)
target_labels = target_tokens.input_ids.clone()
labels = torch.cat([patch_labels, prompt_labels, target_labels], dim=1)
```

### Step 4: Forward Through Qwen

Qwen processes the full sequence with **causal attention** (each position can only attend to itself and earlier positions). At every position `i`, it outputs a probability distribution over the entire vocabulary (~150K tokens):

```
logits: [1, M+1+P+T, vocab_size]
```

### Step 5: Compute Cross-Entropy Loss

The loss is computed with a **shift-by-one**: the model's prediction at position `t-1` is compared against the actual token at position `t`. This is standard causal LM next-token prediction.

```
Position M+P+1: model sees [code + prompt]                 → should predict token_id_1
Position M+P+2: model sees [code + prompt + token_1]       → should predict token_id_2
Position M+P+3: model sees [code + prompt + tok_1 + tok_2] → should predict token_id_3
...
```

At each position, the loss is:

$$ L_t = -\log P(y_t \mid y_{<t},\; X_{\text{code}},\; X_{\text{prompt}}) $$

Where $P(y_t \mid \ldots)$ is the softmax probability Qwen assigned to the correct token.

### Step 6: Final Loss

Averaged over all $T$ target positions:

$$ L = \frac{1}{T} \sum_{t=1}^{T} -\log P(y_t \mid y_{<t},\; X_{\text{code}},\; X_{\text{prompt}}) $$

The pseudocode is **never encoded into a latent representation**. It is tokenized into discrete IDs and used as classification targets. The model learns: given these code embeddings as context, generate the correct pseudocode tokens.

### Backpropagation Chain

This single scalar loss backpropagates through:

1. **Qwen's output head** → loss origin.
2. **Qwen's LoRA weights** → gradients update the adapter matrices.
3. **Qwen's frozen weights** → gradients pass through without updating (computation graph is preserved).
4. **The projected code pseudo-tokens** at positions `0...M` → gradients flow into the input embeddings.
5. **Projector, RvNN, PixelEmbedder** → all trainable weights update via backpropagation.
6. **CodeBERT (frozen)** → gradient flow terminates here.

---

## 8. Combined Model (`combined_model/model.py`)
* **Purpose:** The central wrapper that links the extractors, encoders, patch embedders, and the decoder into a single runnable architecture.

---

## 9. Train Pipeline (`train/train_pipeline.py`)
The main coordination script managing data loading, initialization, loops, and evaluation across all model types.

### Key Features
* **Gradient Accumulation:** Simulates larger batch sizes by summing gradients over multiple steps.
* **Mixed Precision (AMP):** Utilizes `torch.amp` for float16 calculations, halving VRAM usage.
* **OneCycleLR Scheduler:** Warms up the learning rate and decays it via a cosine curve for stable convergence.

### LoRA Configurations
* **Base Params:** Applied to the trainable Projector/RvNN/PixelEmbedder at a higher learning rate.
* **LoRA Params:** Applied to the Qwen decoder adapter weights at a lower learning rate to preserve pre-trained knowledge.

### Evaluations & Metrics
* **BLEU Scores:** Evaluates linguistic similarity between generated output and ground truth.
* **Visualizations:** Auto-generates `loss_curve.png` and `bleu_scores.png`.
* **Profiling:** Measures tokens per second, encode times, and peak VRAM to track computational costs.

---

## 10. Training Dynamics

### Optimizer & Regularization
*   **Optimizer:** The model uses **AdamW** (Adam with Weight Decay) to optimize the trainable parameters.
*   **Weight Decay:** Set to `0.05` to provide L2 regularization and prevent overfitting on the MATLAB dataset.
*   **Learning Rate Scheduler:** A **OneCycleLR** scheduler is used, which implements a warmup period followed by a cosine decay to zero.

### Loss Function
See Section 7 (Qwen Decoder) for the full loss computation walkthrough.

### Hyperparameters
*   **Peak Learning Rates:**
    *   **Base Params** (Projector, RvNN, etc.): `3e-4`
    *   **LoRA Params** (LLM Adapters): `1e-4` (tunable independently).
*   **Effective Batch Size:**
    *   **Batch Size:** 1
    *   **Gradient Accumulation:** 8
    *   **Effective Batch Size:** **8** (allows training large LLM decoders on memory-constrained hardware).
*   **Mixed Precision:** Training uses **Automatic Mixed Precision (AMP)** with `float16` on CUDA devices to reduce VRAM footprint and increase throughput.
