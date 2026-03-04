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

### Gradient Flow Through the Tree (Backward Pass)

The RvNN is trained by the same cross-entropy loss as the Projector. After loss computation, gradients flow back to the concatenation point where the `[M+1, D]` sequence was assembled. There they split: M gradients go to the Projector, 1 gradient goes to the RvNN's global vector.

**Example tree:**

```
        function (root, index 0)
        ├── if_statement (index 1)
        │   ├── assignment (leaf, index 2)
        │   └── assignment (leaf, index 3)
        └── return (leaf, index 4)
```

**Forward (bottom-up):**

```
Step 1: Leaves (2, 3, 4) → return their pixel embeddings directly
Step 2: if_statement (1) → aggregate children [2, 3] via child_aggregator MLP
                         → combine self + child_summary via combiner
Step 3: function (0)     → aggregate children [1, 4] via child_aggregator MLP
                         → combine self + child_summary via combiner
                         → output: [1, D] global vector
```

**Backward (top-down gradient flow):**

```
Step 1: Gradient arrives at the global vector [1, D]
Step 2: Flows into function's combiner → updates combiner weights
        Splits into:
          - gradient to function's self_vec (pixel embedding 0)
          - gradient to child_summary → flows into child_aggregator
            → updates aggregator weights
            Splits into:
              - gradient to if_statement's output
              - gradient to return's pixel embedding (leaf 4)
Step 3: if_statement's gradient → flows into its combiner (same weights, more updates)
        Splits into:
          - gradient to if_statement's self_vec (pixel embedding 1)
          - gradient to its child_summary → into aggregator (same weights again)
            Splits into:
              - gradient to leaf 2's pixel embedding
              - gradient to leaf 3's pixel embedding
Step 4: All pixel embedding gradients → PixelAdapter → updates adapter weights
Step 5: → PixelEmbedder → updates depth/type embeddings
Step 6: Stops at frozen CodeBERT
```

The child_aggregator and combiner are each **one set of weights shared across all tree levels**. Every internal node that used them contributes gradients. A deep tree with many internal nodes gives these MLPs more gradient signal per sample than a shallow one.

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

```
L_t = −log P(y_t | y_{<t}, X_code, X_prompt)
```

Where `P(y_t | ...)` is the softmax probability Qwen assigned to the correct token.

### Clarification: What Is Compared Against What

The ground truth pseudocode serves **two different roles**, and neither involves comparing embeddings against each other:

1. **As input (embedded):** The ground truth is tokenized and converted to Qwen embeddings via its embedding table. These embeddings are concatenated into the input sequence so Qwen can attend over them during teacher forcing. This is how the model sees the correct context at every position.

2. **As labels (integer IDs):** The same ground truth token IDs are used as classification targets for the loss. They are **not** embedded for this purpose — they are plain integer indices.

At each position, Qwen's output head produces a vector of ~150K logits (one per word in its vocabulary). Cross-entropy loss uses the ground truth token ID as an index to look up how much probability the model assigned to the correct word:

```
Qwen output at position t:  [0.001, 0.003, ..., 0.40, ..., 0.002]  (150K probabilities)
                                                    ↑
                                          correct token index (integer)

Loss = −log(0.40)
```

There is **no embedding-vs-embedding comparison**. The loss compares a predicted probability distribution against a correct token index. The model is penalized for assigning low probability to the correct word, regardless of how "close" any embedding is to another.

### Step 6: Final Loss

Averaged over all T target positions:

```
L = (1/T) Σ_{t=1}^{T} −log P(y_t | y_{<t}, X_code, X_prompt)
```

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
The central wrapper that fuses both encoding paths and feeds the result to the decoder.

### Dual-Path Fusion Strategy
The model encodes MATLAB code through two independent paths that capture complementary information:

* **ViT Path (Sequential):** PatchEmbedder → Projector → `[M, D]`. Captures the linear reading order of the code — which operations come before or after others. Analogous to reading code line by line.
* **Tree Path (Structural):** PixelAdapter → RecursiveEncoder → `[1, D]`. Captures the hierarchical nesting — which operations are inside loops, conditionals, or function scopes. Analogous to understanding the control flow graph.

### Concatenation
The two path outputs are concatenated along the sequence dimension:

```
combined = cat([global_vector, seq_vectors], dim=0)
         = cat([1, D] , [M, D]) → [M+1, D]
```

The RvNN global vector is **prepended** (position 0), so it serves as a structural summary token that Qwen attends over first. The M sequential tokens follow, providing fine-grained local context.

### Why Concatenation (Not Addition)
Adding the vectors would force both paths to share the same positions and collapse structural and sequential information into one signal per position. Concatenation preserves both signals as separate tokens, letting Qwen's attention mechanism decide how to weight them at each decoding step.

---

## 9. The "Pixel" Metaphor

### ViT for Code
The architecture borrows its core abstraction from the **Vision Transformer (ViT)**. In a standard ViT, an image is split into a grid of patches, each patch is flattened and projected into the transformer's embedding space, and the transformer processes them as a sequence of tokens.

This project applies the same logic to source code:

| ViT (Images)               | This Project (Code)                     |
|----------------------------|-----------------------------------------|
| Raw image (H x W x 3)     | Raw MATLAB source code                  |
| Image pixels               | Semantic operations (AST nodes)         |
| Patch (16x16 pixel block)  | Group of adjacent operations            |
| Patch projection (Linear)  | Projector (Bottleneck MLP)              |
| ViT Transformer encoder    | Qwen decoder (causal attention)         |

### Why "Pixel"
Each semantic operation extracted from the AST is called a "pixel" because it is the smallest indivisible unit of the code representation — analogous to a pixel being the smallest unit of an image. The PixelEmbedder enriches each pixel with structural metadata (depth, type), just as positional embeddings in ViT encode spatial location.

### Why This Analogy Works
Code has both **local patterns** (adjacent lines often relate) and **global structure** (a return statement only makes sense in the context of the entire function). ViT's patch-based approach captures local patterns via concatenation within patches, while the transformer's attention captures global dependencies. The addition of the RvNN path adds explicit hierarchical structure that ViT cannot capture on its own.

---

## 10. Model Variants (Ablation)

Three model variants exist to isolate the contribution of each encoding path:

### Model 1: ViT-Only (`model/model.py` — `SemanticViT`)
* **Path:** PixelEmbedder → PatchEmbedder → Projector → QwenDecoder
* **Output:** `[M, D]` — only sequential patch tokens, no tree information.
* **Purpose:** Baseline to measure how well sequential reading order alone can drive pseudocode generation.

### Model 2: Tree-Only (`model2/model.py` — `StructuralModel`)
* **Path:** PixelEmbedder → PixelAdapter → RecursiveEncoder → QwenDecoder
* **Output:** `[1, D]` — a single global vector from tree aggregation.
* **Purpose:** Baseline to measure how well hierarchical structure alone can drive generation. Only one token is fed to Qwen, so all code information must be compressed into a single vector.

### Model 3: Combined (`combined_model/model.py` — `CombinedSemanticViT`)
* **Path:** Both paths run in parallel, outputs concatenated.
* **Output:** `[M+1, D]` — sequential tokens + structural global token.
* **Purpose:** The full model. Comparing against Models 1 and 2 reveals whether fusion helps and what each path contributes.

### Ablation Framing
Comparing the three variants answers:
* **Combined vs ViT-Only:** Does adding tree structure improve generation?
* **Combined vs Tree-Only:** Does adding sequential context improve generation?
* **ViT-Only vs Tree-Only:** Which signal is more valuable on its own?

---

## 11. LoRA (Low-Rank Adaptation)

### Why LoRA
Qwen3-4B has ~4 billion parameters. Fine-tuning all of them would require far more VRAM and data than available. LoRA freezes the base weights and injects small trainable matrices into specific attention layers, enabling the decoder to adapt to the code-to-pseudocode task with minimal parameters.

### How It Works
For a frozen weight matrix `W` of shape `[d, d]`, LoRA adds a low-rank decomposition:

```
output = W @ x + (B @ A) @ x
```

Where `A` is `[r, d]` and `B` is `[d, r]` with rank `r` much smaller than `d`. Only `A` and `B` are trained. This adds `2 * r * d` parameters per adapted layer instead of `d * d`.

### Configuration
* **Target Modules:** `q_proj` and `v_proj` in each attention layer. These are the query and value projections — adapting them lets the model change *what it attends to* and *what information it extracts*, which is sufficient for task adaptation. `k_proj` and `o_proj` are left frozen.
* **Target Layers:** Last 12 of 36 layers (`layers[24..35]`). Earlier layers capture general language features and are left untouched. Later layers handle task-specific generation and benefit most from adaptation.
* **Rank (r=16):** Controls the expressiveness of the low-rank update. Higher rank = more capacity but more parameters.
* **Alpha (alpha=128):** Scaling factor applied to the LoRA output. The effective update is scaled by `alpha / rank = 128 / 16 = 8x`, amplifying the LoRA signal.
* **Dropout (0.05):** Applied to LoRA layers to regularize the small adapter.
* **Total LoRA Parameters:** ~2M (12 layers x 2 projections x 2 matrices x rank 16 x dim 2560).

### Dual Learning Rate
LoRA parameters use a separate, lower learning rate (`1e-4`) than the base trainable modules (`3e-4`). This prevents the adapted decoder from changing too aggressively, preserving Qwen's pre-trained language generation ability while still learning to interpret the code pseudo-tokens.

---

## 12. Training vs Inference

### Training (Teacher Forcing)
During training, the model sees the **entire ground truth pseudocode** as input. At each position, it predicts the next token, but the input at every position is always the correct token — never the model's own prediction. All T positions are evaluated in a **single forward pass** (no sequential generation loop).

**Advantage:** Fast and stable. The model always trains on correct context.

**Disadvantage:** Exposure bias — during training, the model never encounters its own mistakes. At inference time, one wrong prediction shifts the context into territory never seen during training, potentially causing error accumulation.

### Inference (Autoregressive Generation)
During inference, the model generates one token at a time. Each predicted token is appended to the sequence and fed back as input for the next step. This continues until an end-of-sequence token is produced or `max_new_tokens` is reached.

```
Step 1: [code tokens + prompt]            → predict "check"
Step 2: [code tokens + prompt + "check"]  → predict "if"
Step 3: [... + "check" + "if"]            → predict "x"
...
```

**Sampling parameters:**
* **Temperature (0.7):** Controls randomness. Lower = more deterministic, higher = more diverse.
* **Top-p / Nucleus Sampling (0.9):** Only samples from the smallest set of tokens whose cumulative probability exceeds 0.9, filtering out low-probability noise.

---

## 13. Dataset

### Source
`philip120/matlab-nl-pseudocode-v2` hosted on Hugging Face. Each sample is a pair of:
* **code:** A MATLAB function or script.
* **nl:** A natural language pseudocode description of what the code does.

### Size and Splits
* **Total samples:** 4,431 (after filtering 9 samples with empty code or pseudocode → 4,422 usable).
* **Training set:** First 80% of samples.
* **Test set:** Last 20% of samples (`train[-20%:]`), used for post-training BLEU evaluation.

### Preprocessing
Each sample is passed through the SemanticExtractor at dataset load time (not during training), which:
1. Parses the MATLAB code via ANTLR into semantic operations.
2. Extracts texts, depths, type_ids, and tree_roots.
3. Caches the features in the dataset object so extraction is not repeated every epoch.

Samples that produce zero semantic operations after extraction are filtered out.

---

## 14. Evaluation Metrics

### BLEU Score
**Bilingual Evaluation Understudy** — measures n-gram overlap between the generated pseudocode and the ground truth reference. Computed per-sample using `nltk.translate.bleu_score.sentence_bleu` with smoothing (Method 1) to handle short sequences.

```
score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=method1)
```

* Tokenization is whitespace-based (`.split()`).
* Scores range from 0.0 (no overlap) to 1.0 (identical).
* Averaged across all evaluation samples for a single aggregate metric.

### Limitations of BLEU for This Task
* **Synonym blindness:** "check if x > 0" and "verify that x is positive" have low BLEU despite being semantically equivalent.
* **Order sensitivity:** Reordering steps that are logically equivalent will reduce the score.
* **Length penalty:** Very short or very long generations are penalized even if correct.

BLEU provides a rough directional signal (is generation quality improving?) but should not be interpreted as a measure of semantic correctness.

### Projector Diagnostics
Two additional metrics are tracked to detect **embedding collapse** (all projected vectors becoming identical):

* **proj_var:** Variance across the D dimensions of the projected vectors, averaged over samples. If this drops to near zero, the projector is collapsing to a constant output regardless of input — the model has stopped encoding meaningful differences.
* **proj_norm:** Average L2 norm of projected vectors. Tracks whether outputs are vanishing (norms → 0) or exploding (norms → infinity).

### Efficiency Profiling
Measured per-sample during evaluation:
* **Encode time:** Time to run the full encoding pipeline (extraction + CodeBERT + Projector/RvNN).
* **Generate time:** Time for Qwen's autoregressive generation loop.
* **Tokens/sec:** Generation throughput.
* **KV cache size:** Memory used by Qwen's key-value attention cache during generation.
* **Peak VRAM:** Maximum GPU memory allocated during the full forward + generate pass.

---

## 15. Train Pipeline (`train/train_pipeline.py`)
The main coordination script managing data loading, initialization, training loop, and evaluation across all model types.

### Key Features
* **Gradient Accumulation:** Simulates larger batch sizes by summing gradients over multiple steps. With batch size 1 and accumulation 8, the effective batch size is 8. Gradients are divided by the accumulation count before the optimizer step.
* **Mixed Precision (AMP):** Utilizes `torch.amp` with `float16` on CUDA. Forward passes run in half precision to halve VRAM usage and increase throughput. The GradScaler handles loss scaling to prevent underflow in float16 gradients.
* **OneCycleLR Scheduler:** Warms up the learning rate over the first 10% of training steps, then decays via a cosine curve to zero. Separate max learning rates for base parameters and LoRA parameters.

### Dual Learning Rate Groups
When LoRA is active, the optimizer has two parameter groups:
* **Base group** (Projector, RvNN, PixelEmbedder, PixelAdapter): trained at `3e-4`.
* **LoRA group** (Qwen adapter matrices): trained at `1e-4`.

This separation prevents the adapter from changing too aggressively relative to the encoder pipeline.

### Checkpointing
Saves at regular intervals (`--save_every`) and whenever a new best loss is achieved. Each checkpoint contains:
* Trainable model parameters (named, for partial loading).
* LoRA adapter state dict.
* Optimizer, scheduler, and scaler states (for exact resume).
* Current epoch, step, best loss, and loss history.

### Post-Training Evaluation
After the training loop completes, the pipeline:
1. Switches to eval mode.
2. Loads the test split (last 20% of data).
3. Generates pseudocode for `--eval_samples` samples.
4. Computes per-sample BLEU scores with smoothing.
5. Records projector diagnostics (variance, norm) and efficiency metrics.
6. Saves loss curve and BLEU histogram as PNG files.
7. Writes all metrics to a JSON file.

---

## 16. Training Dynamics

### Optimizer & Regularization
* **Optimizer:** AdamW (Adam with decoupled weight decay).
* **Weight Decay:** `0.05` — provides L2 regularization to prevent overfitting on the small MATLAB dataset.
* **Learning Rate Scheduler:** OneCycleLR with 10% warmup followed by cosine decay to zero.

### Loss Function
See Section 7 (Qwen Decoder) for the full loss computation walkthrough.

### Hyperparameters
* **Peak Learning Rates:**
    * **Base Params** (Projector, RvNN, etc.): `3e-4`
    * **LoRA Params** (LLM Adapters): `1e-4`
* **Effective Batch Size:**
    * **Batch Size:** 1
    * **Gradient Accumulation:** 8
    * **Effective Batch Size:** **8** (allows training large LLM decoders on memory-constrained hardware).
* **Mixed Precision:** Training uses **Automatic Mixed Precision (AMP)** with `float16` on CUDA devices to reduce VRAM footprint and increase throughput.

---

## 17. Two-Stage Training Strategy

### Motivation: The Unstable Target Problem

When training from scratch in a single stage, the encoder pipeline (PixelEmbedder, Projector, RvNN) and the decoder (Qwen + LoRA) must adapt simultaneously. This creates a fundamental instability: the encoder has no stable target to learn from because the decoder is also changing every step. From the encoder's perspective, the loss landscape shifts constantly — gradients point in one direction on step N and a different direction on step N+1, not because the encoder's representation was wrong, but because the decoder's expectation changed.

This parallels the challenge in multimodal training (e.g., LLaVA): if both the visual encoder and the language decoder are trained simultaneously from random weights, neither converges reliably. The standard solution is to first establish a stable decoder, then train the encoder to produce representations the stable decoder can interpret.

### Stage 1: Text-Only Decoder Fine-Tuning (`train/train_stage1.py`)

Stage 1 trains **only the Qwen decoder** on plain `(MATLAB code, pseudocode)` text pairs — no encoder, no projected embeddings.

**Input format:**

```
[tokenized MATLAB code (max 512 tokens)] + [task prompt] + [target pseudocode]
```

The MATLAB code is passed as raw text through Qwen's embedding table (not through the encoder pipeline). Labels are masked to `-100` for the code and prompt positions; loss is computed only on the pseudocode tokens — identical masking to `forward_train`.

**What Stage 1 teaches the decoder:**
* The mapping from MATLAB syntax to pseudocode phrasing.
* Task-specific vocabulary and output structure.
* How to follow the task prompt format.

After Stage 1, the LoRA weights encode a stable MATLAB→pseudocode prior. The decoder already knows what a good pseudocode output looks like given code context. This is saved to `checkpoints_stage1/best_model.pt` with the `lora_state` key.

### Stage 2: Encoder Training Against Stable Decoder (`train/train_pipeline.py`)

Stage 2 loads the Stage 1 LoRA weights into the decoder via `--stage1_checkpoint`, then trains the full encoder pipeline (Projector, RvNN, PixelEmbedder) against this stable target.

```python
# Loader block in train_pipeline.py (after enable_lora, before optimizer):
if stage1_checkpoint:
    s1_ckpt = torch.load(stage1_checkpoint, ...)
    model.decoder.load_lora_state_dict(s1_ckpt["lora_state"])
```

Because the decoder already knows how to generate pseudocode from text tokens, it provides a consistent training signal to the encoder: "produce embeddings that look like the text representations I was trained on." The encoder learns to map code structure into the same semantic space the decoder expects.

If `--resume` is also passed, the resume checkpoint runs after the Stage 1 loader and overwrites LoRA weights — correct behavior (resume wins over warm-start).

### Checkpoint Format Contract

Both stages share the same `lora_state` key and format, ensuring compatibility with `load_lora_state_dict`:

| Key | Stage 1 | Stage 2 |
|-----|---------|---------|
| `lora_state` | ✓ | ✓ |
| `model_state` | — | ✓ |
| `stage` | 1 | — |
| `model_type` | — | ✓ |
| `optimizer`, `scheduler`, `scaler` | ✓ | ✓ |
| `step`, `epoch`, `best_loss`, `loss_history` | ✓ | ✓ |

### Rank/Layer Mismatch Warning

`--lora_rank` and `--lora_layers` **must match** between Stage 1 and Stage 2. `load_lora_state_dict` uses `strict=False` and silently skips keys with mismatched shapes — a mismatch means Stage 2 starts with random LoRA weights despite passing a Stage 1 checkpoint.

### Usage

```bash
# Stage 1: fine-tune Qwen on plain text MATLAB→pseudocode
python -m train.train_stage1 \
    --epochs 5 --lora_rank 16 --lora_alpha 128 --lora_layers 12 \
    --grad_accum 4 --lr 2e-4 --save_dir checkpoints_stage1

# Stage 2: train encoder with Stage 1 warm-start decoder
python -m train.train_pipeline \
    --model combined --epochs 10 --lora --lora_rank 16 --lora_alpha 128 --lora_layers 12 \
    --grad_accum 8 --lr 3e-4 --lora_lr 1e-4 --dropout 0.05 --bottleneck 768 \
    --stage1_checkpoint checkpoints_stage1/best_model.pt \
    --save_dir checkpoints_stage2
```

### Expected Behavior
* **Stage 1:** Loss decreases from ~3.0 to <1.5 over 5 epochs on ~4K text pairs.
* **Stage 2:** On startup, prints `Loaded N LoRA tensors from Stage 1.` Generation quality (BLEU) should exceed a single-stage baseline trained for the same number of epochs.

---

## 18. Full Layer Unfreeze (Replacing LoRA)

### Motivation

LoRA restricts adaptation to low-rank updates in `q_proj` and `v_proj` only. For a task like MATLAB→pseudocode, where the input modality (projected code embeddings) is fundamentally different from what Qwen was pre-trained on (text tokens), this bottleneck limits how much the decoder can adapt its internal representations to the new input distribution. Full layer unfreeze removes this constraint.

### What Changes

Instead of injecting LoRA adapters, the last N transformer layers of Qwen are fully unfrozen — all weights (q, k, v, o projections, MLP, layer norm) are trainable. The final layer norm (`model.norm`) and the language model head (`lm_head`) are also unfrozen, as they are critical for the output token distribution.

```
Frozen:    Qwen layers 0–17  (general language understanding)
Trainable: Qwen layers 18–35 (task-specific generation)
           model.norm
           lm_head
```

For Qwen3-4B with 36 layers, unfreezing the last 18 layers adds approximately **~2 billion trainable parameters** compared to ~2M for LoRA rank 16.

### VRAM Budget (A100 40GB)

| Component | VRAM |
|-----------|------|
| Full model weights (float16) | ~8 GB |
| Encoder pipeline | ~2 GB |
| Gradients for 18 unfrozen layers | ~4 GB |
| Adam optimizer states (float32) | ~8 GB |
| Activations | ~3 GB |
| **Total** | **~25 GB** |

This fits comfortably on a 40GB A100. Unfreezing all 36 layers would push to ~36GB (tight but possible).

### Dual Learning Rate

Two separate optimizer parameter groups prevent catastrophic forgetting:

* **Encoder group** (PixelEmbedder, Projector, RvNN): `lr = 3e-4` — learns fast from scratch.
* **Qwen group** (unfrozen layers): `lr = 1e-5` — 30× lower, nudges pre-trained weights rather than overwriting them.

The encoder LR is high because these modules start from random initialization and need strong gradient signal. The Qwen LR is low because the pre-trained weights already contain valuable language knowledge — large updates would destroy this.

### Checkpoint Format

Unfrozen Qwen layer weights are saved in a separate `qwen_state` key, distinct from `model_state` (encoder params):

```python
checkpoint = {
    'model_state': {...},   # encoder params only (~50 MB)
    'qwen_state':  {...},   # unfrozen Qwen layers (~4 GB)
    'lora_state':  {},      # empty when using unfreeze
    ...
}
```

`model_state` and `qwen_state` are kept separate so intermediate step checkpoints can optionally skip `qwen_state` to save disk space, while `best_model.pt` always contains the full state.

### Inference

At inference time, `inference.py` auto-detects whether a checkpoint used unfreeze or LoRA by inspecting the checkpoint keys, and calls `unfreeze_layers()` or `enable_lora()` accordingly — no manual flag needed.

### LoRA vs Unfreeze: When to Use Which

| | LoRA | Full Unfreeze |
|--|------|--------------|
| VRAM | ~10 GB | ~25 GB |
| Trainable params | ~2M | ~2B |
| Expressiveness | Low | High |
| Risk of forgetting | Low | Medium (mitigated by low lr) |
| Best for | Limited GPU, many epochs | A100 40GB, new modality |

### Usage

```bash
python -m train.train_full \
    --s1_epochs 5 --s2_epochs 20 \
    --unfreeze_layers 18 --qwen_lr 1e-5 \
    --lr 3e-4 --s2_grad_accum 8
```

Pass `--unfreeze_layers 0` to fall back to LoRA.

---

## 19. Discovered Errors and Fixes

This section documents bugs and architectural errors discovered during training, grouped by when they were found and what was done to correct them.

---

### 19.1 Training Approach 1 — LoRA Fine-Tuning (Single-Stage)

The first training approach used a single stage: train the encoder pipeline (Projector, RvNN, PixelEmbedder) and Qwen LoRA adapters simultaneously from random initialization.

**Error 1: OneCycleLR off-by-one crash**

The scheduler was initialised with:
```python
total_steps = epochs * len(loader) // gradient_accumulation
```
Integer floor division undercounts by 1 whenever `epochs * len(loader)` is not divisible by `gradient_accumulation`. The scheduler then raised a `ValueError` ("Tried to step N+1 times") at the very last batch.

Fix: replace with `math.ceil`:
```python
import math
total_steps = math.ceil(epochs * len(loader) / gradient_accumulation)
```

**Error 2: FP16 gradient unscale crash with unfrozen Qwen layers**

When the full-unfreeze approach was introduced, `GradScaler` raised:
```
ValueError: Attempting to unscale FP16 gradients
```
The unfrozen Qwen layers were loaded in `dtype=torch.float16`, and PyTorch's AMP scaler cannot unscale gradients of FP16 parameters (it is only designed for FP32 parameters with FP16 activations).

Fix: load the entire Qwen model in `bfloat16` and disable the scaler:
```python
# QwenDecoder.__init__
self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
)

# training scripts
scaler = torch.amp.GradScaler("cuda", enabled=False)
```
`bfloat16` has the same dynamic range as `float32`, so no loss scaling is needed. The NaN-loss that followed a previous float16 load was also caused by this dtype mismatch.

**Error 3: Disk full — step checkpoints saved 18 GB each**

Step checkpoints (saved every 100 steps) included `qwen_state` (~3.6 GB, 18 unfrozen layers) and `optimizer.state_dict()` (~14 GB for Adam m/v states on 2.2B parameters). Two saves filled the Colab disk.

Fix: step checkpoints are now "lite" — encoder weights and scheduler only (~50 MB). `best_model.pt` (saved at most once per epoch when loss improves) includes `qwen_state` but skips optimizer state. Optimizer state is never saved with the unfreeze approach.

**Error 4: Gradient clipping excluded unfrozen Qwen layers**

The clip call used `model.get_trainable_parameters()`, which only returns encoder parameters plus LoRA params (via `decoder.get_lora_parameters()`). When unfreeze was active, LoRA was disabled and `get_lora_parameters()` returned an empty list — the 2.2B unfrozen Qwen parameters were never clipped.

Fix: clip across all `requires_grad` parameters:
```python
all_trainable = [p for p in model.parameters() if p.requires_grad]
torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
```

---

### 19.2 Training Approach 2 — Two-Stage with Full Unfreeze

The second approach introduced Stage 1 text-only pre-training and replaced LoRA with full layer unfreeze. The bugs above were fixed. However, a deeper architectural error was discovered by instrumenting projector output norms during training.

---

### 19.3 Critical Architectural Bug: Projector Output Scale Mismatch (45×)

**The bug**

The `Projector` ended with a `LayerNorm(out_dim)` layer:

```python
self.net = nn.Sequential(
    nn.Linear(in_dim, bottleneck_dim),
    nn.LayerNorm(bottleneck_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(bottleneck_dim, out_dim),
    nn.LayerNorm(out_dim),     # ← problem
)
```

`LayerNorm` normalises its input to zero mean and unit variance per token. For a vector of dimension D = 2560, this forces the L2 norm to be:

```
‖x‖ = sqrt(D) ≈ sqrt(2560) ≈ 50.6
```

This is a mathematical invariant — no matter what the preceding Linear layer learns, the final `LayerNorm` always outputs vectors with norm ≈ 50. The projector **cannot** learn to produce embeddings at the right scale.

**Why it matters**

Qwen3-4B's embedding table produces token embeddings with norm ≈ 1.09 (measured: mean = 1.09, std = 0.17). The projected code prefix had norm ≈ 48.6. This is a **45× mismatch**.

```
Qwen token embedding norms:  mean = 1.09  std = 0.17
Projected embedding norms:   mean = 48.6  std = 0.24
```

Diagnosed with `debug_encoder.py`:

```bash
python debug_encoder.py --checkpoint checkpoints_stage2/combined/best_model.pt
```

The consequence: Qwen's attention layers process the prefix tokens as signals 45× larger than normal text tokens. The prefix dominates the attention computation regardless of its actual semantic content — Qwen cannot distinguish a meaningful prefix signal from one that is pure noise at this scale.

**Why generation still showed some structure**

The discrimination check confirmed that different code samples do produce different projected vectors (cosine similarity 0.59–0.83). The encoder was not collapsed. However, the scale mismatch prevented the decoder from reliably using the content of those vectors.

The generation at step ~7000 showed partial structural alignment (if-else code → if-else pseudocode structure) but wrong content, which is consistent with Qwen picking up coarse shape information from the prefix while ignoring fine-grained semantics.

**The fix**

Remove the final `LayerNorm` from the projector. The last `Linear(bottleneck_dim, out_dim)` can then learn to output vectors at whatever scale is needed — including the ~1.0 norm that Qwen expects:

```python
# Before (broken):
self.net = nn.Sequential(
    nn.Linear(in_dim, bottleneck_dim),
    nn.LayerNorm(bottleneck_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(bottleneck_dim, out_dim),
    nn.LayerNorm(out_dim),    # ← removed
)

# After (fixed):
self.net = nn.Sequential(
    nn.Linear(in_dim, bottleneck_dim),
    nn.LayerNorm(bottleneck_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(bottleneck_dim, out_dim),
)
```

With Kaiming initialisation on the final `Linear(768, 2560)`, the initial output norms will be on the order of 1–5, which is compatible with Qwen's expected input scale. During training, the weights will converge to the exact scale the decoder needs.

**Note on existing checkpoints**

Any checkpoint trained with the old projector (norm ~50) cannot be resumed and used directly with the fixed projector. The projector weights in those checkpoints learned a representation in the wrong scale regime. Stage 1 (Qwen-only) checkpoints are unaffected — the fix only changes the encoder pipeline.

**What this means for the next training run**

Stage 2 must restart from scratch. Stage 1 `best_model.pt` is still valid and should be used as the `--stage1_checkpoint`.

```bash
python -m train.train_full \
    --skip_stage1 \
    --stage1_checkpoint checkpoints_stage1/best_model.pt \
    --s2_epochs 20 \
    --unfreeze_layers 18 --qwen_lr 1e-5 \
    --s2_lr 3e-4 --s2_grad_accum 8
```

**Precedent**

LLaVA-1.5 (a production vision-language model) uses a simple two-layer MLP projector with no output normalisation. The final linear layer learns to produce visual tokens at the right scale for the frozen LLM decoder. The presence of `LayerNorm` at the projector output is a documented anti-pattern in multimodal LLM architectures precisely because it prevents scale adaptation.

---

### 19.4 Second Fix Attempt: Removing LayerNorm — Norm Drift

Removing the final `LayerNorm` was the first attempted fix. The projected norm at initialisation dropped from 50 to ~28, suggesting correct initial scale. However, during training the norm drifted upward:

```
step  100:  proj_norm = 28.9
step  650:  proj_norm = 55.8
step  750:  proj_norm = 64.9  ← higher than the original run with LayerNorm
```

**Why this happens:** Without any output constraint, the optimizer is free to increase the projector output magnitude as a proxy for "relevance". Larger projected vectors dominate Qwen's residual stream more strongly, which the optimizer exploits to reduce loss regardless of content quality. There is no penalty on large norms in the loss function. The result is norm drift that recreates and exceeds the original scale problem.

Loss converged faster without LayerNorm (step 700: loss 1.4 vs ~1.7 in the original run), confirming that the higher proj_var (0.70 vs ~0.50) provided richer gradient signal — but the norm growth remained uncontrolled.

**The correct fix: LayerNorm + learned scalar**

The two failure modes require two different solutions:

| Problem | Cause | Fix |
|---------|-------|-----|
| Norm fixed at 50 (old) | `LayerNorm` pins magnitude to `sqrt(D)` | Replace fixed normalisation with learned scale |
| Norm drift to ∞ (second attempt) | No output constraint → optimizer exploits scale | Keep `LayerNorm` for direction stability |

The final architecture combines both: `LayerNorm(out_dim)` normalises direction (preventing collapse), followed by a single learned scalar `output_scale` that controls magnitude:

```python
self.net = nn.Sequential(
    nn.Linear(in_dim, bottleneck_dim),
    nn.LayerNorm(bottleneck_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(bottleneck_dim, out_dim),
    nn.LayerNorm(out_dim),        # stabilises direction
)
# Magnitude: initialised so output norm ≈ Qwen token norm (1.09)
# output_scale = 1.09 / sqrt(2560) ≈ 0.022
self.output_scale = nn.Parameter(torch.tensor(0.022))

def forward(self, x):
    return self.net(x) * self.output_scale.abs()
```

At initialisation, `proj_norm ≈ 50 × 0.022 ≈ 1.1` — matching Qwen's token norm. During training `output_scale` adapts freely via gradient descent, but the `LayerNorm` ensures that only the scalar (not the full weight matrix) adjusts the output magnitude, preventing uncontrolled drift.

---

### 19.5 Third Fix Attempt: LayerNorm + output_scale → Gradient Dampening, Representation Collapse

The LayerNorm + learned scalar approach appeared correct in theory but failed in practice due to gradient dampening.

**The failure**

After ~300 training steps, `proj_var` (variance across the projected token dimensions) collapsed to near zero:

```
step 310:  proj_var = 0.000077,  proj_norm = 1.22
step 350:  proj_var = 0.000081,  proj_norm = 1.24
```

The projector was outputting nearly identical vectors for all code samples — all semantic content was lost. Generation degraded to repetitive filler text regardless of input.

**Why this happens: gradient dampening**

The `output_scale` is initialised to 0.022 to match Qwen's token norm. This scalar sits at the very end of the forward pass. By the chain rule, every gradient flowing back through the projector is multiplied by `output_scale`:

```
∂L/∂W_projector = (∂L/∂output) × output_scale × (∂output/∂W_projector)
                ≈ grad × 0.022 × ...
```

The projector weight matrix receives gradients 45× smaller than they would be without the scalar. This is too small for the optimizer to make meaningful updates. The projector stagnates, all outputs converge toward the same direction (the LayerNorm mean), and `proj_var → 0`.

The scalar itself (`output_scale`) does receive a useful gradient and adapts, but it is a single number — it can only control the global magnitude, not the content of the projected embeddings.

**Summary of three failed fixes**

| Attempt | Change | Initial norm | Problem |
|---------|--------|--------------|---------|
| Original | `LayerNorm(out_dim)` at end | ≈ 50 | 45× too large, decoder ignores content |
| 2nd | Remove final `LayerNorm` | ≈ 28 | Drifts to 65 by step 750 |
| 3rd | `LayerNorm` + `output_scale=0.022` | ≈ 1.1 | Gradient dampened 45×, proj_var → 0 |

All three approaches tried to fix the same symptom (wrong output norm) while keeping the bottleneck MLP architecture. The root cause is the MLP itself: its depth and the LayerNorm interaction create constraints that cannot be cleanly resolved.

---

### 19.6 Final Fix: Single Linear Projector

**Insight from earlier work**

An earlier prototype (`codepatch-paligemma`) used a single `Linear(768, 2048)` projector — one per semantic statement, no bottleneck, no LayerNorm. That model trained stably. The key difference: a single linear layer with Kaiming init produces a naturally correct output norm with no additional machinery.

**The fix**

Replace the entire bottleneck MLP with a single linear layer in all projection points:

*`shared/projector.py`:*
```python
# Before (bottleneck MLP — all three fixes failed):
self.net = nn.Sequential(
    nn.Linear(in_dim, bottleneck_dim),
    nn.LayerNorm(bottleneck_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(bottleneck_dim, out_dim),
    # ± LayerNorm(out_dim), ± output_scale
)

# After (single linear):
self.net = nn.Linear(in_dim, out_dim)   # 3072 → 2560
```

*`combined_model/model.py` — pixel_adapter (tree path):*
```python
# Before:
self.pixel_adapter = nn.Sequential(
    nn.Linear(768, qwen_dim),
    nn.LayerNorm(qwen_dim),   # ← same norm≈50 problem
    nn.GELU()
)

# After:
self.pixel_adapter = nn.Linear(768, qwen_dim)
```

*`model2/recursive_encoder.py` — child_aggregator and combiner:*
```python
# Before: child_aggregator ended with nn.LayerNorm(embed_dim)
# After:  removed — last layer is nn.Linear(hidden_dim, embed_dim)

# Before: combiner = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim))
# After:  combiner = nn.Linear(embed_dim * 2, embed_dim)
```

**Why the tree path needed the same fix**

With the projector fixed to norm ≈ 1.29, leaving the tree path's `pixel_adapter + RecursiveEncoder` with their final `LayerNorm`s would produce a `global_vector` still at norm ≈ 50. The combined tensor `[global_vector, seq_vectors]` would have a 40× norm imbalance between the structural token and the sequential tokens, causing Qwen to weight the structural token almost exclusively.

**Why Kaiming init produces the correct norm**

For `Linear(3072, 2560)` with Kaiming uniform init and unit-normal input:
```
output std ≈ sqrt(2 / fan_in) = sqrt(2 / 3072) ≈ 0.026  (per element)
output norm ≈ sqrt(out_dim) × 0.026 × sqrt(3072/3072)
            = sqrt(2560 × 2 / 3072) ≈ 1.29
```
This is within 20% of Qwen's token norm (1.09) at initialisation — no scaling, no normalisation, no additional parameters needed. During training the weights adapt freely to whatever scale the decoder requires.

For `Linear(768, 2560)` (pixel_adapter):
```
output norm ≈ sqrt(2560 × 2 / 768) ≈ 2.58
```
Still within 3× of 1.09 — acceptable initial condition, adapts during training.

**Precedent**

LLaVA-1.5 uses a two-layer MLP projector with no output normalisation; LLaVA-1.0 uses a single linear projector. Neither uses output LayerNorm. The absence of output normalisation in production multimodal projectors reflects the same lesson: the final linear layer should learn the correct output scale directly.

**Novel contribution preserved**

The single linear change affects only the internal projector implementation. The ViT-for-code patching concept (PatchEmbedder grouping N semantic pixels into M patches, then projecting M patches to Qwen space) remains intact and architecturally distinct from the earlier per-statement encoding approach.

---

### 19.7 Root Cause Analysis: Why Earlier Prototype Worked, Why Current Does Not

After comparing the current codebase against the earlier `codepatch-paligemma` prototype, the following table summarises the architectural differences:

| | codepatch-paligemma (worked) | Current (broken) |
|--|------------------------------|-----------------|
| LLM | Gemma-2B (2048-dim, multimodal-friendly) | Qwen3-4B (2560-dim, text-only) |
| Encoder LR | **2e-5** | 3e-4 (15× higher) |
| LoRA LR | **2e-5** | 1e-4 (5× higher) |
| LoRA rank / alpha | rank=8, alpha=32 | rank=16, alpha=128 |
| Normalization | **None** | F.normalize → collapse |
| Patch strategy | 1 token per semantic patch (CLS) | 4 statements concatenated → 1 token |
| Two-stage training | No — single stage | Yes (but Stage 1 was dropped in latest run) |
| proj_var collapse | Did not occur | Occurred in every normalized run |

**The two reasons old worked and new does not:**

1. **Learning rate 2e-5 prevented norm drift.** With LR=3e-4 (15× higher), the projector weights update aggressively every step. Without any output constraint the norms explode to 200+ within 750 steps. With LR=2e-5 the drift is ~15× slower — norms stay in the 5–20 range over the same number of steps, which is manageable without any normalization at all.

2. **No normalization, no collapse.** Every attempt to constrain the output norm introduced a new failure:
   - `LayerNorm` at output → pins norm to 50 (45× mismatch)
   - `LayerNorm` + learned scalar → gradient dampened 45× → `proj_var → 0`
   - Max-norm clamp at 3.0 → same dampening, same collapse
   - `F.normalize` (unit norm) → weight decay pushes W→0 → all outputs converge to bias direction → `proj_var → 0.00005`

   The old prototype used **no normalization and low LR**. The slow weight updates meant norms never grew fast enough to become a problem, so the linear layer learned useful directions naturally.

3. **Per-statement encoding avoids within-sample collapse.** With patch_size=4, four adjacent statements from the same function are concatenated (3072-dim). Adjacent statements are semantically similar in CodeBERT space → similar 3072-dim vectors → similar projector outputs → within-sample `proj_var ≈ 0`. With patch_size=1 each statement gets its own token, and different statement types (`clear all` vs `for i=1:N` vs `x = sin(theta)`) produce genuinely different CodeBERT embeddings → more diverse projector outputs.

---

### 19.8 Final Training Configuration: Matching the Prototype

**Changes made:**

1. `shared/projector.py`: Removed `F.normalize` — forward pass is now `return self.net(x)` (plain linear).
2. `train/train_pipeline.py`: Encoder weight_decay already set to 0.0 (from Section 19.7 fix). Kept.
3. Command-line: LR lowered to 2e-5 for both encoder and LoRA, rank=8, alpha=32, grad_accum=4, patch_size=1 — matching the prototype's hyperparameters while keeping Qwen3-4B.

**Training command:**

```bash
python -m train.train_pipeline \
    --model combined --epochs 20 \
    --lora --lora_rank 8 --lora_alpha 32 --lora_layers 12 --lora_lr 2e-5 \
    --grad_accum 4 --lr 2e-5 \
    --dropout 0.05 --bottleneck 768 \
    --patch_size 1 \
    --save_every 1000 \
    --save_dir checkpoints_stage2
```

**Expected behaviour:** With LR=2e-5, `proj_norm` should stay below 10 for the first 2000 steps (vs 254 with LR=3e-4). `proj_var` should be non-zero since there is no normalization to cause collapse. No Stage 1 checkpoint — single-stage training matching the prototype. The tree path (RecursiveEncoder + global_vector) is an addition not present in the prototype and constitutes the novel architectural contribution.

---

### 19.9 Post-Prototype Comparison Audit — Four Remaining Bugs

A systematic comparison of the current codebase against the `codepatch-paligemma` prototype revealed four bugs that survived all previous fix rounds. All four were fixed simultaneously.

---

**Bug 1: `F.normalize` on `global_vector` (`combined_model/model.py`)**

The combined model applied `F.normalize(global_vector, dim=-1)` before concatenating the tree path output with the sequential path output. This pinned the global_vector to unit norm (1.0), while the sequential tokens from the projector had freely-adapted norms (~1.29 at init, unconstrained during training).

This is the same class of normalization bug documented in sections 19.3–19.6. The comment next to the line said "same as projector output", but the projector output is a plain `nn.Linear` with no normalization — the comment was wrong.

Consequences:
- **Scale mismatch** between the tree token (norm=1.0) and sequential tokens (norm=free) in the `[M+1, D]` tensor fed to Qwen.
- **Gradient constraint** on the RvNN path — the recursive_encoder and pixel_adapter could not learn to produce embeddings at whatever scale Qwen requires, because `F.normalize` erased their learned magnitude every forward pass.

Fix: removed the `F.normalize` line entirely. The tree path now produces freely-scaled embeddings, matching the projector's design.

---

**Bug 2: `LayerNorm` in `model2/model.py` pixel_adapter**

The tree-only ablation model (`model2/model.py`) still used:

```python
self.pixel_adapter = nn.Sequential(
    nn.Linear(768, qwen_dim),
    nn.LayerNorm(qwen_dim),   # ← norm≈50 problem
    nn.GELU()
)
```

The combined model's pixel_adapter was correctly fixed to `nn.Linear(768, qwen_dim)` in section 19.6, but the same fix was never propagated to model2. This meant the tree-only ablation baseline still suffered from the 45× norm mismatch, making any ablation comparison (Combined vs Tree-Only) invalid.

Fix: replaced with `nn.Linear(768, qwen_dim)`, matching the combined model.

---

**Bug 3: Duplicate `num_trainable_parameters` in `combined_model/model.py`**

Two methods with the same name existed:

```python
# First (line 106): counts ALL requires_grad params
def num_trainable_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Second (line 125): counts only get_trainable_parameters()
def num_trainable_parameters(self):
    return sum(p.numel() for p in self.get_trainable_parameters())
```

Python's method resolution means the second definition silently overrides the first. `get_trainable_parameters()` excludes unfrozen Qwen layers (it only returns encoder + LoRA params), so when using `--unfreeze_layers`, the parameter count printed during training would undercount by ~2 billion parameters.

Fix: removed the first definition. The remaining method via `get_trainable_parameters()` is the one used consistently across all model variants.

---

**Bug 4: Stale comment in `train_pipeline.py`**

The unfreeze branch of the optimizer setup contained:

```python
# No weight_decay for encoder: with F.normalize in the projector,
# weight_decay pushes W→0, collapsing all outputs to bias direction.
{'params': encoder_params, 'lr': lr, 'weight_decay': 0.0},
```

The projector no longer uses `F.normalize` (removed in section 19.8). The comment was stale and misleading. The `weight_decay=0.0` for encoder params may or may not still be optimal — it was originally set to fix a bug that no longer exists — but the comment incorrectly implied the fix was still needed for a current reason.

Fix: removed the stale comment.

---

## 19.10. Changes for Current Training Run

### 19.10.1. Data Leakage Fix

The training pipeline had overlapping train/eval splits:

```python
# BEFORE (leaky)
dataset     = MatlabPseudocodeDataset(split="train")        # 100% of training data
test_dataset = MatlabPseudocodeDataset(split="train[-20%:]") # last 20% of training data
```

The evaluation set was a subset of the training set — the model was being evaluated on data it had already trained on. This inflated BLEU scores and made them unreliable as a measure of generalization.

```python
# AFTER (fixed)
dataset     = MatlabPseudocodeDataset(split=split + "[:80%]") # first 80%
test_dataset = MatlabPseudocodeDataset(split=split + "[80%:]") # last 20%, held out
```

HuggingFace dataset slicing is deterministic, so the split boundary is stable across runs.

### 19.10.2. Removal of Depth and Type Embeddings (PixelEmbedder)

The `PixelEmbedder` module added two learned embedding tables on top of the CodeBERT CLS vectors:

```python
# BEFORE
pixel_embedding = CLS + depth_embedding(depth_id) + type_embedding(type_id)
```

This was removed entirely. CLS embeddings from CodeBERT now flow directly into the downstream modules (PatchEmbedder for the ViT path, PixelAdapter for the tree path).

**Why this is likely better:**

1. **Redundant with CodeBERT.** CodeBERT already encodes the text of each semantic node — which implicitly carries type information. A `for` loop's text looks nothing like an `if` statement's text; CodeBERT's CLS vector already captures this distinction. The type embedding was re-encoding information that was already present.

2. **Depth is structural, not semantic.** The depth of a node in the AST is a property of the tree structure, not of the node's content. The tree path (RecursiveEncoder) already captures structural relationships through recursive aggregation — it knows which nodes are nested inside which. Adding depth as an additive bias to the CLS vector conflates structural position with semantic content.

3. **Small embeddings added to large vectors.** The depth and type embeddings were initialized with `std=0.02`, giving them norms of ~0.5-1.0. CodeBERT CLS vectors have norms of ~8-12. The embeddings contributed <10% of the signal and the model had to learn to either amplify them (fighting the CLS dominance) or ignore them (wasting parameters).

4. **Fewer trainable parameters.** Removing the two embedding tables (`16 × 768 + 16 × 768 = 24,576` parameters) simplifies the model. While the parameter count is small, removing them eliminates a source of noise during early training when gradients are large.

5. **Matches the working prototype.** The `codepatch-paligemma` prototype, which successfully trained, did not use depth or type embeddings. It projected image patch features directly into the decoder space without auxiliary embeddings.

**Pipeline after removal:**

```
MATLAB Code
    │
    ▼ SemanticExtractor
  [texts]
    │
    ▼ CodeBERTEncoder (frozen)
  CLS embeddings [N, 768]
    │
    ├── PatchEmbedder → Projector → [M, dec_dim]   (ViT path)
    └── PixelAdapter  → RecursiveEncoder → [1, dec_dim]  (Tree path)
    │
    ▼ cat → [M+1, dec_dim]
    │
    ▼ Decoder
  output text
```

### 19.10.3. Gemma-2B Decoder Option

Added `google/gemma-2b` as a swappable decoder alongside Qwen3-4B via a `--decoder gemma|qwen` flag. A factory pattern (`shared/decoder_factory.py`) maps the name to the corresponding decoder class.

Key differences between the two decoders:

| Property | Qwen3-4B | Gemma-2B |
|---|---|---|
| Hidden size | 2560 | 2048 |
| Layers | 36 | 18 |
| Parameters | ~3.5B | ~2B |
| Default LoRA layers | 6 | 4 |
| Default unfreeze layers | 18 | 9 |

All downstream dimensions (projector output, pixel_adapter output, recursive_encoder) derive from `decoder.hidden_size`, so switching decoders cascades automatically with no manual dimension changes.

**Why Gemma may help:**

- Smaller model = less VRAM, faster iteration, less risk of the decoder overpowering the encoder signal.
- The working `codepatch-paligemma` prototype used Gemma successfully.
- Gemma internally scales `inputs_embeds` by `sqrt(hidden_size)` (~45×), which amplifies the encoder's projected vectors. The projector learns to compensate, but this built-in scaling may help the model attend to encoder tokens more readily.

**Note:** `google/gemma-2b` is a gated model on HuggingFace. Access requires authentication via `huggingface-cli login` or setting the `HF_TOKEN` environment variable.
