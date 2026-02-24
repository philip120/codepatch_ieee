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
