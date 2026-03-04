# CodePatch: A Text-Text Multimodal Architecture for Code Comprehension

This document outlines the architecture of our novel "CodePatch" model, a multimodal system designed to understand and describe MATLAB code without ever seeing the visual output.

## Acknowledgements and Core Components

Our work stands on the shoulders of giants. This project would not be possible without leveraging these fantastic open-source repositories:

-   **Base PaliGemma Architecture**: The overall structure of our model, including the implementation of the Gemma language model and the multi-modal projector, is based on the work in `hkproj/pytorch-paligemma` repository.
    -   **Source**: [https://github.com/hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma)

-   **MATLAB AST Parser**: For our advanced semantic patching strategy, we use the pure Python MATLAB parser created by `jol-jol`. This was a critical component for moving beyond simple token chunking.
    -   **Source**: [https://github.com/jol-jol/pymatlabparser](https://github.com/jol-jol/pymatlabparser)

## The Core Innovation: Replacing Vision with Code

The foundational idea of this project is to adapt a vision-language model for a code-language task. We treat snippets of code as if they were patches of an image.

To achieve this, we made one central modification to the original PaliGemma architecture:

-   **The SigLIP vision encoder was completely removed.**
-   **It was replaced with a `CodeBERT` model (`microsoft/codebert-base`)**, which acts as our "Code Encoder."

The goal is to teach the model to "see" the semantics of code in the same way the original model was taught to see the content of an image.

### The Patching Mechanism: Summarizing Code for the LLM

To efficiently process long source code files, we don't feed the raw code to the Gemma LLM. Instead, we use a patching strategy to create a short, high-level summary. We have explored two methods for this:

#### 1. Initial Approach: Fixed-Size Token Patches

Our first implementation was a simple and direct approach:

1.  The input MATLAB code is tokenized.
2.  The sequence of tokens is split into arbitrary, fixed-size chunks (e.g., 20 tokens each).
3.  Each chunk becomes a "patch."

While effective for summarization, this method lacks semantic awareness, as it can split logical constructs (like a function call) across multiple patches.

#### 2. Advanced Approach: AST-Based Semantic Patches

Our current, more sophisticated approach uses an Abstract Syntax Tree (AST) to create semantically meaningful patches.

1.  The MATLAB code string is first parsed into a full AST using `pymatlabparser`.
2.  We then traverse this tree and extract nodes that represent complete thoughts or actions (e.g., `statement` nodes).
3.  Each of these nodes is rebuilt into a clean, human-readable line of code (e.g., `plot(t, y)` or `A = -2`).
4.  This list of rebuilt statements becomes our sequence of "semantic patches."

This method is far superior as it provides the model with a list of complete, logical operations, which is a much higher-quality summary of the code's intent.

### Model Data Flow

The end-to-end data flow for training is as follows:

1.  A MATLAB script is parsed into a sequence of **semantic patches** using the AST.
2.  Each patch string is passed to the frozen **CodeBERT encoder** to produce a summary vector.
3.  The sequence of summary vectors is passed through a trainable **Multi-Modal Projector**, which aligns the vectors with Gemma's embedding space.
4.  These "code-word" embeddings are prepended to the text prompt embeddings and fed into the frozen **Gemma LLM**.
5.  The LLM generates a textual description, and the loss is calculated.
6.  Critically, backpropagation **only** updates the weights of the Multi-Modal Projector and the patch position embeddings, leaving the expert CodeBERT and Gemma models untouched. 

## Intuition

The core intuition behind CodePatch is to repurpose a proven vision-language architecture (inspired by PaliGemma) for a code-language task. Just as image patches capture visual semantics that a language model can 'understand' to generate descriptions, we treat snippets of code as 'patches' that capture programmatic semantics. By feeding these through a code-specific encoder (CodeBERT) and projecting them into the language model's (Gemma) embedding space, the model learns to 'see' what the code does—e.g., generating a plot—without ever viewing the actual visual output.

This approach leverages frozen pre-trained experts:
- **CodeBERT** summarizes code into meaningful vectors, preserving syntax and semantics.
- **Gemma** generates fluent text conditioned on these vectors.
Only a lightweight projector and position embeddings are trained, making it efficient and preventing disruption to the experts' knowledge. The result: A model that can describe plots (e.g., shape, min/max, features) directly from MATLAB code, useful for code review, documentation, or accessibility tools.

## Training Methodology: An Evolutionary Approach

Finding the optimal training strategy was an iterative and insightful process. Our final, robust methodology is implemented in `train_on_gpu.py`. Here is the evolution of our approach, which led to the final, successful model.

### Phase 1: Baseline - Projector-Only Training

Our initial approach was the simplest and most direct: freeze both the `CodeBERT` encoder and the `Gemma` LLM, and train only the multi-modal projector and the position embeddings.

-   **Rationale**: This is a very efficient way to establish a baseline. The goal is to see how well the model can perform by simply learning to "translate" the existing, generic `CodeBERT` embeddings into a format `Gemma` can understand.
-   **Outcome**: This method was stable but performance was limited. The validation loss plateaued at a relatively high value (around **~3.5**), and the generated descriptions were often generic and lacked detail.

### Phase 2: Improving Code Understanding - Fine-Tuning the Encoder

The logical next step was to improve the quality of the code representations themselves. We unfroze the `CodeBERT` encoder and trained it alongside the projector.

-   **Rationale**: By fine-tuning the code encoder, we allow it to learn task-specific embeddings. Instead of generic code vectors, it learns to produce vectors that are specifically optimized to be understood by `Gemma` for the purpose of describing plots.
-   **Outcome**: This led to a significant and steady improvement. The validation loss consistently decreased to **~3.25**, proving that tailoring the code embeddings to the task was a crucial step.

### Phase 3: The Instability of Full Fine-Tuning

The most powerful approach is to fine-tune the `Gemma` LLM itself. Our first attempt was to unfreeze the entire model and train it directly.

-   **Rationale**: Allowing the LLM to adapt its weights should, in theory, yield the highest possible performance.
-   **Outcome**: This resulted in a catastrophic failure. While the *training loss* plummeted, the model experienced **mode collapse**. It began producing incoherent, multilingual garbage, completely forgetting its ability to generate sensible text. This highlighted the extreme instability and danger of direct full fine-tuning on a specialized dataset.

### Phase 4: The Solution - Stable Fine-Tuning with PEFT/LoRA

To overcome the instability of full fine-tuning, we adopted a state-of-the-art technique: **Parameter-Efficient Fine-Tuning (PEFT)**, specifically using **Low-Rank Adaptation (LoRA)**.

-   **Rationale**: LoRA is designed to adapt large models without the risk of catastrophic forgetting. Instead of modifying all 2+ billion of Gemma's weights, we keep the original model frozen and inject tiny, trainable "adapter" layers into its architecture. We only train these adapters (in our case, only **0.03%** of the total parameters).
-   **Benefits**:
    -   **Stability**: The core language capabilities of Gemma are preserved.
    -   **Efficiency**: Training is significantly faster and requires much less VRAM.
-   **Outcome**: This was the breakthrough. The LoRA-based fine-tuning was completely stable and highly effective, allowing the validation loss to drop to its lowest point yet (**~2.4**). This produced a model that generates relevant, coherent, and mostly accurate descriptions, successfully achieving the project's goal.

### The Critical Role of Validation and Early Stopping

Throughout all phases, a proper validation workflow was essential to navigate these challenges.

-   **Train/Validation Split**: We used a 90/10 split of the dataset. The model trains on 90% and is tested against the unseen 10% after each epoch.
-   **Monitoring Validation Loss**: We learned that a low *training loss* can be misleading (as seen in Phase 3). The **validation loss** is the true indicator of a model's ability to generalize.
-   **Early Stopping**: The script was configured to save a checkpoint *only* when the validation loss improved. This ensures that the final saved model (`best_model_checkpoint.pt`) represents the model at its peak performance, not at its most overfitted state. 