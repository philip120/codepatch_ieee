# train/model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CodeBERT for code node embeddings ---
_codebert_tokenizer = None
_codebert_model = None

def _load_codebert():
    global _codebert_tokenizer, _codebert_model
    if _codebert_model is None:
        print("Loading CodeBERT...")
        _codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        _codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        _codebert_model.to(DEVICE)
        _codebert_model.eval()
    return _codebert_tokenizer, _codebert_model


def codebert_embed_nodes(nodes: list[str]) -> torch.Tensor:
    """
    Embed a list of code node strings using CodeBERT.
    Returns mean-pooled embedding of shape [768].
    """
    tokenizer, model = _load_codebert()

    if not nodes:
        return torch.zeros(768, device=DEVICE)

    # Tokenize all nodes, concatenate with [SEP]
    combined_text = " [SEP] ".join(nodes)

    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool over sequence length (excluding special tokens)
    # outputs.last_hidden_state: [1, seq_len, 768]
    embeddings = outputs.last_hidden_state[0]  # [seq_len, 768]
    pooled = embeddings.mean(dim=0)  # [768]

    return pooled


# --- Qwen for text embeddings ---
_qwen_tokenizer = None
_qwen_model = None

def _load_qwen():
    global _qwen_tokenizer, _qwen_model
    if _qwen_model is None:
        print("Loading Qwen2 (1.5B)...")
        model_name = "Qwen/Qwen2-1.5B"
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qwen_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        _qwen_model.to(DEVICE)
        _qwen_model.eval()
    return _qwen_tokenizer, _qwen_model


def qwen_embed_text(text: str) -> torch.Tensor:
    """
    Embed natural language text using Qwen2.
    Returns mean-pooled embedding of shape [1536].
    """
    tokenizer, model = _load_qwen()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool over sequence length
    embeddings = outputs.last_hidden_state[0]  # [seq_len, 1536]
    pooled = embeddings.mean(dim=0)  # [1536]

    # Convert to float32 for loss computation
    return pooled.float()


# --- Projection MLP ---
class ProjectionMLP(nn.Module):
    """
    Projects CodeBERT embeddings (768) to Qwen embedding space (1536).
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 1536, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
