import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ----------------------------
# 1) One tiny training example
# ----------------------------
EXAMPLE = {
    "code_nodes": [
        "for n = 1:delx + 1",
        "if steep",
        "myline(n) = mymat(x_n, y_n);",
        "else",
        "myline(n) = mymat(y_n, x_n);",
        "end",
    ],
    "nl_text": "Iterate over pixels and sample values depending on line steepness.",
}

# ----------------------------
# 2) Load frozen CodeBERT
# ----------------------------
code_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(DEVICE)
codebert.eval()
for p in codebert.parameters():
    p.requires_grad = False

# ----------------------------
# 3) Load frozen Qwen (embeddings only)
# ----------------------------
# If you don't have Qwen locally / it's too big, start with a smaller Qwen model.
QWEN_ID = "Qwen/Qwen2.5-1.5B"  # safer first; upgrade later
qwen_tok = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
qwen = AutoModelForCausalLM.from_pretrained(
    QWEN_ID,
    trust_remote_code=True,
    torch_dtype=DTYPE,
).to(DEVICE)
qwen.eval()
for p in qwen.parameters():
    p.requires_grad = False

# ----------------------------
# 4) Helpers
# ----------------------------
@torch.no_grad()
def embed_code_node(text: str) -> torch.Tensor:
    inp = code_tok(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    out = codebert(**inp)
    cls = out.last_hidden_state[:, 0, :]  # [1, 768]
    return cls

def embed_code_nodes_mean(nodes: list[str]) -> torch.Tensor:
    embs = [embed_code_node(t) for t in nodes]      # list of [1,768]
    H = torch.cat(embs, dim=0)                      # [N,768]
    return H.mean(dim=0)                            # [768]

@torch.no_grad()
def embed_nl_mean(text: str) -> torch.Tensor:
    inp = qwen_tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    token_emb = qwen.model.embed_tokens(inp.input_ids)  # [1,T,D]
    return token_emb.mean(dim=1).squeeze(0)             # [D]

# ----------------------------
# 5) Build MLP (only trainable part)
# ----------------------------
z_nl = embed_nl_mean(EXAMPLE["nl_text"])
D_QWEN = z_nl.shape[0]

mlp = nn.Sequential(
    nn.Linear(768, 1024),
    nn.GELU(),
    nn.Linear(1024, 1024),
    nn.GELU(),
    nn.Linear(1024, D_QWEN),
).to(DEVICE)

opt = torch.optim.AdamW(mlp.parameters(), lr=1e-4)

# ----------------------------
# 6) Single training step
# ----------------------------
mlp.train()

z_code = embed_code_nodes_mean(EXAMPLE["code_nodes"])   # [768]
z_pred = mlp(z_code)                                    # [D_QWEN]

with torch.no_grad():
    z_target = embed_nl_mean(EXAMPLE["nl_text"])         # [D_QWEN]

loss = 1.0 - F.cosine_similarity(z_pred.unsqueeze(0), z_target.unsqueeze(0)).mean()

loss.backward()
opt.step()
opt.zero_grad()

print("DEVICE:", DEVICE)
print("z_code:", tuple(z_code.shape))
print("z_target:", tuple(z_target.shape))
print("loss:", float(loss.item()))
