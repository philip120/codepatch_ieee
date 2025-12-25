# train/test_dataset.py
from load_dataset import load_matlab_nl_dataset
from semantic_adapter import code_to_nodes
from dataset import CodeNLDataset

raw = load_matlab_nl_dataset(split="train")
dataset = CodeNLDataset(raw[:3], code_to_nodes)

for i in range(len(dataset)):
    ex = dataset[i]
    print("----")
    print("NODES:")
    for n in ex["code_nodes"][:5]:
        print(" ", n)
    print("NL:")
    print(ex["nl_text"][:200])
