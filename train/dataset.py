# train/dataset.py
class CodeNLDataset:
    def __init__(self, raw_examples, code_to_nodes_fn):
        self.data = []

        for ex in raw_examples:
            nodes = code_to_nodes_fn(ex["code"])
            if not nodes:
                continue

            self.data.append({
                "code_nodes": nodes,
                "nl_text": ex["nl"]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
