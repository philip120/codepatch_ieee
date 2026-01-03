# train/dataset.py
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Parsing timed out")


class CodeNLDataset:
    def __init__(self, raw_examples, code_to_nodes_fn, timeout_sec=5):
        self.data = []
        failed = 0

        total = len(raw_examples)
        print(f"Processing {total} examples...")

        for i, ex in enumerate(raw_examples):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{total}] processed, {len(self.data)} valid, {failed} failed")

            try:
                # Set timeout for parsing (Unix only)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_sec)
                except (AttributeError, ValueError):
                    pass  # Windows doesn't support SIGALRM

                nodes = code_to_nodes_fn(ex["code"])

                try:
                    signal.alarm(0)  # Cancel timeout
                except (AttributeError, ValueError):
                    pass

                if not nodes:
                    failed += 1
                    continue

                self.data.append({
                    "code_nodes": nodes,
                    "nl_text": ex["nl"]
                })

            except Exception as e:
                failed += 1
                try:
                    signal.alarm(0)
                except (AttributeError, ValueError):
                    pass
                continue

        print(f"Done! {len(self.data)} valid examples, {failed} failed")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
