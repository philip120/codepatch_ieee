# train/load_dataset.py
from datasets import load_dataset

def load_matlab_nl_dataset(split="train"):
    """
    Loads philip120/matlab-nl-pseudocode from Hugging Face
    and returns a list of {code, nl} dicts.
    """

    ds = load_dataset(
        "philip120/matlab-nl-pseudocode-v2",
        split=split
    )

    examples = []

    for row in ds:
        code = row.get("code")
        nl   = row.get("nl")

        if not code or not nl:
            continue

        examples.append({
            "code": code,
            "nl": nl
        })

    return examples
