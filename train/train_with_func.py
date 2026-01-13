# train/train_with_func.py
"""
Semantic ViT Training - Pixel Grouping Approach

Each semantic operation = 1 pixel
Group pixels into patches (like ViT)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.semantic_adapter import code_to_nodes

# ==============================================================================
# STEP 1: SEMANTIC EXTRACTION
# ==============================================================================

# Type vocabulary - maps operation types to embedding indices
TYPE_TO_ID = {
    'function': 0,
    'if': 1,
    'elseif': 2,
    'else': 3,
    'for': 4,
    'while': 5,
    'switch': 6,
    'case': 7,
    'otherwise': 8,
    'assignment': 9,
    'call': 10,
    'return': 11,
    'break': 12,
    'continue': 13,
    'try': 14,
    'catch': 15,
}
NUM_TYPES = len(TYPE_TO_ID)
MAX_DEPTH = 16


def extract_semantic_features(code: str) -> dict:
    """
    STEP 1: Extract semantic operations from MATLAB code.

    Each operation becomes a "pixel" that will later be grouped into patches.

    Args:
        code: MATLAB source code string

    Returns:
        dict with:
            - texts: list of operation text strings (for CodeBERT)
            - depths: list of depth integers (for depth embedding)
            - type_ids: list of type embedding indices (for type embedding)
    """
    ops = code_to_nodes(code, as_objects=True)

    if not ops:
        return {'texts': [], 'depths': [], 'type_ids': []}

    texts = [op.text for op in ops]
    depths = [min(op.depth, MAX_DEPTH - 1) for op in ops]
    type_ids = [TYPE_TO_ID.get(op.type, 9) for op in ops]  # default: assignment

    return {
        'texts': texts,
        'depths': depths,
        'type_ids': type_ids,
    }


# ==============================================================================
# TEST
# ==============================================================================
if __name__ == "__main__":
    test_code = """
    function y = test(x)
        if x > 0
            y = x * 2;
            disp('positive');
        elseif x < 0
            y = -x;
        else
            y = 0;
        end
        for i = 1:10
            disp(i);
        end
    end
    """

    print("STEP 1: SEMANTIC EXTRACTION")
    print("=" * 60)

    features = extract_semantic_features(test_code)
    ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}

    print(f"\n{len(features['texts'])} pixels extracted:\n")
    for i, (text, depth, tid) in enumerate(zip(
        features['texts'], features['depths'], features['type_ids']
    )):
        print(f"  pixel[{i}]: depth={depth}, type={ID_TO_TYPE[tid]:<10}, text={text[:35]}")

    print("\n" + "=" * 60)
    print("WHAT WILL BE FED TO CODEBERT (one per pixel):")
    print("=" * 60)
    for i, text in enumerate(features['texts']):
        print(f"  CodeBERT input[{i}]: \"{text}\"")

    print("\n" + "=" * 60)
    print("WHAT WILL BE FED TO DEPTH EMBEDDING:")
    print("=" * 60)
    print(f"  depth_ids = {features['depths']}")

    print("\n" + "=" * 60)
    print("WHAT WILL BE FED TO TYPE EMBEDDING:")
    print("=" * 60)
    print(f"  type_ids = {features['type_ids']}")
    print(f"  types    = {[ID_TO_TYPE[t] for t in features['type_ids']]}")

    print("\n" + "=" * 60)
    print("Ready for STEP 2: CodeBERT embeddings")
