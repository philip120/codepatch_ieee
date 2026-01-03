# train/semantic_adapter.py
import sys
import os

# Add ast_parser to path so its internal imports work
_ast_parser_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ast_parser")
if _ast_parser_dir not in sys.path:
    sys.path.insert(0, _ast_parser_dir)

# Suppress SLY parser warnings during import
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()

from semantic_extractor import extract_semantic_nodes
from matlab_lexer import MatlabLexer
from matlab_parser import MatlabParser

lexer = MatlabLexer()
parser = MatlabParser()

sys.stderr = _stderr  # Restore stderr


def code_to_nodes(code: str):
    """Convert MATLAB code to semantic node strings."""
    try:
        nodes = extract_semantic_nodes(code, parser, lexer)
        return [n.text for n in nodes]
    except Exception:
        # Fallback: simple line split
        lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('%')]
        return lines if lines else None
