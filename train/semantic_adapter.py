# train/semantic_adapter.py
from ast.semantic_extractor import extract_semantic_nodes
from ast.matlab_lexer import MatlabLexer
from ast.matlab_parser import MatlabParser


lexer = MatlabLexer()
parser = MatlabParser()

def code_to_nodes(code: str):
    nodes = extract_semantic_nodes(code, parser, lexer)
    return [n.text for n in nodes]
