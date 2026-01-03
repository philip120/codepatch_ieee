# train/semantic_adapter.py
from ast_parser.semantic_extractor import extract_semantic_nodes
from ast_parser.matlab_lexer import MatlabLexer
from ast_parser.matlab_parser import MatlabParser


lexer = MatlabLexer()
parser = MatlabParser()

def code_to_nodes(code: str):
    nodes = extract_semantic_nodes(code, parser, lexer)
    return [n.text for n in nodes]
