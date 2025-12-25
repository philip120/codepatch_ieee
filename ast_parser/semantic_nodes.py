from dataclasses import dataclass
from ast_utils import reconstruct_code_from_ast

@dataclass
class SemanticNode:
    id: int
    text: str

def extract_flat_semantic_nodes(ast) -> list[SemanticNode]:
    """
    Walks the AST and extracts a flat list of semantic code units.
    """
    nodes = []
    next_id = 0

    def walk(node):
        nonlocal next_id

        if not isinstance(node, tuple):
            return

        kind = node[0]

        # --- atomic statements ---
        if kind in ("statement", "command", "return"):
            text = reconstruct_code_from_ast(node).strip()
            if text:
                nodes.append(SemanticNode(next_id, text))
                next_id += 1
            return

        # --- function ---
        if kind == "func_def":
            fields = dict(node[1])
            header = fields.get("function")
            body = fields.get("body", ())

            header_text = "function " + reconstruct_code_from_ast(header).strip()
            nodes.append(SemanticNode(next_id, header_text))
            next_id += 1

            for stmt in body:
                walk(stmt)
            return

        # --- for loop ---
        if kind == "for_loop":
            fields = dict(node[1])
            header = fields.get("for")
            body = fields.get("body", ())

            header_text = "for " + reconstruct_code_from_ast(header).strip()
            nodes.append(SemanticNode(next_id, header_text))
            next_id += 1

            for stmt in body:
                walk(stmt)
            return

        # --- if block ---
        if kind == "if_block":
            items = node[1]
            i = 0
            while i < len(items):
                tag, payload = items[i]

                if tag in ("if", "elseif"):
                    cond_text = tag + " " + reconstruct_code_from_ast(payload).strip()
                    nodes.append(SemanticNode(next_id, cond_text))
                    next_id += 1

                    if i + 1 < len(items) and items[i + 1][0] == "body":
                        for stmt in items[i + 1][1]:
                            walk(stmt)
                        i += 2
                    else:
                        i += 1
                    continue

                if tag == "else_body":
                    nodes.append(SemanticNode(next_id, "else"))
                    next_id += 1
                    for stmt in payload:
                        walk(stmt)
                    i += 1
                    continue

                i += 1
            return

        # --- generic fallback ---
        text = reconstruct_code_from_ast(node).strip()
        if text:
            nodes.append(SemanticNode(next_id, text))
            next_id += 1

    # AST root is ('code_block', (...))
    if ast and ast[0] == "code_block":
        for stmt in ast[1]:
            walk(stmt)

    return nodes


from semantic_nodes import SemanticNode, extract_flat_semantic_nodes
from block_splitter import split_top_level_blocks

def extract_semantic_nodes_with_fallback(code, parser, lexer):
    nodes = []
    next_id = 0

    blocks = split_top_level_blocks(code)

    for block in blocks:
        try:
            ast = parser.parse(lexer.tokenize(block))
            block_nodes = extract_flat_semantic_nodes(ast)

            if not block_nodes:
                raise ValueError("Empty block parse")

            for n in block_nodes:
                nodes.append(SemanticNode(next_id, n.text))
                next_id += 1

        except Exception:
            # block-level fallback (NOT file-level)
            nodes.append(SemanticNode(next_id, block.strip()))
            next_id += 1

    return nodes
