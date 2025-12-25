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
