from semantic_nodes import SemanticNode, extract_flat_semantic_nodes
from block_splitter import split_top_level_blocks
from ast_utils import suppress_stderr

def weak_line_fallback(block: str):
    """
    Splits the block by lines, removing comments and empty lines.
    This is a "last resort" fallback when AST parsing fails.
    """
    lines = []
    # Use splitlines(keepends=False) to get clean lines
    # Also handle multiline continuations (...)
    
    buffer = []
    
    for line in block.splitlines():
        t = line.strip()

        # drop empty lines
        if not t:
            continue

        # drop pure comments (lines starting with %)
        if t.startswith("%"):
            continue
            
        if t.endswith("..."):
             buffer.append(t)
             continue
        
        if buffer:
            buffer.append(t)
            lines.append("\n".join(buffer))
            buffer = []
        else:
            lines.append(t)
            
    if buffer:
        lines.append("\n".join(buffer))

    return lines


def extract_semantic_nodes(code: str, parser, lexer):
    from block_splitter import split_top_level_blocks

    nodes = []
    next_id = 0

    # 1. Split code into top-level blocks (Functions, Scripts)
    blocks = split_top_level_blocks(code)

    for block in blocks:
        try:
            # 2. Try to parse the block into an AST
            with suppress_stderr():
                ast = parser.parse(lexer.tokenize(block))
            
            # 3. Extract semantic nodes from AST
            subnodes = extract_flat_semantic_nodes(ast)
            
            if not subnodes:
                raise ValueError("AST extraction returned no nodes")
                
        except Exception:
            # 4. Fallback: Split by line if AST parsing fails
            lines = weak_line_fallback(block)

            # If even weak fallback gives nothing, keep raw block
            if not lines:
                raw = block.strip()
                if not raw:
                    continue
                subnodes = [SemanticNode(None, raw)]
            else:
                subnodes = [SemanticNode(None, line) for line in lines]

        for n in subnodes:
            n.id = next_id
            nodes.append(n)
            next_id += 1

    return nodes
