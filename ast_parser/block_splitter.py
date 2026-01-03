from ast_parser.matlab_lexer import MatlabLexer

def split_top_level_blocks(code: str) -> list[str]:
    """
    Splits MATLAB code into top-level blocks:
    - function ... end (or implicit end)
    - top-level script code
    
    Uses the lexer to correctly track block depth and handles implicit function endings
    based on indentation (sibling functions).
    """
    if not code.strip():
        return []
        
    lexer = MatlabLexer()
    padded_code = code + '\n'
    tokens = list(lexer.tokenize(code)) 
    
    blocks = []
    
    last_split = 0
    depth = 0
    in_function = False
    function_start_col = -1
    
    block_openers = {'IF', 'FOR', 'WHILE', 'SWITCH', 'TRY'}
    
    for token in tokens:
        if token.type == 'FUNCTION':
            # Calculate column
            line_start = padded_code.rfind('\n', 0, token.index) + 1
            col = token.index - line_start
            
            # Heuristic: If we are in a function, and we see another 'function' keyword
            # at the same or lower indentation level, assume the previous function implicitly ended.
            if in_function and col <= function_start_col:
                # Implicit close of previous function
                blocks.append(padded_code[last_split : line_start])
                last_split = line_start
                depth = 0
                in_function = False
            
            if depth == 0:
                # Top level function start
                # Flush preceding script/whitespace if any
                if line_start > last_split:
                    text = padded_code[last_split : line_start]
                    if text.strip():
                        blocks.append(text)
                
                # Start recording this function block
                # We start at line_start to capture indentation
                last_split = line_start
                in_function = True
                function_start_col = col
            
            depth += 1
            
        elif token.type in block_openers:
            depth += 1
            
        elif token.type == 'END':
            if depth > 0:
                depth -= 1
                if depth == 0 and in_function:
                    # End of top-level function found explicitly
                    end_pos = token.index + len(token.value)
                    blocks.append(padded_code[last_split : end_pos])
                    last_split = end_pos
                    in_function = False
    
    # Flush remainder
    if last_split < len(padded_code):
        text = padded_code[last_split:]
        if text.strip():
            blocks.append(text)
            
    return blocks
