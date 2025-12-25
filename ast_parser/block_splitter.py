import re

def split_top_level_blocks(code: str):
    """
    Splits MATLAB code into top-level blocks:
    - function ... end
    - remaining script code
    """
    lines = code.splitlines(keepends=True)
    blocks = []

    current = []
    depth = 0
    in_function = False

    for line in lines:
        stripped = line.strip()

        # function start (only at top level)
        if stripped.startswith("function") and depth == 0:
            if current:
                blocks.append("".join(current))
                current = []
            in_function = True
            depth = 1
            current.append(line)
            continue

        if in_function:
            current.append(line)

            if re.match(r'^\s*function\b', stripped):
                depth += 1
            elif re.match(r'^\s*end\b', stripped):
                depth -= 1
                if depth == 0:
                    blocks.append("".join(current))
                    current = []
                    in_function = False
            continue

        # outside functions
        current.append(line)

    if current:
        blocks.append("".join(current))

    return [b for b in blocks if b.strip()]
