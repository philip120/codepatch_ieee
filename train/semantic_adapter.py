# train/semantic_adapter.py
"""
Semantic Adapter for MATLAB Code → Tree-Structured Semantic Operations

Extracts semantic operations from MATLAB code while PRESERVING TREE STRUCTURE.
Each operation includes its depth in the AST - this serves as position encoding.

Uses the improved MATLAB grammar with proper switch/case support.
"""
import sys
import os
import re
from dataclasses import dataclass
from typing import Optional

# Add ANTLR parser to path
_antlr_parser_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "grammars-v4", "matlab"
)
if _antlr_parser_dir not in sys.path:
    sys.path.insert(0, _antlr_parser_dir)

try:
    from antlr4 import InputStream, CommonTokenStream
    from matlabLexer import matlabLexer
    from matlabParser import matlabParser
    ANTLR_AVAILABLE = True
except ImportError:
    ANTLR_AVAILABLE = False


@dataclass
class SemanticNode:
    """A semantic operation with tree position info."""
    depth: int
    type: str
    text: str
    parent_type: Optional[str] = None

    def __repr__(self):
        indent = "  " * self.depth
        return f"{indent}[D{self.depth}] {self.type}: {self.text}"


def get_text(node) -> str:
    """Get full text from parse tree node."""
    if node is None:
        return ""
    if not hasattr(node, 'children') or node.children is None:
        return node.getText() or ""
    return "".join(get_text(child) for child in node.children)


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().rstrip(';').rstrip(',').strip()
    return text


def extract_tree_operations(tree, parser) -> list[SemanticNode]:
    """Extract semantic operations with tree depth from new grammar."""
    operations = []

    def get_rule_name(node):
        if hasattr(node, 'getRuleIndex'):
            return parser.ruleNames[node.getRuleIndex()]
        return None

    def walk(node, depth: int = 0, parent_type: str = None):
        if node is None:
            return

        rule = get_rule_name(node)
        children = list(node.children) if hasattr(node, 'children') and node.children else []

        # === FUNCTION DEFINITION ===
        if rule == 'def_function':
            # Build signature from children BEFORE first statement
            sig_parts = []
            for child in children:
                child_rule = get_rule_name(child)
                if child_rule == 'statement':
                    break  # Stop at first statement (body starts)
                if child_rule == 'end_function':
                    break
                # Collect signature tokens
                child_text = child.getText() if hasattr(child, 'getText') else ""
                if child_text and child_text != 'end':
                    sig_parts.append(child_text)

            if sig_parts:
                # Join with spaces for readability
                sig_text = ' '.join(sig_parts)
                # Clean up spacing around operators
                sig_text = re.sub(r'\s*=\s*', ' = ', sig_text)
                sig_text = re.sub(r'\s*\(\s*', '(', sig_text)
                sig_text = re.sub(r'\s*\)\s*', ')', sig_text)
                sig_text = re.sub(r'\s*,\s*', ', ', sig_text)
                operations.append(SemanticNode(
                    depth=depth,
                    type='function',
                    text=sig_text.strip(),
                    parent_type=parent_type
                ))

            # Recurse into statements inside function
            for child in children:
                child_rule = get_rule_name(child)
                if child_rule == 'statement':
                    walk(child, depth + 1, 'function')
            return

        # === IF STATEMENT ===
        if rule == 'st_if':
            for i, child in enumerate(children):
                child_text = child.getText() if hasattr(child, 'getText') else ""

                if child_text == 'if':
                    # Get condition (next child should be xpr_tree)
                    if i + 1 < len(children):
                        cond = children[i + 1]
                        if get_rule_name(cond) == 'xpr_tree':
                            cond_text = clean_text(get_text(cond))
                            operations.append(SemanticNode(
                                depth=depth,
                                type='if',
                                text=f"if {cond_text}",
                                parent_type=parent_type
                            ))

                elif child_text == 'elseif':
                    if i + 1 < len(children):
                        cond = children[i + 1]
                        if get_rule_name(cond) == 'xpr_tree':
                            cond_text = clean_text(get_text(cond))
                            operations.append(SemanticNode(
                                depth=depth,
                                type='elseif',
                                text=f"elseif {cond_text}",
                                parent_type=parent_type
                            ))

                elif child_text == 'else':
                    operations.append(SemanticNode(
                        depth=depth,
                        type='else',
                        text='else',
                        parent_type=parent_type
                    ))

                elif get_rule_name(child) == 'statement':
                    walk(child, depth + 1, 'if')
            return

        # === FOR LOOP ===
        if rule == 'st_for':
            # Extract: for var = range
            var_name = None
            range_text = None
            for i, child in enumerate(children):
                child_text = child.getText() if hasattr(child, 'getText') else ""
                child_rule = get_rule_name(child)

                if child_rule == 'atom_var':
                    var_name = child_text
                elif child_rule == 'xpr_tree' and var_name:
                    range_text = clean_text(get_text(child))
                    break

            if var_name and range_text:
                operations.append(SemanticNode(
                    depth=depth,
                    type='for',
                    text=f"for {var_name} = {range_text}",
                    parent_type=parent_type
                ))

            # Recurse into statements
            for child in children:
                if get_rule_name(child) == 'statement':
                    walk(child, depth + 1, 'for')
            return

        # === WHILE LOOP ===
        if rule == 'st_while':
            for i, child in enumerate(children):
                child_rule = get_rule_name(child)
                if child_rule == 'xpr_tree':
                    cond_text = clean_text(get_text(child))
                    operations.append(SemanticNode(
                        depth=depth,
                        type='while',
                        text=f"while {cond_text}",
                        parent_type=parent_type
                    ))
                    break

            for child in children:
                if get_rule_name(child) == 'statement':
                    walk(child, depth + 1, 'while')
            return

        # === SWITCH STATEMENT ===
        if rule == 'st_switch':
            switch_expr = None
            for i, child in enumerate(children):
                child_text = child.getText() if hasattr(child, 'getText') else ""
                child_rule = get_rule_name(child)

                if child_text == 'switch' and i + 1 < len(children):
                    expr = children[i + 1]
                    if get_rule_name(expr) == 'xpr_tree':
                        switch_expr = clean_text(get_text(expr))
                        operations.append(SemanticNode(
                            depth=depth,
                            type='switch',
                            text=f"switch {switch_expr}",
                            parent_type=parent_type
                        ))

                elif child_text == 'case' and i + 1 < len(children):
                    expr = children[i + 1]
                    if get_rule_name(expr) == 'xpr_tree':
                        case_text = clean_text(get_text(expr))
                        operations.append(SemanticNode(
                            depth=depth + 1,
                            type='case',
                            text=f"case {case_text}",
                            parent_type='switch'
                        ))

                elif child_text == 'otherwise':
                    operations.append(SemanticNode(
                        depth=depth + 1,
                        type='otherwise',
                        text='otherwise',
                        parent_type='switch'
                    ))

                elif child_rule == 'statement':
                    walk(child, depth + 2, 'case')
            return

        # === TRY/CATCH ===
        if rule == 'st_try':
            operations.append(SemanticNode(
                depth=depth,
                type='try',
                text='try',
                parent_type=parent_type
            ))
            for child in children:
                child_text = child.getText() if hasattr(child, 'getText') else ""
                if child_text == 'catch':
                    operations.append(SemanticNode(
                        depth=depth,
                        type='catch',
                        text='catch',
                        parent_type='try'
                    ))
                elif get_rule_name(child) == 'statement':
                    walk(child, depth + 1, 'try')
            return

        # === RETURN STATEMENT ===
        if rule == 'st_return':
            operations.append(SemanticNode(
                depth=depth,
                type='return',
                text='return',
                parent_type=parent_type
            ))
            return

        # === BREAK STATEMENT ===
        if rule == 'st_break':
            operations.append(SemanticNode(
                depth=depth,
                type='break',
                text='break',
                parent_type=parent_type
            ))
            return

        # === CONTINUE STATEMENT ===
        if rule == 'st_continue':
            operations.append(SemanticNode(
                depth=depth,
                type='continue',
                text='continue',
                parent_type=parent_type
            ))
            return

        # === ASSIGNMENT ===
        if rule == 'st_assign':
            text = clean_text(get_text(node))
            operations.append(SemanticNode(
                depth=depth,
                type='assignment',
                text=text,
                parent_type=parent_type
            ))
            return

        # === FUNCTION CALL (as expression) ===
        if rule == 'xpr_function':
            text = clean_text(get_text(node))
            operations.append(SemanticNode(
                depth=depth,
                type='call',
                text=text,
                parent_type=parent_type
            ))
            return

        # === STATEMENT (wrapper) ===
        if rule == 'statement':
            # Check for return/break/continue by text (grammar may vary)
            stmt_text = get_text(node).strip().rstrip(';').strip()
            if stmt_text == 'return':
                operations.append(SemanticNode(
                    depth=depth, type='return', text='return', parent_type=parent_type
                ))
                return
            elif stmt_text == 'break':
                operations.append(SemanticNode(
                    depth=depth, type='break', text='break', parent_type=parent_type
                ))
                return
            elif stmt_text == 'continue':
                operations.append(SemanticNode(
                    depth=depth, type='continue', text='continue', parent_type=parent_type
                ))
                return
            # Otherwise recurse into children
            for child in children:
                walk(child, depth, parent_type)
            return

        # === TOP LEVEL ===
        if rule in ('matlab_file', None):
            for child in children:
                walk(child, depth, parent_type)
            return

        # === DEFAULT: recurse ===
        for child in children:
            walk(child, depth, parent_type)

    walk(tree)
    return operations


def code_to_nodes(code: str, as_objects: bool = False) -> list:
    """
    Convert MATLAB code to semantic operations with tree depth.

    Args:
        code: MATLAB source code
        as_objects: If True, return SemanticNode objects

    Returns:
        List of operations
    """
    operations = []

    if ANTLR_AVAILABLE:
        try:
            input_stream = InputStream(code)
            lexer = matlabLexer(input_stream)
            stream = CommonTokenStream(lexer)
            parser_inst = matlabParser(stream)

            # Suppress errors
            from antlr4.error.ErrorListener import ErrorListener
            class Silent(ErrorListener):
                def syntaxError(self, *args, **kwargs):
                    pass
            lexer.removeErrorListeners()
            parser_inst.removeErrorListeners()
            lexer.addErrorListener(Silent())
            parser_inst.addErrorListener(Silent())

            tree = parser_inst.matlab_file()
            operations = extract_tree_operations(tree, parser_inst)

        except Exception as e:
            pass

    if not operations:
        operations = fallback_extract(code)

    if as_objects:
        return operations
    else:
        return [f"[D{op.depth}] {op.type}: {op.text}" for op in operations]


def code_to_nodes_with_depth(code: str) -> tuple[list[str], list[int]]:
    """
    Convert code to (texts, depths) tuple for the model.

    Returns:
        (texts, depths) where both are aligned lists
    """
    operations = code_to_nodes(code, as_objects=True)
    texts = [op.text for op in operations]
    depths = [op.depth for op in operations]
    return texts, depths


def fallback_extract(code: str) -> list[SemanticNode]:
    """Fallback when ANTLR fails."""
    code = re.sub(r'%.*$', '', code, flags=re.MULTILINE)
    operations = []
    depth = 0

    for line in code.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('function'):
            operations.append(SemanticNode(depth=0, type='function', text=line.rstrip(';')))
            depth = 1
        elif line.startswith('if '):
            operations.append(SemanticNode(depth=depth, type='if', text=line.rstrip(',')))
            depth += 1
        elif line.startswith('elseif '):
            operations.append(SemanticNode(depth=max(0, depth-1), type='elseif', text=line.rstrip(',')))
        elif line == 'else':
            operations.append(SemanticNode(depth=max(0, depth-1), type='else', text='else'))
        elif line.startswith('for '):
            operations.append(SemanticNode(depth=depth, type='for', text=line.rstrip(',')))
            depth += 1
        elif line.startswith('while '):
            operations.append(SemanticNode(depth=depth, type='while', text=line.rstrip(',')))
            depth += 1
        elif line.startswith('switch '):
            operations.append(SemanticNode(depth=depth, type='switch', text=line))
            depth += 1
        elif line.startswith('case '):
            operations.append(SemanticNode(depth=depth, type='case', text=line))
        elif line == 'otherwise':
            operations.append(SemanticNode(depth=depth, type='otherwise', text='otherwise'))
        elif line == 'return' or line == 'return;':
            operations.append(SemanticNode(depth=depth, type='return', text='return'))
        elif line == 'break' or line == 'break;':
            operations.append(SemanticNode(depth=depth, type='break', text='break'))
        elif line == 'continue' or line == 'continue;':
            operations.append(SemanticNode(depth=depth, type='continue', text='continue'))
        elif line == 'end':
            depth = max(0, depth - 1)
        elif '=' in line and not line.startswith('='):
            operations.append(SemanticNode(depth=depth, type='assignment', text=line.rstrip(';')))
        elif line.endswith(')') or line.endswith(');'):
            operations.append(SemanticNode(depth=depth, type='call', text=line.rstrip(';')))

    return operations


def code_to_ast_nodes(code: str) -> list[str]:
    return code_to_nodes(code, as_objects=False)


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

    print("=" * 70)
    print("TREE-STRUCTURED SEMANTIC OPERATIONS")
    print("=" * 70)

    ops = code_to_nodes(test_code, as_objects=True)
    print(f"\nFound {len(ops)} operations:\n")
    for op in ops:
        indent = "  " * op.depth
        print(f"{indent}[depth={op.depth}] {op.type}: {op.text}")

    print("\n" + "=" * 70)
    print("SWITCH/CASE TEST")
    print("=" * 70)

    switch_code = """
    function result = test_switch(x)
        switch x
            case 1
                result = 'one';
            case 2
                result = 'two';
            otherwise
                result = 'other';
        end
    end
    """

    ops = code_to_nodes(switch_code, as_objects=True)
    print(f"\nFound {len(ops)} operations:\n")
    for op in ops:
        indent = "  " * op.depth
        print(f"{indent}[depth={op.depth}] {op.type}: {op.text}")

    print("\n" + "=" * 70)
    print("RETURN/BREAK/CONTINUE TEST")
    print("=" * 70)

    control_code = """
    function result = find_value(arr, target)
        for i = 1:length(arr)
            if arr(i) == target
                result = i;
                return;
            end
            if arr(i) < 0
                continue;
            end
            if arr(i) > 1000
                break;
            end
        end
        result = -1;
    end
    """

    ops = code_to_nodes(control_code, as_objects=True)
    print(f"\nFound {len(ops)} operations:\n")
    for op in ops:
        indent = "  " * op.depth
        print(f"{indent}[depth={op.depth}] {op.type}: {op.text}")

    print("\n" + "=" * 70)
    print("REAL FILE TEST")
    print("=" * 70)

    real_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "matlab_dataset/aludnam_MATLAB__analyzingtool_computebic/code.m"
    )
    try:
        with open(real_file) as f:
            real_code = f.read()
        ops = code_to_nodes(real_code, as_objects=True)
        print(f"\nFound {len(ops)} operations:\n")
        for op in ops[:25]:
            indent = "  " * op.depth
            text = op.text[:50] + "..." if len(op.text) > 50 else op.text
            print(f"{indent}[D{op.depth}] {op.type}: {text}")
        if len(ops) > 25:
            print(f"\n  ... and {len(ops) - 25} more operations")
    except FileNotFoundError:
        print("  (file not found)")
