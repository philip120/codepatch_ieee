from dataclasses import dataclass
from enum import Enum

class BlockType(Enum):
    FUNCTION = 0
    IF = 1
    WHILE = 2
    FOR = 3
    SWITCH = 4
    CASE = 5
    TRY = 6

@dataclass
class CodeBlock():
    type: BlockType
    opener: str
    contents: list

class MatlabParser():
    def __init__(self):
        self.block_starters = ["function", "if", "while", "for", "switch", "try"]
        self.longest_starter = max([len(starter) for starter in self.block_starters])

    def parse(self, code) -> CodeBlock:
        def agregate(lines):
            try:
                block = CodeBlock(BlockType(
                [lines[0].startswith(starter) for starter in self.block_starters].index(True)),
                lines[0],
                [])
            except ValueError as e:
                print(lines[0])
                raise e
            index = 1
            while index < len(lines):
                if lines[index].startswith(tuple(self.block_starters)):
                    inner_block = agregate(lines[index:])
                    index += len(inner_block.contents) + 2 # +2 for opener and "end"
                    block.contents.append(inner_block)
                    continue 
                if lines[index].startswith("end"):
                    return block
                block.contents.append(lines[index])
                index += 1
                
        lines = self._split(code)
        tree = agregate(lines)
        return tree

    def _split(self, code) -> list[str]:
        lines = []
        lines_raw = code.split("\n")
        next_line = ""
        for line in lines_raw:
            if line.endswith("..."):
                next_line += line[:-3]
                continue
            next_line += line
            lines.append(next_line.strip())
            next_line = ""
        while "" in lines:
            lines.remove("")

        return lines
