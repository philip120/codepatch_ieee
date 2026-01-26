from parser_new import MatlabParser, CodeBlock

TEST_CODE = """
function paths = djikstra(connections)
    sz = size(connections);
    if length(sz) != 2 | sz(1) != sz(2) 
        error('Error: connection matrix must be 2-dimensional square matrix')
    end
    paths = cell(sz(1), sz(1));
    for index = 1:sz(1)
        distance = Inf(sz(1),1);
        distance(index) = 0;
        prev = Nan(sz(1),1);
        visited = zeros(sz(1),1)
        while ~all(visited)
            temp_distance = distance;
            temp_distance(visited) = inf;
            c_d, c_i = min(temp_distance);
            visited(c_i) = 1;
            for next = 1:sz(1)
	            to_next = c_d + connections(c_i, next);
				if connections(c_i, next) > 0 && to_next < distance(next)
		            distance(next) = to_next;
	                prev(next) = c_i;
                end
            end
        end
        for vert = 1:sz(1)
            while 1
                p = prev(vert);
                if isnan(p)
                    break
                end
                paths(index,vert) = [paths(index,vert), p]; 
            end
        end
    end
end
"""

def main():
    parser = MatlabParser()

    tree = parser.parse(TEST_CODE)
    def show_block(block: CodeBlock, level: int) -> None:
        print("\t" * level,block.opener)
        for line in block.contents:
            if isinstance(line, CodeBlock):
                show_block(line, level + 1)
            else:
                print("\t" * (level + 1),line)
        print("\t" * level,"end")
    show_block(tree, 0)

if __name__ == "__main__":
    main()