from semantic_nodes import extract_flat_semantic_nodes
from ast_utils import suppress_stderr

if __name__ == "__main__":
    from matlab_lexer import MatlabLexer
    from matlab_parser import MatlabParser

    code = """
    function zvalues = interactivedendrogramvalues(fig)
% zvalues = interactivedendrogramvalues(fig)
% from http://www.mathworks.com/help/techdoc/ref/datacursormode.html#bsawkea-7
dcm_obj = datacursormode(fig);
stop = 0;
inp = input('How many nodes?\\n');
for ii=1:inp
    
    set(dcm_obj,'DisplayStyle','datatip',...
        'SnapToDataVertex','off','Enable','on')
    
    disp('Click on a node, then press Return.')
    figure(fig)
    pause                            % Wait while the user does this.    
    c_info = getCursorInfo(dcm_obj);
    zvalues(ii)=c_info.Position(2);        
    fprintf('Zvalue:%g\\n', zvalues(ii))
end
    """

    lexer = MatlabLexer()
    parser = MatlabParser()
    try:
        with suppress_stderr():
            ast = parser.parse(lexer.tokenize(code))
        from semantic_extractor import extract_semantic_nodes
        nodes = extract_semantic_nodes(code, parser, lexer)


        if not nodes:
            raise ValueError("Empty semantic extraction")

    except Exception as e:
        print(f"Error: {e}")
        from semantic_nodes import SemanticNode
        nodes = [SemanticNode(0, code.strip())]

    for n in nodes:
        print(f"[{n.id}] {n.text}")
