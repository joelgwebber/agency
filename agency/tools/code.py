import tree_sitter_python as tspython
from tree_sitter import Language, Parser

lang_python = Language(tspython.language())
parser = Parser(lang_python)

tree = parser.parse(
    bytes(
        """
def foo():
    if bar:
        baz()
""",
        "utf8",
    )
)
