from agency.schema import prop, schema, schema_for
from agency.tool import Stack, Tool, ToolDecl


class LSPTest(Tool):
    @schema
    class Params:
        question: str = prop("")

    @schema
    class Returns:
        answer: str = prop("")

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "lsp-test",
            "",
            schema_for(LSPTest.Params),
            schema_for(LSPTest.Returns),
        )

    def invoke(self, stack: Stack):
        stack.respond({})


# from multilspy import SyncLanguageServer
# from multilspy.language_server import Language, MultilspyConfig, MultilspyLogger
#
# class LSP:
#     server: SyncLanguageServer
#
#     def __init__(self, src_dir: str):
#         config = MultilspyConfig.from_dict({"code_language": Language.PYTHON})
#         logger = MultilspyLogger()
#         self.server = SyncLanguageServer.create(config, logger, src_dir)
#
#     def symbol_at(self, file: str, line: int, col: int) -> str:
#         with self.server.start_server():
#             hover = self.server.request_hover(file, line, col)
#             if hover and "value" in hover["contents"]:
#                 return hover["contents"]["value"]
#         return ""


# import os
# import threading
# from subprocess import PIPE, Popen
#
# import pylspclient
# from pylspclient.lsp_client import TextDocumentIdentifier, TextDocumentItem
# from pylspclient.lsp_pydantic_strcuts import LanguageIdentifier
#
#
# pylsp_cmd = ["python", "-m", "lsp"]
#
#
# def consume_stderr(proc):
#     for line in iter(proc.stderr.readline, b""):
#         print(f"LSP stderr: {line.decode()}")
#
#
# proc = Popen(pylsp_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=0)
# stderr_thread = threading.Thread(target=consume_stderr, args=(proc,), daemon=True)
# stderr_thread.start()
#
#
# rpc = pylspclient.JsonRpcEndpoint(proc.stdin, proc.stdout)
# endpoint = pylspclient.LspEndpoint(rpc)
# lsp = pylspclient.LspClient(endpoint)
#
# caps = {
#     "textDocument": {
#         "completion": {
#             "completionItem": {
#                 "commitCharactersSupport": True,
#                 "documentationFormat": ["markdown", "plaintext"],
#                 "snippetSupport": True,
#             }
#         }
#     }
# }
# root_path = os.path.abspath("./research/src/")
# root_uri = "uri://" + root_path
#
# init_rsp = lsp.initialize(None, None, root_uri, None, caps, "off", None)
# lsp.initialized()
#
# file_path = "./test.py"
# file_uri = "uri://" + file_path
# file_text = open(file_path, "r").read()
# lsp.didOpen(
#     TextDocumentItem(
#         uri=file_uri,
#         languageId=LanguageIdentifier.PYTHON,
#         version=1,
#         text=file_text,
#     )
# )
# symbols = lsp.documentSymbol(TextDocumentIdentifier(uri=file_uri))
# print(symbols)
#
#
# def cleanup():
#     lsp.shutdown()
#     lsp.exit()
#     proc.stdin.close() if proc.stdin else None
#     proc.stdout.close() if proc.stdout else None
#     proc.stderr.close() if proc.stderr else None
#     proc.terminate()
#     proc.wait(timeout=5)
