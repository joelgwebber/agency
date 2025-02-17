from typing import Dict

from agency.tool import Tool


class Toolbox:
    _toolbox: Dict[str, Tool]

    def __init__(self, *tools: Tool):
        self._toolbox = {tool.decl.id: tool for tool in tools}

    def tool_by_id(self, tool_id: str) -> Tool:
        if tool_id in self._toolbox:
            return self._toolbox[tool_id]
        raise Exception(f"no such tool: {tool_id}")
