from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from google.cloud.aiplatform_v1beta1 import FunctionCall
from vertexai.generative_models import FunctionDeclaration, Part
from vertexai.generative_models import Tool as LangTool

from agency.utils import timestamp


class Type(Enum):
    String = 1
    Real = 2
    Integer = 3
    Boolean = 4
    Array = 5
    Object = 6
    DateTime = 7


_json_type = {
    Type.String: "string",
    Type.Integer: "integer",
    Type.Real: "number",
    Type.Boolean: "boolean",
    Type.Array: "array",
    Type.Object: "object",
    Type.DateTime: "string",
}


@dataclass
class Prop:
    typ: Type
    desc: str
    default: Any = None
    enum: Optional[List[Any]] = None
    properties: Optional[Dict[str, Prop]] = None
    items: Optional[Prop] = None

    def as_dict(self) -> Dict[str, Any]:
        """Produces a dictionary in the structure expected by the OpenAPI schema."""
        d: Dict[str, Any] = {
            "type": _json_type[self.typ],
            "description": self.desc,
        }

        if self.typ == Type.Object:
            if self.properties is None:
                raise Exception("Object type requires 'properties'")

            d["properties"] = {}
            d["required"] = []
            for name, p in self.properties.items():
                d["properties"][name] = p.as_dict()
                if p.default is None:
                    d["required"].append(name)

        if self.typ == Type.Array:
            if self.items is None:
                raise Exception("Array type requires 'items'")

            d["items"] = self.items.as_dict()

        return d


class Decl:
    fn: Callable
    name: str
    desc: str
    root: Prop

    def __init__(self, fn: Callable, name: str, desc: str, props: Dict[str, Prop]):
        self.fn = fn
        self.name = name
        self.desc = desc
        self.root = Prop(Type.Object, "", properties=props)


class Tool:
    _funcs: List[FunctionDeclaration]
    _decls: Dict[str, Decl]

    def __init__(self):
        self._funcs = []
        self._decls = {}

    @property
    def funcs(self) -> List[FunctionDeclaration]:
        return self._funcs

    def _add_decl(self, decl: Decl) -> None:
        if decl.name in self._decls:
            raise Exception(f"duplicate declaration {decl.name}")

        self._decls[decl.name] = decl
        self._funcs.append(
            FunctionDeclaration(
                name=decl.name,
                description=decl.desc,
                parameters=decl.root.as_dict(),
            )
        )

    def dispatch(self, fn: FunctionCall) -> Part:
        args = {}
        decl = self._decls[fn.name]

        if decl.root.properties is None:
            raise Exception("Bad declaration. This shouldn't happen.")

        # Parse args.
        for name, p in decl.root.properties.items():
            # Fill in default values.
            if p.default is not None:
                args[name] = p.default

            if name in fn.args:
                val = fn.args[name]
                args[name] = _parse_val(val, p)

        result = decl.fn(**args)

        # TODO: Why the hell does this blow up now?
        # return Part.from_function_response(
        #     fn.name, response={"content": {"result": result}}
        # )

        # This is a hack, but works.
        return Part.from_dict(
            {
                "function_response": {
                    "name": fn.name,
                    "response": {
                        "content": {
                            "result": result,
                        }
                    },
                }
            }
        )


class ToolBox:
    _funcs: List[FunctionDeclaration]
    _tools: Dict[str, Tool]

    def __init__(
        self,
        tools: List[Tool],
    ):
        self._funcs = []
        self._tools = {}

        # Register tools.
        for tool in tools:
            self._register(tool)

    def _register(self, tools: Tool) -> None:
        self._funcs.extend(tools.funcs)
        for name in tools._decls:
            if name in self._tools:
                raise Exception(f"duplicate tool {name}")
            self._tools[name] = tools

    @property
    def lang_tools(self) -> LangTool:
        return LangTool(self._funcs)

    def dispatch(self, fn: FunctionCall) -> Part:
        try:
            tool = self._tools[fn.name]
        except KeyError:
            return Part.from_function_response(
                fn.name, {"error": f"unknown function {fn.name}"}
            )

        try:
            return tool.dispatch(fn)
        except Exception as e:
            # Catch exceptions, log them, and send them to the model in hopes it will sort itself.
            msg = f"exception calling {fn.name}: {e}"
            print(msg, traceback.format_stack())
            return Part.from_function_response(fn.name, {"error": msg})


def _parse_val(val: Any, p: Optional[Prop]) -> Any:
    if p is None:
        raise Exception(f"Need a value type to parse {val}")

    match p.typ:
        case Type.String:
            return val
        case Type.Real:
            return float(val)
        case Type.Integer:
            # Sometimes we get a float for an int.
            try:
                return int(val)
            except:
                return int(float(val))
        case Type.Boolean:
            return bool(val)
        case Type.DateTime:
            return timestamp.fromisoformat(val)

        case Type.Array:
            return [_parse_val(item, p.items) for item in val]

        case Type.Object:
            if p.properties is None:
                raise Exception(f"Need property types to parse object {val} : {p}")
            return {k: _parse_val(v, p.properties[k]) for (k, v) in val.items()}
