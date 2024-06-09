from __future__ import annotations

import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from google.cloud.aiplatform_v1beta1 import FunctionCall
from vertexai.generative_models import FunctionDeclaration, Part
from vertexai.generative_models import Tool as LangTool

from agency.utils import timestamp

FUNC_KEY = "_func"
SCHEMA_KEY = "_schema"


class Type(Enum):
    """Types specified by json-schema. More optional detail can be specified in 'format'."""

    String = "string"
    Real = "integer"
    Integer = "number"
    Boolean = "boolean"
    Array = "array"
    Object = "object"
    DateTime = "string"


class Format(Enum):
    """Not all of these are supported by all implementations, but specifying them shouldn't
    interfere with anything."""

    Default = ""

    DateTime = "date-time"  # standard
    Date = "date"
    Time = "time"
    Duration = "duration"
    Email = "email"
    Uuid = "uuid"
    Uri = "uri"

    Integer32 = "int32"  # Gemini
    Integer64 = "int64"
    Float32 = "float"
    Float64 = "double"


@dataclass
class Schema:
    """Schema type used for tool arguments. Roughly analgous to the json-schema / open-api
    flavors used by language-model function calling interfaces."""

    typ: Type
    desc: Optional[str] = None
    format: Format = Format.Default
    default: Any = None
    enum: Optional[List[Any]] = None
    properties: Optional[Dict[str, Schema]] = None
    items: Optional[Schema] = None

    def as_dict(self) -> Dict[str, Any]:
        """Produces a dictionary in the structure expected by the Gemini / OpenAPI schema."""

        d: Dict[str, Any] = {
            "type": self.typ.value,
            "description": self.desc,
        }

        if self.enum is not None:
            d["enum"] = self.enum

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


def func_for(func: Callable) -> Func:
    """TODO: doc"""

    if not hasattr(func, FUNC_KEY):
        raise Exception(f"{func} requires the @lmfunc annotation")
    return getattr(func, FUNC_KEY)


@dataclass
class Func:
    fn: Callable
    name: str
    desc: str
    args: Schema

    def __init__(self, fn: Callable, name: str, desc: str, props: Any):
        self.fn = fn
        self.name = name
        self.desc = desc
        if isinstance(props, Schema):
            self.args = props
        else:
            self.args = Schema(Type.Object, "", Format.Default, properties=props)


class Tool:
    _funcs: List[FunctionDeclaration]
    _decls: Dict[str, Func]

    def __init__(self):
        self._funcs = []
        self._decls = {}

    @property
    def funcs(self) -> List[FunctionDeclaration]:
        return self._funcs

    def _add_func2(self, func: Callable) -> None:
        self._add_func(func_for(func))

    def _add_func(self, decl: Func) -> None:
        if decl.name in self._decls:
            raise Exception(f"duplicate declaration {decl.name}")

        self._decls[decl.name] = decl
        self._funcs.append(
            FunctionDeclaration(
                name=decl.name,
                description=decl.desc,
                parameters=decl.args.as_dict(),
            )
        )

    def dispatch(self, fn: FunctionCall) -> Part:
        args = {}
        decl = self._decls[fn.name]

        if decl.args.properties is None:
            raise Exception("Bad declaration. This shouldn't happen.")

        # Parse args.
        for name, p in decl.args.properties.items():
            # Fill in default values.
            if p.default is not None:
                args[name] = p.default

            if name in fn.args:
                val = fn.args[name]
                args[name] = _parse_val(val, p)

        args = {"args": args}
        print(">>>>", args)
        result = decl.fn(self, **args)

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
            print(">>>", fn)
            return tool.dispatch(fn)
        except Exception as e:
            # Catch exceptions, log them, and send them to the model in hopes it will sort itself.
            msg = f"exception calling {fn.name}: {e}"
            print(msg, "\n".join(traceback.format_exception(e)))
            return Part.from_function_response(fn.name, {"error": msg})


def _parse_val(val: Any, p: Optional[Schema]) -> Any:
    # print(">>>", val, p)
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
