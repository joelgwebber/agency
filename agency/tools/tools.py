from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from vertexai.generative_models import FunctionCall, FunctionDeclaration, Part
from vertexai.generative_models import Tool as LangTool

from agency.utils import print_proto, timestamp

DECL_KEY = "_decl"
SCHEMA_KEY = "_schema"


# TODO:Add more types and translate to OpenAPI's type + format.
# - date
# - time
# - duration
# - email
# - uuid
# - uri
# - int32/64, float32/64 (Gemini-specific)
class Type(Enum):
    """Types specified by OpenAPI formats.
    These are used for "format"; _openapi_type defined type "type" field."""

    String = "string"
    Real = "integer"
    Integer = "number"
    Boolean = "boolean"
    Array = "array"
    Object = "object"
    DateTime = "date-time"


# OpenAPI "format" values for various types.
# Types absent from this map require no format specifier.
_openapi_type: Dict[Type, str] = {
    Type.String: "string",
    Type.Real: "integer",
    Type.Integer: "number",
    Type.Boolean: "boolean",
    Type.Array: "array",
    Type.Object: "object",
    Type.DateTime: "string",
}


@dataclass
class Schema:
    """Schema type used for tool arguments. Roughly analgous to the json-schema / open-api
    flavors used by language-model function calling interfaces."""

    typ: Type
    desc: Optional[str] = None
    default: Any = None

    # Enumeration options.
    enum: Optional[List[Any]] = None

    # Schema for array items.
    item_schema: Optional[Schema] = None

    # Schemae for object properties.
    prop_schemae: Optional[Dict[str, Schema]] = None

    # Object dataclass to instantiate.
    cls: Optional[type] = None

    # TODO: This is very Gemini-specific at the moment.
    # Should be easy to generalize to other function-calling model APIs.
    def to_openapi(self) -> Dict[str, Any]:
        """Produces a dictionary in the structure expected by the Gemini / OpenAPI schema."""

        # TODO: Better validation of valid states (e.g., enums with primitives, etc).

        d: Dict[str, Any] = {
            "type": _openapi_type[self.typ],
            "format": self.typ.value,
            "description": self.desc,
        }

        if self.enum is not None:
            d["enum"] = self.enum

        if self.typ == Type.Object:
            if self.item_schema is not None:
                # TODO: Confirm this will work properly for an open-ended dict in OpenAPI.
                return d
            elif self.prop_schemae is None:
                raise Exception("Object type requires prop_schemae")

            d["properties"] = {}
            d["required"] = []
            for name, p in self.prop_schemae.items():
                d["properties"][name] = p.to_openapi()
                if p.default is None:
                    d["required"].append(name)

        if self.typ == Type.Array:
            if self.item_schema is None:
                raise Exception("Array type requires 'items'")
            d["items"] = self.item_schema.to_openapi()

        return d


@dataclass
class Decl:
    """TODO: doc"""

    fn: Callable
    name: str
    desc: str
    args: Schema

    def __init__(self, fn: Callable, name: str, desc: str, args: Any):
        self.fn = fn
        self.name = name
        self.desc = desc
        if isinstance(args, Schema):
            self.args = args
        else:
            self.args = Schema(Type.Object, "", prop_schemae=args)


class Tool:
    _decls: Dict[str, Decl]
    _funcs: List[FunctionDeclaration]

    def __init__(self):
        self._funcs = []
        self._decls = {}

    @property
    def funcs(self) -> List[FunctionDeclaration]:
        """TODO: doc"""
        return self._funcs

    def declare(self, func: Callable) -> None:
        """TODO: doc"""

        decl = _decl_for(func)
        if decl.name in self._decls:
            raise Exception(f"duplicate declaration {decl.name}")

        self._decls[decl.name] = decl
        self._funcs.append(
            FunctionDeclaration(
                name=decl.name,
                description=decl.desc,
                parameters=decl.args.to_openapi(),
            )
        )

    def dispatch(self, fn: FunctionCall) -> Part:
        decl = self._decls[fn.name]
        if decl.args.prop_schemae is None:
            raise Exception("Bad declaration. This shouldn't happen.")

        # Parse args and call the target function.
        args = _parse_val(fn.args, decl.args)
        result = decl.fn(self, args)

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
    """TODO: doc"""

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
            msg = f"""exception calling {fn.name}: {e}
                     {"\n".join(traceback.format_exception(e))}"""

            return Part.from_function_response(fn.name, {"error": msg})


def _decl_for(func: Callable) -> Decl:
    if not hasattr(func, DECL_KEY):
        raise Exception(f"{func} requires the @lmfunc annotation")
    return getattr(func, DECL_KEY)


# `schema` is only optional to avoid making every call-site messy.
def _parse_val(val: Any, schema: Optional[Schema]) -> Any:
    if schema is None:
        raise Exception(f"Need a value type to parse {val}")

    match schema.typ:
        # Simple types.
        case Type.String:
            return val
        case Type.Real:
            return float(val)
        case Type.Integer:
            return _loose_int(val)
        case Type.Boolean:
            return bool(val)
        case Type.DateTime:
            return timestamp.fromisoformat(val)

        # Arrays.
        case Type.Array:
            return [_parse_val(item, schema.item_schema) for item in val]

        # Objects.
        case Type.Object:
            if schema.item_schema is not None:
                # TODO: Validate item type.
                return dict(val.items())
            if schema.prop_schemae is None or schema.cls is None:
                raise Exception(
                    f"Need property schemae to parse object {val} : {schema}"
                )

            # Parse ctor args and instantiate the dataclass.
            ctor_args = {
                k: _parse_val(v, schema.prop_schemae[k]) for (k, v) in val.items()
            }

            sig = inspect.signature(schema.cls)
            # TODO: ...
            return schema.cls(**ctor_args)


def _loose_int(val: Any) -> int:
    # Sometimes we get a float for an int.
    try:
        return int(val)
    except:
        return int(float(val))
