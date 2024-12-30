from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from agency.router import FunctionDesc, OpenAPISchema
from agency.utils import timestamp

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
    desc: str
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
    def to_openapi(self) -> OpenAPISchema:
        """Produces a dictionary in the structure expected by the Gemini / OpenAPI schema."""

        # TODO: Better validation of valid states (e.g., enums with primitives, etc).

        d: OpenAPISchema = {
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
class ToolDecl:
    """Declaration for a tool that can be used by a language model (via a Minion).
    Their ids must be unique within the context of a single Minion."""

    id: str
    desc: str
    params: Schema

    def __init__(self, name: str, desc: str, params: Schema):
        self.id = name
        self.desc = desc
        self.params = params

    def to_func(self) -> FunctionDesc:
        return {
            "name": self.id,
            "description": self.desc,
            "parameters": self.params.to_openapi(),
        }


# `schema` is only optional to avoid making every call-site messy.
def parse_val(val: Any, schema: Optional[Schema]) -> Any:
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
            return [parse_val(item, schema.item_schema) for item in val]

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
                k: parse_val(v, schema.prop_schemae[k]) for (k, v) in val.items()
            }

            # TODO: Validate signature
            # sig = inspect.signature(schema.cls)
            return schema.cls(**ctor_args)


def _loose_int(val: Any) -> int:
    # Sometimes we get a float for an int.
    try:
        return int(val)
    except:
        return int(float(val))
