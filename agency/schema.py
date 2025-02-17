from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from enum import Enum
from inspect import isclass
from typing import Any, Callable, Dict, Iterable, List, Optional, cast, get_type_hints

from agency.models import OpenAPISchema
from agency.utils import timestamp

SCHEMA_KEY = "_schema"


def schema(cls=None, cls_desc: str = ""):
    """A class decorator that can be used on a dataclass, giving it an OpenAPI schema (accessible via schema_for(cls),
    and allowing it to be parsed with tools.parse_val(). This effectively extends @dataclass.
    """

    def decorator(_cls) -> type:
        # Schema objects are also dataclasses.
        # This cast is correct, but I can't seem to make the typechecker happy.
        cls: type = dataclass(_cls)  # pyright: ignore
        assert is_dataclass(_cls)

        # Precache schema for this class.
        _ensure_schema(cls, cls_desc)
        return cls

    return decorator(cls)


# TODO: Properly typing the return type causes all kinds of problems at call sites. Figure out why.
def prop(
    desc: str,
    *,
    default: Any = None,
    default_factory: Optional[Callable] = None,
) -> Any:
    """A initializer equivalent to dataclasses.Field, that gives the field a description to be used in its schema.
    This should only be used on fields with the @schema() annotation."""

    if default_factory:
        return field(default_factory=default_factory, metadata={"desc": desc})
    elif default is not None:
        return field(default=default, metadata={"desc": desc})
    return field(metadata={"desc": desc})


def schema_for(cls: type) -> Schema:
    """Gets the OpenAPI schema for a class with the @schema annotation."""

    if not hasattr(cls, SCHEMA_KEY):
        raise Exception(f"{cls} requires the @schema annotation")
    return getattr(cls, SCHEMA_KEY)


def _ensure_schema(cls: type, desc: str, default: Optional[Any] = None) -> Schema:
    # Use cached schema if there is one.
    if hasattr(cls, SCHEMA_KEY):
        # TODO: Consider allowing the field desc to override or contribute to the class desc,
        # which would require allocating a new Schema object when used in that scenario.
        return getattr(cls, SCHEMA_KEY)

    # Builtin types. These can't be cached, because of descriptions and defaults.
    elif cls == str:
        return Schema(typ=Type.String, desc=desc, default=default)
    elif cls == int:
        return Schema(typ=Type.Integer, desc=desc, default=default)
    elif cls == float:
        return Schema(typ=Type.Real, desc=desc, default=default)
    elif cls == bool:
        return Schema(typ=Type.Boolean, desc=desc, default=default)
    elif _is_type(cls, timestamp):
        return Schema(typ=Type.DateTime, desc=desc, default=default)

    # Enums.
    elif _is_enum(cls):
        enums: List[str] = []
        for enum in cast(Iterable, cls):
            # Ensure all are strings.
            # TODO: Are other types supported in OpenAPI?
            if type(enum.value) is not str:
                raise TypeError(f"expected str for {enum}; got {enum.value}")
            enums.append(enum.value)
        return Schema(typ=Type.String, desc=desc, default=default, enum=enums)

    # List types.
    elif _is_list(cls):
        # TODO:: better errors. Should items get a separate description?
        item_type = cls.__args__[0]  # pyright: ignore
        return Schema(
            typ=Type.Array,
            desc=desc,
            default=default,
            item_schema=_ensure_schema(item_type, ""),
        )

    # Dict types.
    elif _is_dict(cls):
        item_type = cls.__args__[1]
        return Schema(
            typ=Type.Object, desc=desc, default=default, item_schema=item_type
        )

    # Object types:
    elif is_dataclass(cls):
        # Compute schema properties from fields.
        props: Dict[str, Schema] = {}
        hints = get_type_hints(cls)
        for prop_fld in fields(cls):
            # Compute default value
            prop_default = None
            if prop_fld.default_factory != MISSING:
                prop_default = cast(Callable, prop_fld.default_factory)()
            elif prop_fld.default != MISSING:
                prop_default = prop_fld.default

            prop_type = hints[prop_fld.name]
            props[prop_fld.name] = _ensure_schema(
                prop_type, _meta_desc(prop_fld), prop_default
            )

        # Use cls_desc, ignoring the field's description.
        schema = Schema(
            typ=Type.Object, desc=desc, default=default, prop_schemae=props, cls=cls
        )
        setattr(cls, SCHEMA_KEY, schema)
        return schema

    raise Exception(f"{cls} requires the @schema annotation")


def _meta_desc(fld: Field) -> str:
    if "desc" in fld.metadata:
        return fld.metadata["desc"]
    return ""


def _is_enum(cls: Any) -> bool:
    if not isclass(cls):
        return False
    return issubclass(cls, Enum)


def _is_type(a, b) -> bool:
    # Seriously?! Fucking python has no sane way to reliably compare types.
    return f"{a.__module__}.{a.__name__}" == f"{b.__module__}.{b.__name__}"


def _is_list(typ) -> bool:
    return isinstance(typ, type(List)) or (
        hasattr(typ, "__origin__") and typ.__origin__ is list
    )


def _is_dict(typ) -> bool:
    return isinstance(typ, type(Dict)) or (
        hasattr(typ, "__origin__") and typ.__origin__ is dict
    )


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


def serialize_val(val: Any, schema: Optional[Schema]) -> Any:
    if schema is None:
        raise Exception(f"Need a schema to serialize {val}")

    match schema.typ:
        # Simple types.
        case Type.String:
            return str(val)
        case Type.Real:
            return float(val)
        case Type.Integer:
            return int(val)
        case Type.Boolean:
            return bool(val)
        case Type.DateTime:
            return timestamp.isoformat(val)

        # Arrays.
        case Type.Array:
            print(">>>", val)
            print(">>>", schema.item_schema)
            return [serialize_val(item, schema.item_schema) for item in val]

        # Objects.
        case Type.Object:
            if schema.item_schema is not None:
                # TODO: Validate item type.
                return dict(val.items())
            if schema.prop_schemae is None or schema.cls is None:
                raise Exception(
                    f"Need property schemae to serialize object {val} : {schema}"
                )

            # Serialize dataclass fields.
            serialized_obj = {}
            for name, prop_schema in schema.prop_schemae.items():
                serialized_obj[name] = serialize_val(getattr(val, name), prop_schema)
            return serialized_obj


def parse_val(val: Any, schema: Schema) -> Any:
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
            if not schema.item_schema:
                raise Exception(f"missing item_schema for {schema}")
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
