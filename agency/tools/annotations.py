from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from enum import Enum
from inspect import isclass
from typing import Any, Callable, Dict, Iterable, List, Optional, cast, get_type_hints

from agency.tools.tools import SCHEMA_KEY, Schema, Type
from agency.utils import timestamp


def schema(cls_desc: str = ""):
    """TODO: doc"""

    def decorator(_cls) -> type:
        # Schema objects are also dataclasses.
        # This cast is correct, but I can't seem to make the typechecker happy.
        cls: type = dataclass(_cls)  # pyright: ignore
        assert is_dataclass(_cls)

        # Precache schema for this class.
        _ensure_schema(cls, cls_desc)
        return cls

    return decorator


# TODO: Figure out how to fix field()'s type checker magic here.
# Returning the correct Field[T] breaks callers.
def prop(
    desc: str,
    *,
    default: Any = None,
    default_factory: Optional[Callable] = None,
) -> Any:
    """TODO: doc"""

    if default_factory:
        return field(default_factory=default_factory, metadata={"desc": desc})
    elif default:
        return field(default=default, metadata={"desc": desc})
    return field(metadata={"desc": desc})


def schema_for(cls: type) -> Schema:
    """TODO: doc"""

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
