from __future__ import annotations

from dataclasses import Field, dataclass, field, fields, is_dataclass
from enum import Enum
from inspect import isclass
from typing import Any, Dict, List, Optional, TypeVar, get_type_hints

from agency.tools.tools import DECL_KEY, SCHEMA_KEY, Decl, Schema, Type
from agency.utils import timestamp

ARGS_NAME = "args"


# TODO: Figure out how to fix field()'s type checker magic here.
# Returning the correct Field[T] breaks callers.
def prop(desc: str, *, default: Any = None) -> Any:
    """TODO: doc"""
    return field(default=default, metadata={"desc": desc})


def decl(func_name: str, func_desc: str):
    """TODO: doc"""

    def decorator(func):
        hints = get_type_hints(func)
        if len(hints) > 2 or ARGS_NAME not in hints:
            # TODO: Check this more carefully. It should be either (args) or (self, args), but sometimes
            # we get just (args) for methods.
            raise Exception(f"@{decl.__name__} may have only a single 'args' parameter")

        args = hints[ARGS_NAME]
        setattr(
            func,
            DECL_KEY,
            Decl(func, func_name, func_desc, _schema_for(args)),
        )
        return func

    return decorator


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


def _schema_for(cls: type) -> Schema:
    if not hasattr(cls, SCHEMA_KEY):
        raise Exception(f"{cls} requires the @lmschema annotation")
    return getattr(cls, SCHEMA_KEY)


def _ensure_schema(cls: type, desc: str, default: Optional[Any] = None) -> Schema:
    # Use cached schema if there is one.
    if hasattr(cls, SCHEMA_KEY):
        # TODO: Consider allowing the field desc to override or contribute to the class desc,
        # which would require allocating a new Schema object when used in that scenario.
        return getattr(cls, SCHEMA_KEY)

    # Builtin types. These can't be cached, because of descriptions and defaults.
    elif cls == str:
        return Schema(Type.String, desc, default)
    elif cls == int:
        return Schema(Type.Integer, desc, default)
    elif cls == float:
        return Schema(Type.Real, desc, default)
    elif cls == bool:
        return Schema(Type.Boolean, desc, default)
    elif _is_type(cls, timestamp):
        return Schema(Type.DateTime, desc, default)

    # Enums.
    elif _is_enum(cls):
        enums: List[str] = []
        for enum in cls:
            # Ensure all are strings.
            # TODO: Are other types supported in OpenAPI?
            if type(enum.value) is not str:
                raise TypeError(f"expected str for {enum}; got {enum.value}")
            enums.append(enum.value)
        return Schema(Type.String, desc, default, enum=enums)

    # List types.
    elif _is_list(cls):
        # TODO:: better errors. Should items get a separate description?
        item_type = cls.__args__[0]  # pyright: ignore
        return Schema(Type.Array, desc, items=_ensure_schema(item_type, ""))

    # Object types:
    elif is_dataclass(cls):
        # Compute schema properties from fields.
        props: Dict[str, Schema] = {}
        hints = get_type_hints(cls)
        for prop_fld in fields(cls):
            prop_type = hints[prop_fld.name]
            props[prop_fld.name] = _ensure_schema(
                prop_type, _meta_desc(prop_fld), prop_fld.default
            )

        # Use cls_desc, ignoring the field's description.
        schema = Schema(Type.Object, desc, properties=props, cls=cls)
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
    # Seriously?! Fucking python has bolted generics in the most confusing way imaginable.
    return getattr(typ, "__origin__", None) is list
