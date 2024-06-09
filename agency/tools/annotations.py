from __future__ import annotations

from dataclasses import Field, dataclass, fields, is_dataclass
from enum import Enum
from inspect import isclass
from typing import Any, Dict, List, Optional, get_type_hints

from agency.tools.tools import FUNC_KEY, SCHEMA_KEY, Format, Func, Schema, Type
from agency.utils import timestamp

ARGS_NAME = "args"


def schema_for(cls: type) -> Schema:
    """TODO: doc"""

    if not hasattr(cls, SCHEMA_KEY):
        raise Exception(f"{cls} requires the @lmschema annotation")
    return getattr(cls, SCHEMA_KEY)


def lmfunc(func_name: str, func_desc: str):
    """TODO: doc"""

    def decorator(func):
        hints = get_type_hints(func)
        if len(hints) != 2 or ARGS_NAME not in hints:
            raise Exception(
                f"@{lmfunc.__name__} may have only a single 'args' parameter"
            )

        args = hints[ARGS_NAME]
        setattr(
            func,
            FUNC_KEY,
            Func(func, func_name, func_desc, schema_for(args)),
        )
        return func

    return decorator


def lmschema(cls_desc: str):
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


def _ensure_schema(cls: type, desc: str, default: Optional[Any] = None) -> Schema:
    """TODO: doc"""

    # Use cached schema if there is one.
    if hasattr(cls, SCHEMA_KEY):
        # TODO: Consider allowing the field desc to override or contribute to the class desc,
        # which would require allocating a new Schema object when used in that scenario.
        return getattr(cls, SCHEMA_KEY)

    # Builtin types. These can't be cached, because of descriptions and defaults.
    elif cls == str:
        return Schema(Type.String, desc, Format.Default, default)
    elif cls == int:
        return Schema(Type.Integer, desc, Format.Default, default)
    elif cls == float:
        return Schema(Type.Real, desc, Format.Default, default)
    elif cls == bool:
        return Schema(Type.Boolean, desc, Format.Default, default)
    elif _is_type(cls, timestamp):
        return Schema(Type.DateTime, desc, Format.Default, default)

    # Enums.
    elif _is_enum(cls):
        enums: List[str] = []
        for enum in cls:
            # Ensure all are strings.
            # TODO: Are other types supported in OpenAPI?
            if type(enum.value) is not str:
                raise TypeError(f"expected str for {enum}; got {enum.value}")
            enums.append(enum.value)
        return Schema(Type.String, desc, Format.Default, default, enum=enums)

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
        schema = Schema(Type.Object, desc, properties=props)
        setattr(cls, SCHEMA_KEY, schema)
        return schema

    raise Exception(f"{cls} requires the @{lmschema.__name__} annotation")


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
