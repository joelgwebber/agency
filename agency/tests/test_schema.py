from enum import Enum
from typing import List

from agency.schema import Schema, Type, parse_val, prop
from agency.schema import schema as schema_decorator
from agency.schema import schema_for
from agency.utils import timestamp


def test_basic_schema_types():
    @schema_decorator("A test class with basic types")
    class BasicTypes:
        string: str = prop("A string field")
        integer: int = prop("An integer field")
        real: float = prop("A float field")
        boolean: bool = prop("A boolean field")
        date: timestamp = prop("A timestamp field")

    schema = schema_for(BasicTypes)
    assert schema.typ == Type.Object
    assert schema.desc == "A test class with basic types"

    props = schema.prop_schemae
    assert props is not None

    assert props["string"].typ == Type.String
    assert props["integer"].typ == Type.Integer
    assert props["real"].typ == Type.Real
    assert props["boolean"].typ == Type.Boolean
    assert props["date"].typ == Type.DateTime


def test_enum_schema():
    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    @schema_decorator("A class with an enum")
    class WithEnum:
        color: Color = prop("Color choice")

    schema = schema_for(WithEnum)
    props = schema.prop_schemae
    assert props is not None

    color_schema = props["color"]
    assert color_schema.typ == Type.String
    assert color_schema.enum == ["red", "green", "blue"]


def test_nested_schema():
    @schema_decorator("Inner class")
    class Inner:
        value: str = prop("Inner value")

    @schema_decorator("Outer class")
    class Outer:
        inner: Inner = prop("Nested inner")
        inners: List[Inner] = prop("List of inners")

    schema = schema_for(Outer)
    props = schema.prop_schemae
    assert props is not None

    assert props["inner"].typ == Type.Object
    assert props["inners"].typ == Type.Array
    assert props["inners"].item_schema is not None
    assert props["inners"].item_schema.typ == Type.Object


def test_default_values():
    @schema_decorator()
    class WithDefaults:
        required: str = prop("Required field")
        optional: str = prop("Optional field", default="default")
        factory: List[str] = prop("Factory field", default_factory=list)

    schema = schema_for(WithDefaults)
    props = schema.prop_schemae
    assert props is not None

    assert props["required"].default is None
    assert props["optional"].default == "default"
    assert props["factory"].default == []


def test_parse_basic_types():
    assert parse_val("test", Schema(typ=Type.String, desc="")) == "test"
    assert parse_val(42, Schema(typ=Type.Integer, desc="")) == 42
    assert parse_val(3.14, Schema(typ=Type.Real, desc="")) == 3.14
    assert parse_val(True, Schema(typ=Type.Boolean, desc="")) == True

    # Test timestamp parsing
    ts = parse_val("2024-01-01T00:00:00Z", Schema(typ=Type.DateTime, desc=""))
    assert isinstance(ts, timestamp)
    assert ts.year == 2024
    assert ts.month == 1
    assert ts.day == 1


def test_parse_complex_types():
    @schema_decorator()
    class Person:
        name: str = prop("Person's name")
        age: int = prop("Person's age")

    data = {"name": "Alice", "age": 30}
    schema = schema_for(Person)

    result = parse_val(data, schema)
    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_list():
    schema = Schema(
        typ=Type.Array, desc="", item_schema=Schema(typ=Type.String, desc="")
    )

    result = parse_val(["a", "b", "c"], schema)
    assert result == ["a", "b", "c"]


def test_parse_dict():
    schema = Schema(
        typ=Type.Object, desc="", item_schema=Schema(typ=Type.String, desc="")
    )

    data = {"key1": "value1", "key2": "value2"}
    result = parse_val(data, schema)
    assert result == data


def test_openapi_schema():
    @schema_decorator("A test class")
    class TestClass:
        string: str = prop("A string")
        number: int = prop("A number")
        items: List[str] = prop("A list")

    schema = schema_for(TestClass)
    openapi = schema.to_openapi()

    assert openapi["type"] == "object"
    assert "properties" in openapi
    assert "required" in openapi

    props = openapi["properties"]
    assert props["string"]["type"] == "string"
    assert props["number"]["type"] == "number"
    assert props["items"]["type"] == "array"
    assert "items" in props["items"]
    assert props["items"]["items"]["type"] == "string"
