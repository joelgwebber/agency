from __future__ import annotations

from typing import Any, Dict, List, NotRequired, TypedDict


class OpenAPISchema(TypedDict):
    """OpenAPI schema type used to describe function parameters."""

    type: str
    description: NotRequired[str]
    enum: NotRequired[list[Any]]
    properties: NotRequired[Dict[str, OpenAPISchema]]
    required: NotRequired[List[str]]
    items: NotRequired[OpenAPISchema]
