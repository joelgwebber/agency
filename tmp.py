from typing import List
from agency.schema import prop, schema


@schema
class Returns:
    results: List[str] = prop("search results")
    error: str = prop("API error", default="")
