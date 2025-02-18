from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from agency.schema import parse_val, prop, schema, schema_for, serialize_val
from agency.tool import Stack, Tool, ToolDecl

# Just use Tavily for now.
_TAVILY_API_URL = "https://api.tavily.com"


@schema
class SearchResult:
    url: str = prop("result url")
    content: str = prop("result content")


class Search(Tool):

    @schema
    class Params:
        query: str = prop("url to fetch")
        max_results: int = prop("maximum number of results", default=5)

    @schema
    class Returns:
        results: List[SearchResult] = prop("search results")
        error: str = prop("API error", default="")

    _api_key: str

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "search-query",
            "Performs a web search",
            schema_for(Search.Params),
            schema_for(Search.Returns),
        )

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        raw_json = self._raw_results(args.query, args.max_results)
        cleaned = self._clean_results(raw_json)
        stack.respond(serialize_val(cleaned, self.decl.returns))

    def _raw_results(
        self,
        query: str,
        max_results: Optional[int] = 5,
        search_depth: Optional[str] = "advanced",
        include_domains: Optional[List[str]] = [],
        exclude_domains: Optional[List[str]] = [],
        include_answer: Optional[bool] = False,
        include_raw_content: Optional[bool] = False,
        include_images: Optional[bool] = False,
    ) -> Dict:
        params = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
        }
        rsp = requests.post(
            # type: ignore
            f"{_TAVILY_API_URL}/search",
            json=params,
        )
        # Return the response json directly, even if it's an error.
        # TODO: We may want to settle on a standard format for errors, so we can control behavior better.
        return rsp.json()

    # Results json from the Tavily API has the following structure:
    # { query: str
    #   follow_up_questions?: (unk)
    #   answer?: str
    #   images?: List[(unk)]
    #   results: List[{
    #     title: str
    #     url: str
    #     content: str
    #     score: number
    #     raw_content?: (unk)
    #   }]
    #   response_time: number
    # }
    def _clean_results(self, results: Dict[str, Any]) -> Search.Returns:
        """Clean results from Tavily Search API."""
        if "results" in results:
            clean_results: List[SearchResult] = []
            for result in results["results"]:
                clean_results.append(
                    SearchResult(url=result["url"], content=result["content"])
                )
            return Search.Returns(results=clean_results)

        # Error details.
        elif "details" in results:
            return Search.Returns(error=results["details"])

        # Unknown output.
        return Search.Returns(error="unknown api error")
