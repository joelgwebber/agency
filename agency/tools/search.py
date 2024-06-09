from __future__ import annotations

from typing import Dict, List, Optional, Union

import requests

from agency.tools import Tool
from agency.tools.annotations import decl, prop, schema

# Just use Tavily for now.
_TAVILY_API_URL = "https://api.tavily.com"


@schema()
class SearchArgs:
    query: str = prop("The url to fetch")
    max_results: int = prop("Maximum number of results", default=5)


class Search(Tool):
    _api_key: str

    def __init__(self, api_key: str):
        Tool.__init__(self)
        self._api_key = api_key
        self.declare(self.search_query)

    @decl("search_query", "Performs a web search.")
    def search_query(self, args: SearchArgs) -> Union[Dict, List[Dict]]:
        raw_json = self._raw_results(args.query, args.max_results)
        return self._clean_results(raw_json)

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
        # Return the repsonse json directly, even if it's an error.
        # TODO: We may want to settle on a standard format for errors, so we can control behavior better.
        return rsp.json()

    def _clean_results(self, results: Dict) -> Union[Dict, List[Dict]]:
        """Clean results from Tavily Search API."""
        if "results" in results:
            clean_results = []
            for result in results["results"]:
                clean_results.append(
                    {
                        "url": result["url"],
                        "content": result["content"],
                    }
                )
            return clean_results

        # Error details.
        elif "details" in results:
            return results["details"]

        # Unknown output.
        return results
