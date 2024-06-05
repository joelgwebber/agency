from typing import Dict, List, Optional

import requests

from agency.tools import Decl, Prop, Tool, Type

# Just use Tavily for now.
_TAVILY_API_URL = "https://api.tavily.com"


class Search(Tool):
    _api_key: str

    def __init__(self, api_key: str):
        Tool.__init__(self)
        self._api_key = api_key

        self._add_decl(
            Decl(
                self.search_query,
                "search_query",
                "Performs an web search.",
                {
                    "query": Prop(Type.String, "The url to fetch"),
                    "max_results": Prop(
                        Type.String, "The maximum number of results to fetch"
                    ),
                },
            )
        )

    def search_query(self, query: str, max_results: int) -> list[Dict]:
        raw_json = self._raw_results(query, max_results)
        return self._clean_results(raw_json["results"])

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
        response = requests.post(
            # type: ignore
            f"{_TAVILY_API_URL}/search",
            json=params,
        )
        response.raise_for_status()
        return response.json()

    def _clean_results(self, results: List[Dict]) -> List[Dict]:
        """Clean results from Tavily Search API."""
        clean_results = []
        for result in results:
            clean_results.append(
                {
                    "url": result["url"],
                    "content": result["content"],
                }
            )
        return clean_results
