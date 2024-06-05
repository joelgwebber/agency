import io
import json
from typing import List, Optional
from urllib.parse import urljoin

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from playwright.sync_api import Response, sync_playwright
from unstructured.partition.auto import partition


class BrowseInput(BaseModel):
    url: str = Field(description="")


class BrowseTool(BaseTool):
    name: str = "browse_url"
    description: str = """
        Browse to a URL, extract text and outgoing links.
        Useful when you need to get the content at a URL, regardless of the resource type contained there.
        Input must be a url to browse to.
        Output will be markdown-formatted text extracted from the resource, including outgoing links.
    """

    def _run(self, base_url: str) -> str:
        with sync_playwright() as p:
            try:
                # Fetch the content using a real browser via playwright, as bytes.
                browser = p.chromium.launch(headless=False)
                context = browser.new_context()
                # Malenia.apply_stealth(context) # Necessary for some pages to load?
                page = context.new_page()
                rsp = page.goto(base_url)
                content_type = _content_type(rsp)
                file = io.BytesIO(bytes(page.content(), "UTF-8"))
                browser.close()
            except Exception as e:
                return repr(e)

            # Parse out the elements using unstructured.partition.
            texts: List[str] = []
            elems = partition(file=file, content_type=content_type)
            for elem in elems:
                meta = elem.metadata
                if elem.text != "":
                    texts.append(elem.text)
                if meta.link_texts is not None and meta.link_urls is not None:
                    for idx, text in enumerate(meta.link_texts or []):
                        # TODO: Deal with <base> tag?
                        url = urljoin(base_url, meta.link_urls[idx])
                        text = text or ""
                        texts.append(f"[{text.strip()}]({url})")

            rsp = {"content": "\n".join(texts)}
            return json.dumps(rsp)


def _content_type(rsp: Optional[Response]) -> str:
    content_type = "text/html"
    if rsp is not None:
        header = rsp.header_value("Content-Type")
        if header is not None:
            # We only want the type, not the optional charset bits.
            idx = header.index(";")
            if idx >= 0:
                header = header[0:idx]
            content_type = header
    return content_type
