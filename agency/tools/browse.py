import io
from typing import List, Optional
from urllib.parse import urljoin

from playwright.sync_api import Response, sync_playwright
from unstructured.partition.auto import partition

from agency.tools import Func, Schema, Tool, Type


class Browse(Tool):

    def __init__(self):
        Tool.__init__(self)

        self._add_func(
            Func(
                self.browse_url,
                "browse_url",
                "Returns the contents at the specified URL.",
                {
                    "url": Schema(
                        Type.String,
                        "The url to fetch",
                    ),
                },
            )
        )

    def browse_url(self, url: str) -> str:
        with sync_playwright() as p:
            try:
                # Fetch the content using a real browser via playwright, as bytes.
                browser = p.chromium.launch(headless=False)
                context = browser.new_context()
                # Malenia.apply_stealth(context) # Necessary for some pages to load?
                page = context.new_page()
                rsp = page.goto(url)
                content_type = _content_type(rsp)
                file = io.BytesIO(bytes(page.content(), "UTF-8"))
                browser.close()
            except Exception as e:
                return repr(e)

            # Parse out the elements using unstructured.partition.
            # TODO: Why isn't this getting navigation links?
            texts: List[str] = []
            elems = partition(file=file, content_type=content_type)
            for elem in elems:
                meta = elem.metadata
                if elem.text != "":
                    texts.append(elem.text)
                if meta.link_texts is not None and meta.link_urls is not None:
                    for idx, text in enumerate(meta.link_texts or []):
                        # TODO: Deal with <base> tag?
                        resolved_url = urljoin(url, meta.link_urls[idx])
                        text = text or ""
                        texts.append(f"[{text.strip()}]({resolved_url})")

            return "\n".join(texts)


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
