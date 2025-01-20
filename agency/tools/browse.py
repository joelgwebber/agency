import io
from typing import List, Optional
from urllib.parse import urljoin

from playwright.sync_api import Response, sync_playwright
from unstructured.partition.auto import partition

from agency.schema import parse_val, prop, schema, schema_for
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult


# NOTE: The unstructured partition() function is skipping some elements for HTML,
# including the use of <figure>, which is quite common on Wikipedia and other sources.
# Filed a bug here: https://github.com/Unstructured-IO/unstructured/issues/3606
class Browse(Tool):
    @schema()
    class Params:
        url: str = prop("url to fetch")

    @schema()
    class Returns:
        text: str = prop("text representation of the url content")

    decl = ToolDecl(
        "browse-url",
        "Returns the contents at the specified URL.",
        schema_for(Params),
        schema_for(Returns),
    )

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, Browse.decl.params)

        with sync_playwright() as p:
            try:
                # Fetch the content using a real browser via playwright, as bytes.
                browser = p.chromium.launch(headless=False)
                context = browser.new_context()

                # TODO: Necessary for some pages to load?
                # Malenia.apply_stealth(context)

                page = context.new_page()
                rsp = page.goto(args.url)
                content_type = _content_type(rsp)
                file = io.BytesIO(bytes(page.content(), "UTF-8"))
                browser.close()
            except Exception as e:
                return ToolResult({"error": repr(e)})

            # Parse out the elements using unstructured.partition.
            # TODO: Why isn't this getting navigation links or images?
            texts: List[str] = []
            elems = partition(file=file, content_type=content_type)
            for elem in elems:
                meta = elem.metadata

                # Text
                if elem.text != "":
                    texts.append(elem.text)

                # Links
                if meta.link_texts is not None and meta.link_urls is not None:
                    for idx, text in enumerate(meta.link_texts):
                        texts.append(_format_link(args.url, meta.link_urls[idx], text))

                # Images
                if meta.image_path is not None:
                    print(f"--> image: {elem}")
                    texts.append("!" + _format_image(args.url, meta.image_path))

            return ToolResult({"text": "\n".join(texts)})


def _format_link(base: str, url: str, text: str) -> str:
    # TODO: Deal with <base> tag.
    resolved_url = urljoin(base, url)
    text = text or ""
    return f"[{text.strip()}]({resolved_url})"


def _format_image(base: str, url: str) -> str:
    # TODO: Deal with <base> tag.
    resolved_url = urljoin(base, url)
    return f"![[{resolved_url}]]"


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
