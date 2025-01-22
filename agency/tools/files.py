from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from agency.schema import Schema, Type, parse_val, prop, schema, schema_for
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult


@dataclass
class ReadFile(Tool):
    @schema()
    class Params:
        file: str = prop("the filename to read")

    @schema()
    class Returns:
        lines: List[str] = prop("the file's contents as individual lines")

    decl = ToolDecl(
        "read-file",
        "reads a file from disk",
        schema_for(Params),
        schema_for(Returns),
    )

    root_path: str

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, ReadFile.decl.params)
        return _handle_file_operation(
            args.file,
            self.root_path,
            lambda path: {
                "lines": [
                    line.rstrip("\n")
                    for line in open(path, "r", encoding="utf-8").readlines()
                ]
            },
        )


@dataclass
class EditFile(Tool):
    @schema()
    class Params:
        file: str = prop("the filename to edit")
        old_text: str = prop("the original text to be updated")
        new_text: str = prop("the new text to replace the original")

    decl = ToolDecl(
        "edit-file",
        "edits a file's contents",
        schema_for(Params),
        Schema(Type.Object, ""),
    )

    root_path: str

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, EditFile.decl.params)
        print("-old text-------------------------\n", args.old_text)
        print("-new text-------------------------\n", args.new_text)
        print("----------------------------------")
        return _handle_file_operation(
            args.file,
            self.root_path,
            lambda path: (
                {"error": err}
                if (err := _edit_file_content(path, args.old_text, args.new_text))
                else {}
            ),
        )


def _handle_file_operation(
    filename: str, root_path: str, operation: Callable[[str], Dict]
) -> ToolResult:
    """Handle common file operation patterns with error handling.

    Args:
        filename: The file to operate on
        root_path: The root path to prepend
        operation: The actual file operation to perform

    Returns:
        A ToolResult with either the operation result or an error message
    """
    try:
        file_path = f"{root_path}/{filename}"
        return ToolResult(operation(file_path))
    except FileNotFoundError:
        return ToolResult({"error": f"File not found: {filename}"})
    except PermissionError:
        return ToolResult({"error": f"Permission denied accessing file: {filename}"})
    except Exception as e:
        return ToolResult({"error": f"Error accessing file {filename}: {str(e)}"})


def _edit_file_content(path: str, old_text: str, new_text: str) -> Optional[str]:
    """Edit a file's content, replacing old_lines with new_lines.

    Args:
        path: Path to the file
        old_lines: Lines to find and replace
        new_lines: New content to insert

    Returns:
        Error message if operation failed, None if successful
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if old_text not in content:
        return (
            f"Could not find the exact lines to replace in {path}. "
            "Please verify the old_lines match exactly."
        )

    new_content = content.replace(old_text, new_text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return None
