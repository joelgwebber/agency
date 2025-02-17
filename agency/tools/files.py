from __future__ import annotations

from dataclasses import dataclass
from typing import List

from agency.schema import (
    Schema,
    Type,
    parse_val,
    prop,
    schema,
    schema_for,
    serialize_val,
)
from agency.tool import Stack, Tool, ToolDecl


@dataclass
class ReadFile(Tool):
    @schema
    class Params:
        file: str = prop("the filename to read")

    @schema
    class Returns:
        lines: List[str] = prop("the file's contents as individual lines")

    root_path: str

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "read-file",
            "reads a file from disk",
            schema_for(ReadFile.Params),
            schema_for(ReadFile.Returns),
        )

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        path = f"{self.root_path}/{args.file}"
        try:
            lines = [
                line.rstrip("\n")
                for line in open(path, "r", encoding="utf-8").readlines()
            ]
            stack.respond(
                serialize_val(ReadFile.Returns(lines=lines), self.decl.returns)
            )
        except OSError as err:
            stack.error(f"{err} reading {args.file}")


@dataclass
class EditFile(Tool):
    @schema
    class Params:
        file: str = prop("the filename to edit")
        old_text: str = prop("the original text to be updated")
        new_text: str = prop("the new text to replace the original")

    root_path: str

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "edit-file",
            "edits a file's contents",
            schema_for(EditFile.Params),
            Schema(Type.Object, ""),
        )

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        path = f"{self.root_path}/{args.file}"
        print("-old text-------------------------\n", args.old_text)
        print("-new text-------------------------\n", args.new_text)
        print("----------------------------------")

        try:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                content = ""

            new_content = content.replace(args.old_text, args.new_text)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            stack.respond({})
        except OSError as err:
            stack.error(f"{err} editing {args.file}")
