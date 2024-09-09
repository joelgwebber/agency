import readline  # Do not remove; affects the import() function.

from rich.console import Console
from rich.markdown import Markdown

from agency.agency import Agency


class AgencyUI:
    _agency: Agency

    def __init__(self, agency: Agency):
        self._agency = agency

    def run(self):
        console = Console()

        while True:
            try:
                user_input = input("> ").strip()
                match user_input.lower():
                    case "done" | "quit" | "exit":
                        break
                    case "history":
                        console.print(self._agency.history)
                    case "":
                        continue
                    case _:
                        for response in self._agency.ask(user_input):
                            md = Markdown(response + "\n")
                            console.print(md)
            except (EOFError, KeyboardInterrupt):
                break
