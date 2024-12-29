from rich.console import Console

from agency.agency import Agency


class AgencyUI:
    _agency: Agency
    _tool_id: str

    def __init__(self, agency: Agency, tool_id: str):
        self._agency = agency
        self._tool_id = tool_id

    def run(self):
        console = Console()

        while True:
            try:
                user_input = input("> ").strip()
                match user_input.lower():
                    case "done" | "quit" | "exit":
                        break
                    case "":
                        continue
                    case _:
                        response = self._agency.ask(
                            self._tool_id, {"question": user_input}
                        )
                        # md = Markdown(response + "\n")
                        console.print(response)
            except (EOFError, KeyboardInterrupt):
                break
