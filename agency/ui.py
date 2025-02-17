import readline

from rich.console import Console

from agency.agent import Agent

# The readline import is seemingly unused, but imported for the side-effect of using readline() for input on Unix-like systems.
_ = readline


class AgentUI:
    _agent: Agent

    def __init__(self, agency: Agent):
        self._agent = agency

    def run(self):
        console = Console()

        # TODO: ^C to interrupt generation.
        stack = self._agent.start()
        while True:
            try:
                user_input = input("> ").strip()
                match user_input.lower():
                    case "done" | "quit" | "exit":
                        # Done/quit/exit/^D will quit.
                        break
                    case "":
                        # Ignore empty inputs.
                        continue
                    case _:
                        # Ask a question.
                        response = self._agent.ask(stack, user_input)
                        console.print(response)
            except EOFError:
                break
