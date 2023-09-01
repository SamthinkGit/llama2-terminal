from datetime import datetime
from typing import List, Optional
from llama2terminal.config.sysutils import TerminalColors

class CommandEntry:
    def __init__(self, command: str):
        self.command = command
        self.timestamp = datetime.now()
        self.output = None
        self.error = None

    def set_output(self, output: str):
        self.output = output

    def set_error(self, error: str):
        self.error = error

    def to_raw(self) -> str:
        return ("<(TIMESTAMP)" + str(self.timestamp) + "(\\TIMESTAMP)\n" +
            "(COMMAND)" + self.command + "(\\COMMAND)\n" +
            "(OUTPUT)" + str(self.output) + "(\\OUTPUT)\n" +
            "(ERROR)" + str(self.error) + "(\\ERROR)>")


    def __str__(self):
        return f"{TerminalColors.ORANGE}{self.timestamp}: {self.command}\n \
            {TerminalColors.GREEN}Output:\n{TerminalColors.ENDC}{self.output}\n \
            {TerminalColors.WARNING}Error:\n{self.error} {TerminalColors.ENDC}"

class CommandLogger:
    def __init__(self):
        self.commands: List[CommandEntry] = []

    def log_command(self, command: str, output: Optional[str] = None, error: Optional[str] = None):
        entry = CommandEntry(command)
        if output:
            entry.set_output(output)
        if error:
            entry.set_error(error)
        self.commands.append(entry)

    def clear(self):
        self.commands.clear()

    def to_raw(self) -> List[str]:
        raw_logs = []
        for entry in self.commands:
            raw_logs.append(entry.to_raw())
        return raw_logs

    def display_log(self):
        for entry in self.commands:
            print(entry)
            print('=' * 50)

    def __str__(self):
        raw_logs = [cmd.to_raw() for cmd in self.commands]
        return '\n'.join(raw_logs)