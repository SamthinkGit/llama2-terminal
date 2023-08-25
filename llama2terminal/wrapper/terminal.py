import cmd2
import os
import inquirer
import config
import yaml

from llama2terminal.wrapper.colors import TerminalColors
from llama2terminal.wrapper.history import CommandLogger
from llama2terminal.wrapper.io import CommandLineReader

class ShellWrapper(cmd2.Cmd):

    def __init__(self):
        super().__init__()
        with open('params.yaml', 'r') as file:
            self.params = yaml.safe_load(file)
        self.listening = False
        self.sys_type = self.params['DEFAULT']['system']
        self.cmd_logger = CommandLogger()
        self.change_prompt()

        self.clr = CommandLineReader(self.sys_type)

    def change_prompt(self):
        color = TerminalColors.GREEN if self.listening else TerminalColors.ENDC
        self.prompt = f'[{color}Llama2{TerminalColors.ENDC}] {config.system_shortnames[self.sys_type]} {os.getcwd()}> '
    
    def default(self, statement):
        output, error = self.clr.run_command(statement.raw)
        self.poutput(output)
        self.perror(error)
        if self.listening:
            self.cmd_logger.log_command(statement.raw, output=output, error=error)

    def do_exit(self, args):
        self.clr.close()
        return True

    def do_llama(self, args):
        match args:
            case "listen":
                self.listening = True
                self.change_prompt()
            case "pause":
                self.listening = False
                self.change_prompt()
            case "sys":
                self.sys_type = inquirer.prompt(config.system_choices)['system']
                self.clr.close()
                self.clr = CommandLineReader(self.sys_type)
                self.change_prompt()
            case "log":
                self.cmd_logger.display_log()
            case "clear":
                self.cmd_logger.clear()
            case "stop":
                self.clr.close()
                return True
