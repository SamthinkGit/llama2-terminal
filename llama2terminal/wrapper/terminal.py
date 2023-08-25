import cmd2
import os
import inquirer
import yaml


import llama2terminal.wrapper.config as config
from llama2terminal.wrapper.colors import TerminalColors
from llama2terminal.wrapper.history import CommandLogger
from llama2terminal.wrapper.IOUtils import CommandLineReader
from llama2terminal.packages.load import package_loader

class ShellWrapper(cmd2.Cmd):

    def __init__(self):

        super().__init__()

        self.l2t_path = config.get_l2t_path()
        yaml_path = os.path.join(self.l2t_path, "wrapper", "params.yaml")
        with open(yaml_path) as file:
            self.params = yaml.safe_load(file)

        self.listening = False
        self.sys_type = self.params['DEFAULT']['system']
        self.cmd_logger = CommandLogger()
        self.clr = CommandLineReader(self.sys_type)
        self.change_prompt()

    def change_prompt(self):
        color = TerminalColors.GREEN if self.listening else TerminalColors.ENDC
        pwd = self.clr.get_pwd().replace('\n','')
        self.prompt = f"[{color}Llama2{TerminalColors.ENDC}] {config.system_shortnames[self.sys_type]} {pwd}> "
    
    def default(self, statement):
        output, error = self.clr.run_command(statement.raw)
        if len(output) > 0:
            self.poutput(output)
        if len(error) > 0:
            self.perror(error)
        if self.listening:
            self.cmd_logger.log_command(statement.raw, output=output, error=error)
        self.change_prompt()

    def do_exit(self, args):
        self.clr.close()
        return True

    def do_llama(self, args):

        split_args = args.split()
        if not split_args:
            return

        match split_args[0]:
            case "listen":
                self.listening = True
            case "pause":
                self.listening = False
            case "sys":
                self.sys_type = inquirer.prompt(config.system_choices)['system']
                self.clr.close()
                self.clr = CommandLineReader(self.sys_type)
            case "log":
                self.cmd_logger.display_log()
            case "clear":
                self.cmd_logger.clear()
            case "stop":
                self.clr.close()
                return True
            case _:
                try:
                    package_loader.run_module(split_args[0],split_args[1:])
                except ValueError as e:
                    self.perror(e)
        
        self.change_prompt()

                
