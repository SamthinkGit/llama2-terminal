import gc
import torch
import sys

from llama2terminal.base.agents.agents import LlamaCommandLineReader
from llama2terminal.wrapper.colors import TerminalColors as c
from llama2terminal.wrapper.utils import typing_print
import llama2terminal.wrapper.terminal as wrapper
import llama2terminal.packages.log.launch as launch


def __start__(args):

    cmd_logger = wrapper.ShellWrapper.terminal.cmd_logger
    if len(args) == 1:
        cmd_logger.display_log()
        return

    match args[1]:
        case "load":
            if launch.llama_log_model is None:
                launch.llama_log_model = LlamaCommandLineReader(cmd_logger)
            else:
                print("Llama Logger has been inited previously", file=sys.stderr)
        
        case "query":
            query = input("Query: ")
            output = launch.llama_log_model.get_prediction(query=query, mute_active=False)
            print(output)

        case "free":
            free(launch.llama_log_model)

        case _:
            print("Command not found")

def free(llama):
    llama.free_resources()
    llama = None
    gc.collect()
    torch.cuda.empty_cache()
