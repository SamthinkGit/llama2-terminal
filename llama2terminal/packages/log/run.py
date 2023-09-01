import gc
import torch
import pdb
import sys

from llama2terminal.base.agents.templates import LlamaCommandLineReaderAgent
from llama2terminal.config.sysutils import TerminalColors as c, typing_print
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
                launch.llama_log_model = LlamaCommandLineReaderAgent(cmd_logger=cmd_logger)
                launch.llama_log_model.build_model()
            else:
                print("Llama Logger has been inited previously", file=sys.stderr)
        
        case "query":

            if launch.llama_log_model is None:
                print("Llama model has not been loaded, use llama load first", file=sys.stderr)
                return
            
            query = input("Query: ")
            output = launch.llama_log_model.run(query=query, mute_active=False)
            typing_print(c.BLUE + "(AI):" + output + c.ENDC + '\n')

        case "talk":
            if launch.llama_log_model is None:
                print("Llama model has not been loaded, use llama load first", file=sys.stderr)
                return

            print(f"{c.ORANGE}==== CHATBOX, use 'exit' to escape ===={c.ENDC}")
            while True:
            
                query = input("Query: ")
                if query == "exit":
                    break

                output = launch.llama_log_model.run(query=query, mute_active=True)
                typing_print(c.BLUE + "(AI):" + output + c.ENDC + '\n')
                

        case "free":
            free(launch.llama_log_model)
            launch.llama_log_model = None

        case _:
            print(f"Command {args[1]} not found in log package", file=sys.stderr)

def free(llama):
    llama.free_resources()
    llama = None
    gc.collect()
    torch.cuda.empty_cache()
