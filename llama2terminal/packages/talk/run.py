from llama2terminal.base.agents import LlamaConversationalAgent
from llama2terminal.wrapper.colors import TerminalColors as c
from llama2terminal.wrapper.utils import typing_print
import gc
import torch

def __start__(*args):
    
    llama = LlamaConversationalAgent()
    print(f"{c.ORANGE}==== CHATBOX, use 'exit' to escape ===={c.ENDC}")
    try:
        while True:
            query = input(f"> ")
            if query == "exit":
                free(llama)
                break
            typing_print(c.BLUE + "(AI):" + llama.get_prediction(query) + c.ENDC + '\n')
    
    except Exception as e:
        free(llama)
        
def free(llama):
    llama.free_resources()
    llama = None
    gc.collect()
    torch.cuda.empty_cache()
