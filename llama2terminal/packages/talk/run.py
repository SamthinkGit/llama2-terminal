from llama2terminal.base.agents.templates import LlamaConversationalAgent
from llama2terminal.config.sysutils import TerminalColors as c, typing_print
import traceback
import gc
import torch

def __start__(args):
    
    llama = LlamaConversationalAgent()
    llama.build_model()
    print(f"{c.ORANGE}==== CHATBOX, use 'exit' to escape ===={c.ENDC}")
    try:
        while True:
            query = input(f"> ")
            if query == "exit":
                free(llama)
                break

            typing_print(c.BLUE + "(AI):" + llama.run(query) + c.ENDC + '\n')
    
    except Exception as e:
        print("Exception: ")
        traceback.print_exc()
        free(llama)
        
def free(llama):
    llama.free()
    llama = None
    gc.collect()
    torch.cuda.empty_cache()
