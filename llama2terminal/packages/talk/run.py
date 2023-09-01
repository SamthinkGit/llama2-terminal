from llama2terminal.base.agents.templates import LlamaConversationalAgent
from llama2terminal.config.sysutils import TerminalColors as c, typing_print, config_yaml as y
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

            typing_print(c.BLUE + f"({y['agent']['name']}): " + llama.run(query, mute_active=True) + c.ENDC + '\n')
    
    except Exception as e:
        print("Exception: ")
        traceback.print_exc()
        free(llama)
        
def free(llama):
    llama.free()
    llama = None
    gc.collect()
    torch.cuda.empty_cache()
