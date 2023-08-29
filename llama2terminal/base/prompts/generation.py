from typing import Optional, List, Dict, Any
from abc import abstractmethod, ABC

class DynamicParam(ABC):
    
    def __init__(self, sanitize: Optional[bool] = True):
        self.sanitize = sanitize

    def to_dynamic(self, to_str_callback):
        self._dynamic_str = to_str_callback

    def sanitize_str(self, input: str) -> str:
        return input.replace('\\','\\\\')
        
    def __str__(self) -> str:
        if hasattr(self, "_dynamic_str"):
            if self.sanitize:
                return self.sanitize_str(self._dynamic_str())
            return self._dynamic_str()
        return "Dynamic Param function not implemented"
    
class PromptGenerator:
    
    def __init__(
        self,
        name: Optional[str] = "LLM",
        history: Optional[str] = None,
        hard_params: Optional[Dict[str, Any]] = None,
        dynamic_params: Optional[Dict[str, DynamicParam]] = None,
        modules: Optional[List[str]] = None,
        query: Optional[str] = ""
    ) -> None:
        self.name = name
        self.history = history
        self.hard_params = hard_params
        self.dynamic_params = dynamic_params
        self.modules = modules if modules else []
        self.query = query

    def __str__(self) -> str:

        prompt = f"--- START OF CONTEXT ---"
        prompt = f"[INST] Your name is {self.name} [INST]\n"
        
        if self.modules:
            module_list = '\n- '.join(self.modules)
            prompt += f"[MODULES] You MUST ALWAYS follow these rules:\n- {module_list}\n[MODULES]\n"


        if self.dynamic_params:
            for param, value in self.dynamic_params.items():
               prompt += f"[{param}]\n {str(value)} \n[\{param}]\n"

        if self.hard_params:
           for param, value in self.hard_params.items():
               prompt += f"[{param}] {value} [\{param}]\n"
        if self.history:
            prompt += f"[HISTORY] {self.history} [HISTORY]\n"
        
        prompt += f"[QUERY] {self.query} [QUERY]\n"
        prompt += f"--- END OF CONTEXT ---\n"
        prompt += f"Final Answer: "
        
        return prompt

# Uso
# prompt = PromptTemplate(
#     name = "Sarah",
#     history="Miaw",
#     hard_params = {
#         'CODE': "set x as miaw",
#         'CODE2': "set y as mew"
#     },
#     modules = [
#        MODULES['SHORT_ANSWER'],
#        MODULES['ONLY_CODE'],
#        MODULES['EXAMPLES']
#     ],
#     query="Explain how to code a cat"
# )
# print(prompt.to_raw())
