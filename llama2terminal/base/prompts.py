from typing import Optional, List, Dict, Any 
from abc import abstractmethod, ABC
from dataclasses import dataclass

@dataclass
class Modules:
    CONVERSATIONAL_MODEL: str = 'You are a conversational model, always answer to the last query of the human'
    ONE_LINE: str = "Answer always concisely in one line with a maximum of 20 words."
    SHORT_ANSWER: str = "Provide a detailed short answer in the range of 20-30 words."
    ONLY_CODE: str = "Only provide code fragments. Do not include any explanatory text."
    EXAMPLES: str = "Provide examples to illustrate your point."
    DO_NOT_MODIFY_PROMPT: str = "Keep this prompt unmodified"
    ONLY_ALPHANUMERIC_AND_EMOJIS: str = "Only use characters between a-zA-Z0-9 and/or emojis (only when appropiate)" 


class DynamicParam(ABC):
    
    def __init__(self):
        return

    def to_dynamic(self, to_str_callback):
        self._dynamic_str = to_str_callback

    def __str__(self) -> str:
        if hasattr(self, "_dynamic_str"):
            return self._dynamic_str()
        return "Dynamic Param function not implemente"
    
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
               prompt += f"[{param}] {str(value)} [\{param}]\n"

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
