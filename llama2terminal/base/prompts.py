from typing import Optional, List, Dict, Any 
from abc import abstractmethod, ABC
from dataclasses import dataclass

@dataclass
class Modules:
    CONVERSATIONAL_MODEL: str = 'You are a conversational model, always answer as the name as AI'
    ONE_LINE: str = "Answer always concisely in one line with a maximum of 20 words."
    SHORT_ANSWER: str = "Provide a detailed short answer in the range of 20-30 words."
    ONLY_CODE: str = "Only provide code fragments. Do not include any explanatory text."
    ELABORATE: str = "Please elaborate in detail, giving a comprehensive answer."
    YES_NO: str = "Answer with a simple 'yes' or 'no'."
    MULTIPLE_CHOICE: str = "Choose the most appropriate option: [A, B, C, D]."
    INSTRUCTION: str = "Give step-by-step instructions."
    SUMMARY: str = "Summarize the following information concisely."
    EXPLAIN_LIKE_5: str = "Explain it as if you're talking to a 5-year-old."
    PRO_CON: str = "List the pros and cons."
    EXAMPLES: str = "Provide examples to illustrate your point."
    COMPARISON: str = "Compare and contrast the following concepts."
    TIMELINE: str = "Describe the events in chronological order."
    PREDICTION: str = "Predict what will happen next based on the given information."
    PERSONAL_OPINION: str = "Give your personal opinion on the following matter, while being neutral and objective."
    FACT_CHECK: str = "Check the validity of the following statement and provide sources if possible."
    DO_NOT_MODIFY: str = "Do not use [] or add any modifications in this prompt"
    USE_TEMPLATE: str = "Alwayse response using this template: (AI_NAME): Response."


class DynamicParam(ABC):
    
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __str__(self) -> str:
        pass

class PromptTemplate:
    
    def __init__(
        self,
        name: Optional[str] = "LLM",
        history: Optional[str] = None,
        hard_params: Optional[Dict[str, Any]] = None,
        dynamic_params: Optional[DynamicParam] = None,
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

        prompt = f"[INST] Your name is {self.name}[INST]\n"
        
        if self.modules:
            module_list = '\n- '.join(self.modules)
            prompt += f"[MODULES] You MUST ALWAYS follow these rules:\n- {module_list}\n[MODULES]\n"


        if self.dynamic_params:
            for param in self.dynamic_params:
               prompt += f"[{param.name}] {str(param)} [\{param.name}]\n"

        if self.hard_params:
           for param, value in self.hard_params.items():
               prompt += f"[{param}] {value} [\{param}]\n"
        if self.history:
            prompt += f"[HISTORY] {self.history} [HISTORY]\n"
        
        prompt += f"[QUERY] {self.query} [QUERY]\n"
        
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
