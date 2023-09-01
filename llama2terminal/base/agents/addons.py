from abc import abstractmethod, ABC
from haystack.agents.memory import ConversationMemory
from llama2terminal.wrapper.history import CommandLogger
from llama2terminal.base.agents.build import LlamaModel
from llama2terminal.base.prompts.generation import DynamicParam

class Addon(ABC):
    
    def __init__(self, name) -> None:
        self.name = name
        pass

    @abstractmethod
    def build(self, llama: LlamaModel) -> None:
        pass

    @abstractmethod
    def run(self, llama: LlamaModel) -> None:
        pass

    @abstractmethod
    def free(self, llama: LlamaModel) -> None:
        pass

class ConversationalMemoryAddon(Addon):
    
    def __init__(self) -> None:
        super().__init__("ConversationalMemoryAddon")
    
    def build(self, llama: LlamaModel) -> None:

        memory = ConversationMemory()
        self.dynamic_memory = DynamicParam()
        self.dynamic_memory.to_dynamic(memory.load)
        llama.agent.memory = memory
        llama.prompt_generator.dynamic_params['MEMORY'] = self.dynamic_memory

    def run(self, llama: LlamaModel) -> None:
        pass

    def free(self, llama: LlamaModel) -> None:
        pass

class CommandLineReaderAddon(Addon):
    
    def __init__(self, cmd_logger: CommandLogger) -> None:
        self.cmd_logger = cmd_logger
        super().__init__("CommandLineReaderAddon")
    
    def build(self, llama: LlamaModel) -> None:

        self.dynamic_param = DynamicParam()
        self.dynamic_param.to_dynamic(self.cmd_logger.__str__)
        llama.prompt_generator.dynamic_params['COMMAND_LINE_LOG'] = self.dynamic_param

    def run(self, llama: LlamaModel) -> None:
        pass

    def free(self, llama: LlamaModel) -> None:
        pass