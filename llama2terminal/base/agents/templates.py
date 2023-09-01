import torch
import sys
import gc

import llama2terminal.base.agents.addons as addons
from llama2terminal.wrapper.history import CommandLogger
from llama2terminal.base.prompts.templates import Modules as mod
from llama2terminal.base.agents.build import LlamaModel

from haystack.agents.memory import ConversationMemory

from abc import ABC, abstractmethod
from typing import Optional, Any

class LlamaTemplateAgent(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def build_model(self, fuse: Optional[Any] = None) -> Optional[Any]:
        pass

    @abstractmethod
    def run(self, query: str, mute_active: bool = True) -> str:
        pass
    
    @abstractmethod
    def free(self) -> None:
        pass
        
class LlamaConversationalAgent(LlamaTemplateAgent):
    
    def __init__(self) -> None:
        pass

    def build_model(self, fuse: Any | None = None) -> Any | None:
        llama = LlamaModel()
        llama.build_model(
            fuse=fuse,
            max_steps=1,
            memory=ConversationMemory(),
            addons=[addons.ConversationalMemoryAddon()]
        )

        llama.prompt_generator.modules += [
            mod.BEHAVIOR.CONVERSATIONAL_MODEL,
            mod.LENGTH.SHORT_ANSWER,
            mod.LENGTH.ONE_LINE,
            mod.CHARACTERS.ONLY_ALPHANUMERIC_AND_EMOJIS,
            mod.SECURITY.DO_NOT_MODIFY_PROMPT,
            mod.SECURITY.ANTI_REPEAT,
            mod.SECURITY.ANTI_BLOCK,
        ]

        self.llama = llama
        
    def run(self, query: str, mute_active: bool = True) -> str:
        return self.llama.run(query=query, mute_active=mute_active)

    def free(self) -> None:
        self.llama.free()

class LlamaCommandLineReaderAgent(LlamaTemplateAgent):
    
    def __init__(self, cmd_logger: CommandLogger) -> None:
        self.cmd_logger = cmd_logger
        super().__init__()

    def build_model(self, fuse: Any | None = None) -> Any | None:
        llama = LlamaModel()
        llama.build_model(
            fuse=fuse,
            max_steps=1,
            memory=ConversationMemory(),
            addons=[
                addons.ConversationalMemoryAddon(),
                addons.CommandLineReaderAddon(cmd_logger=self.cmd_logger)
            ]
        )

        llama.prompt_generator.modules += [
            mod.BEHAVIOR.PROBLEM_RESOLVER_MODEL,
            mod.TEMPLATES.CMD_LOGGER_READER,
            mod.LENGTH.APPROPIATE_LENGTH,
            mod.CHARACTERS.ONLY_ALPHANUMERIC,
            mod.SECURITY.DO_NOT_MODIFY_PROMPT,
            mod.SECURITY.ANTI_REPEAT,
            mod.SECURITY.ANTI_BLOCK,
        ]

        self.llama = llama

    def run(self, query: str, mute_active: bool = True) -> str:
        return self.llama.run(query=query, mute_active=mute_active)

    def free(self) -> None:
        self.llama.free()