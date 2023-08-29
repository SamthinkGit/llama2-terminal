import os
import logging
import yaml
import torch
import warnings
import gc

import llama2terminal.base.prompts.templates as template
from llama2terminal.base.prompts.generation import PromptGenerator, DynamicParam
from llama2terminal.wrapper.config import get_l2t_path
from llama2terminal.wrapper.IOUtils import mute
from llama2terminal.wrapper.history import CommandLogger

from abc import abstractmethod, ABC
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.agents.base import PromptTemplate, ToolsManager, Tool
from haystack.agents.conversational import ConversationalAgent, Agent
from haystack.agents.memory import NoMemory, ConversationMemory 
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, PromptNode
from haystack.pipelines import Pipeline

class LlamaAgent(ABC):

    def __init__(self):
        self.l2t_path = get_l2t_path()
        config_path = os.path.join(self.l2t_path, "base", "config.yaml")
        with open(config_path, 'r') as stream:
            self.config = yaml.safe_load(stream)

        logging.basicConfig(
            format="%(levelname)s - %(name)s -  %(message)s",
            level=self.config['general']['logging_level']
        )
        if self.config['general']['logging_level'] == "ERROR":
            warnings.filterwarnings("ignore")
            
        logging.getLogger("haystack").setLevel(eval(f"logging.{self.config['general']['logging_haystack_level']}"))

    def build_model(self) -> PromptNode:

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.config['general']['pytorch_cuda_config']
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            hf_token = self.config['model'].get('token')
            os.environ['HUGGINGFACE_TOKEN'] = hf_token
            if not hf_token or hf_token == 'HUGGINGFACE_TOKEN':
                raise ValueError("HUGGINGFACE_TOKEN not found")

        torch.cuda.empty_cache()

        # Building Model
        model_id = self.config['model']['id']
        max_length = self.config['model']['token_length']

        model = AutoModelForCausalLM.from_pretrained(
            model_id, load_in_4bit=True, token=hf_token
        )
        model.config.pretraining_tp = 1
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = model
        self.tokenizer = tokenizer

        pn = PromptNode(
            model_id,
            max_length=max_length,
            model_kwargs={
                'model': model,
                'tokenizer': tokenizer,
                'task_name': 'text-generation',
                'device': None,
                'stream': True
            }
        )

        return pn

    def get_prediction(self, query) -> str:
        torch.cuda.empty_cache()
        output = self.prompt_node.run(query)['answers'][0].answer.split('\n')[-1]
        return output
    
    def free_resources(self):
        self.tokenizer = None
        self.model = None
        self.prompt_node = None
        torch.cuda.empty_cache()
        gc.collect()


class LlamaConversationalAgent(LlamaAgent):

    def __init__(self, fuse_with: Optional[LlamaAgent] = None):
        torch.cuda.empty_cache()
        super().__init__() 

        if fuse_with:
            self.prompt_node = fuse_with.prompt_node
        else:
            self.prompt_node = self.build_model()

        template = PromptTemplate("{query}")

        memory = ConversationMemory() 
        self.dynamic_memory = DynamicParam()
        self.dynamic_memory.to_dynamic(memory.load)

        self.agent = Agent(
           prompt_node=self.prompt_node,
           prompt_template=template,
           memory=memory,
           max_steps=1,
        )
       

    def get_prediction(self, query: str, mute_active: bool = True) -> str:

        with mute(mute_active):
            torch.cuda.empty_cache()
            self.agent.prompt_template = PromptTemplate(
                str(PromptGenerator(
                    name=self.config['agent']['name'],
                    dynamic_params={
                        'MEMORY': self.dynamic_memory
                    },
                    modules= [
                        template.Modules.BEHAVIOR.CONVERSATIONAL_MODEL,
                        template.Modules.LENGTH.SHORT_ANSWER,
                        template.Modules.LENGTH.ONE_LINE,
                        template.Modules.CHARACTERS.ONLY_ALPHANUMERIC_AND_EMOJIS,
                        template.Modules.SECURITY.DO_NOT_MODIFY_PROMPT,
                        template.Modules.SECURITY.ANTI_REPEAT,
                    ],
                    query="{query}"
                ))
            )

            output = self.agent.run(query=query)['transcript']
            return output

        
    def free_resources(self):
        super().free_resources()
        self.agent = None
        self.dynamic_memory = None
        self.prompt_node = None
        torch.cuda.empty_cache()
        gc.collect()

class LlamaCommandLineReader(LlamaAgent):

    def __init__(self, cmd_logger: CommandLogger, fuse_with: Optional[Any] = None):

        torch.cuda.empty_cache()
        super().__init__() 

        if fuse_with:
            self.prompt_node = fuse_with.prompt_node
        else:
            self.prompt_node = self.build_model()

        self.cmd_logger = DynamicParam()
        self.cmd_logger.to_dynamic(cmd_logger.__str__)

        template = PromptTemplate("{query}")
        self.agent = Agent(
           prompt_node=self.prompt_node,
           prompt_template=template,
           max_steps=1,
        )

    def get_prediction(self, query: str, mute_active: bool = True) -> str:
        
        with mute(mute_active):

            torch.cuda.empty_cache()

            self.agent.prompt_template = PromptTemplate(
                str(PromptGenerator(
                    name=self.config['agent']['name'],
                    dynamic_params={
                        'COMMAND_LINE_LOG': self.cmd_logger
                    },
                    modules= [
                        template.Modules.BEHAVIOR.PROBLEM_RESOLVER_MODEL,
                        template.Modules.LENGTH.APPROPIATE_LENGTH,
                        template.Modules.CHARACTERS.ONLY_ALPHANUMERIC,
                        template.Modules.SECURITY.DO_NOT_MODIFY_PROMPT,
                        template.Modules.SECURITY.ANTI_REPEAT,
                    ],
                    query="{query}"
                    )
                )
            )

            output = self.agent.run(query=query)['transcript']
            return output
    
    def free_resources(self):
        super().free_resources()
        self.agent = None
        self.prompt_node = None
        torch.cuda.empty_cache()
        gc.collect()