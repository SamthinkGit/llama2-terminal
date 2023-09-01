import torch
import sys
import os
import gc

import llama2terminal.config.sysutils as sysutils

from llama2terminal.base.prompts.generation import DynamicParam, PromptGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.nodes import PromptNode
from haystack.nodes.prompt.prompt_template import PromptTemplate
from haystack.agents.memory import Memory
from haystack.agents.base import Agent, ToolsManager
from typing import Optional, Any, List

class LlamaModel():
    
    def __init__(self):

        sysutils.ensure_hf_token()
        pass 

    
    def build_model(
        self,
        fuse: Optional[Any] = None,
        name: Optional[str] = sysutils.config_yaml['agent']['name'],
        prompt_generator: Optional[PromptGenerator] = None,
        tools: Optional[List[Any]] = [],
        memory: Optional[Memory] = None,
        max_steps: Optional[int] = 8,
        addons: Optional[List[Any]] = []
    ):

        torch.cuda.empty_cache()
        model_id = sysutils.config_yaml['model']['id']
        max_length = sysutils.config_yaml['model']['token_length']
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = sysutils.config_yaml['general']['pytorch_cuda_config']

        model = AutoModelForCausalLM.from_pretrained(
            model_id, load_in_4bit=True, token=hf_token
        )

        model.config.pretraining_tp = 1
        tokenizer = AutoTokenizer.from_pretrained(model_id)
#        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.prompt_node = fuse
        self.addons = addons

        if self.prompt_node is None:
            self.fused = False
            self.prompt_node = PromptNode(
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
        else:
            self.fused = True
        
        self.prompt_generator = prompt_generator
        if self.prompt_generator is None:
            self.prompt_generator = PromptGenerator(
                name=name,
                hard_params=dict(),
                dynamic_params=dict(),
                modules=list(),
                query="{query}"
            )
        
        self.agent = Agent(
            prompt_node=self.prompt_node,
            prompt_template=PromptTemplate("{query}"),
            tools_manager=ToolsManager(self.tools),
            memory = self.memory,
            max_steps=self.max_steps
        )

        for addon in self.addons:
            addon.build(llama=self)


    def run(self, query: str, mute_active: bool = True) -> str:


        with sysutils.mute(mute_active):
            torch.cuda.empty_cache()

            for addon in self.addons:
                addon.run(llama=self)

            self.agent.prompt_template = PromptTemplate(str(self.prompt_generator))
            output = self.agent.run(query=query)['transcript']
        
        return output

    def free(self) -> None:
        if not self.fused:
            self.model = None
            self.tokenizer = None

        self.tools = None
        self.memory = None
        self.max_steps = None
        self.prompt_node = None
        self.prompt_generator = None
        self.addons = None
        self.agent = None

        torch.cuda.empty_cache()
        gc.collect()
        