import os
import logging
import yaml
import torch

from llama2terminal.base.prompts import PromptTemplate, Modules
from llama2terminal.wrapper.config import get_l2t_path

from abc import abstractmethod, ABC
from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import NoMemory
from haystack.nodes import PromptNode

class LlamaAgent(ABC):

    def __init__(self):
        self.l2t_path = get_l2t_path()
        config_path = os.path.join(self.l2t_path, "llama2terminal", "base", "config.yaml")
        with open(config_path, 'r') as stream:
            self.config = yaml.safe_load(stream)

        logging.basicConfig(
            format="%(levelname)s - %(name)s -  %(message)s",
            level=self.config['general']['logging_level']
        )

        self.memory = []
        self.prompt_node = self.build_model()
        self.init_model()

    @abstractmethod
    def init_model(self):
        pass

    def build_model(self) -> PromptNode:
        # Initializing
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.config['general']['pytorch_cuda_config']
        torch.cuda.empty_cache()

        # Building Model
        hf_token = self.config['model']['token']
        model_id = self.config['model']['id']
        max_length = self.config['model']['token_length']

        model = AutoModelForCausalLM.from_pretrained(
            model_id, load_in_4bit=True, use_auth_token=hf_token
        )
        model.config.pretraining_tp = 1
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

        pn = PromptNode(
            model_id,
            max_length=max_length,
            model_kwargs={
                'model': model,
                'tokenizer': tokenizer,
                'task_name': 'text2text-generation',
                'device': None,
                'stream': True
            }
        )

        return pn

    @abstractmethod
    def get_prediction(self, query) -> str:
        torch.cuda.empty_cache()
        output = self.prompt_node.run(query)['answers'][0].answer.split('\n')[-1]
        return output


class LlamaConversationalAgent(LlamaAgent):

    def init_model(self):
       self.agent = ConversationalAgent(
            prompt_node=self.prompt_node,
            memory=NoMemory()
        )

    def get_prediction(self, query) -> str:

        torch.cuda.empty_cache()
        history = '\n'.join(self.memory)

        prompt = PromptTemplate(
            name=self.config['agent']['name'],
            history=history,
            modules= [
                Modules.SHORT_ANSWER,
                Modules.ONE_LINE,
                Modules.DO_NOT_MODIFY,
                Modules.USE_TEMPLATE
            ],
            query=query
        )

        output = self.agent.run(str(prompt))['answers'][0].answer.split('\n')[-1]
        self.memory.append("Human:" + query)
        self.memory.append(output)

        return output