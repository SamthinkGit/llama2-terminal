import torch
import logging
import os
import yaml
from haystack.agents.memory import NoMemory
from haystack.agents.conversational import ConversationalAgent, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from haystack.nodes import PromptNode

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=config['general']['logging_level'])

agent_memory = []

def build_llama_agent() -> ConversationalAgent:
    
    # Initializing
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config['general']['pytorch_cuda_config']
    torch.cuda.empty_cache()

    # Building Model
    hf_token = config['model']['token']
    model_id = config['model']['id']
    max_length = config['model']['token_length']

    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, use_auth_token=hf_token)
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

    pn = PromptNode(model_id,
                    max_length=max_length,
                    model_kwargs={'model':model,
                                'tokenizer':tokenizer,
                                'task_name':'text2text-generation',
                                'device':None,
                                'stream':True
                                }
                    )

    conversational_agent = ConversationalAgent(
        prompt_node = pn,
        memory=NoMemory()
    )
    return conversational_agent

def get_prediction(agent,query) -> str:

    torch.cuda.empty_cache()

    history = '\n'.join(agent_memory)

    prompt=f" [INST] Your name is {config['agent']['name']}, {config['agent']['prompt']} [INST]\n \
      [HISTORY] {history} [HISTORY] \n\
      [QUERY] {query} [\QUERY]\n"

    output = agent.run(prompt)['answers'][0].answer.split('\n')[-1]

    agent_memory.append("Human:" + query)
    agent_memory.append(output)
    return output