import os
import yaml
from inquirer import prompt, List, Text
from llama2terminal.wrapper.colors import TerminalColors
from llama2terminal.wrapper.config import get_l2t_path

def generate_config():
    # Display banner
    print(f"{TerminalColors.ORANGE}==================================={TerminalColors.ENDC}")
    print(f"{TerminalColors.ORANGE}       Llama2 Terminal Config      {TerminalColors.ENDC}")
    print(f"{TerminalColors.ORANGE}==================================={TerminalColors.ENDC}")

    # Welcome message
    print("\nWelcome to the Llama2 Terminal Configuration Tool!")
    print("We'll ask you for some data to set up your environment.\n")

    # Choose the model
    questions = [
        List('model',
             message="Choose a Llama model",
             choices=[
                 'meta-llama/Llama-2-7b-hf',
                 'meta-llama/Llama-2-7b-chat-hf (recommended)',
                 'meta-llama/Llama-2-13b-hf',
                 'meta-llama/Llama-2-13b-chat-hf (recommended for PCs with >12GB VRAM)',
                 'meta-llama/Llama-2-70b-hf',
                 'meta-llama/Llama-2-70b-chat-hf (most effective, very heavy)'
             ])
    ]
    answers = prompt(questions)
    model_id = answers['model'].split()[0]

    # Check for Hugging Face token in environment variables
    hugging_face_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hugging_face_token:
        questions = [
            Text('token', message="Enter your Hugging Face token")
        ]
        answers = prompt(questions)
        hugging_face_token = answers['token']

    # Choose a name for the model
    questions = [
        Text('name', message="Choose a name for your model (Sarah by default)", default="Sarah")
    ]
    answers = prompt(questions)
    agent_name = answers['name']

    # Choose the system
    questions = [
        List('system',
             message="¿In which system are you working?",
             choices=[
              'powershell.exe',
              'cmd.exe',
             ])
    ]
    answers = prompt(questions)
    system_choice = answers['system']

    # Update the YAML files
    l2t_path = get_l2t_path()
    config_path = os.path.join(l2t_path, "base", "config.yaml")
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
        config_data['model']['id'] = model_id
        config_data['model']['token'] = hugging_face_token
        config_data['agent']['name'] = agent_name
        with open(config_path, 'w') as outfile:
            yaml.safe_dump(config_data, outfile)

    params_path = os.path.join(l2t_path, "wrapper", "params.yaml")
    with open(params_path, 'r') as file:
        params_data = yaml.safe_load(file)
        params_data['DEFAULT']['system'] = system_choice
        with open(params_path, 'w') as outfile:
            yaml.safe_dump(params_data, outfile)

    # Completion message
    print("\nYour Llama2 Terminal is now configured!")
    print("If you need to change this settings, you can run <llama config>'.")
    print("Enjoy using Llama2 Terminal!")