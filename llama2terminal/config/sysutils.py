import time
import warnings
import logging
import io
import os
import yaml
import sys
import pkg_resources
import inquirer
from contextlib import contextmanager

# --- ENCODING ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# --- System Dictionaries ---
system_shortnames = {
    'powershell.exe': 'PS',
    'cmd.exe': 'CMD',
}

system_end_msgs = {
    'powershell.exe': '##END_OF_OUTPUT##',
    'cmd.exe': 'REM END_OF_OUTPUT',
}

system_pwd = {
    'powershell.exe': 'pwd | findstr :',
    'cmd.exe': 'cd',
}

system_choices = [
    inquirer.List(
        'system',
        message="Select system to use",
        choices=['powershell.exe', 'cmd.exe'],
    )
]
 
class TerminalColors:
    HEADER = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # reset/no color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    ORANGE = '\033[38;2;255;165;0m'

# --- System Utilities ---

def get_l2t_path():
    
        l2t_path = os.environ.get("L2T_PATH")
        if not l2t_path:
            l2t_path = pkg_resources.resource_filename("llama2terminal", '')
            os.environ["L2T_PATH"] = l2t_path
        return l2t_path

def typing_print(text, delay=0.01):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)

def ensure_hf_token():
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        hf_token = config_yaml['model']['token']
        os.environ['HUGGINGFACE_TOKEN'] = hf_token
        if not hf_token or hf_token == 'HUGGINGFACE_TOKEN':
            raise ValueError("HUGGINGFACE_TOKEN not found")

@contextmanager
def mute(active: bool = True):

    if active:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    else:
        yield
# --- LOADING CONFIG PARAMS ---
config_path = os.path.join(get_l2t_path(), "config", "config.yaml")
with open(config_path, 'r') as stream:
    config_yaml = yaml.safe_load(stream)

# --- LOADING LOGGING ---
logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s",
    level=config_yaml['general']['logging_level']
)
if config_yaml['general']['logging_level'] == "ERROR":
    warnings.filterwarnings("ignore")

logging.getLogger("haystack").setLevel(eval(f"logging.{config_yaml['general']['logging_haystack_level']}"))
