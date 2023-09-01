import os
import yaml
from llama2terminal.wrapper.terminal import ShellWrapper
from llama2terminal.config.sysutils import get_l2t_path, config_yaml
from llama2terminal.config.welcomeLauncher import generate_config


def main():    

    l2t_path = get_l2t_path()
    config_path = os.path.join(l2t_path, "config", "config.yaml")
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    if config_data['general']['welcome']:
        print("It looks like its your first time here.")
        print("Loading Welcome settings...")
        config_data['general']['welcome'] = False
        with open(config_path, 'w') as outfile:
            yaml.safe_dump(config_data, outfile)
        generate_config()

    terminal = ShellWrapper()
    terminal.cmdloop()

if __name__ == "__main__":
    main()