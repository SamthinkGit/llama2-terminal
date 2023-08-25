import yaml
import os
import importlib.util

from llama2terminal.wrapper.config import get_l2t_path

class PackageLoader:

    def __init__(self):

        self.l2t_path = get_l2t_path()
        yaml_path = os.path.join(self.l2t_path, "packages", "pkg.yaml")

        with open(yaml_path, "r") as file:
            self.packages = yaml.safe_load(file)

        self.loaded_modules = {}

    def unload_module(self, package_name):
        if package_name in self.packages:
            del self.packages[package_name]

    def load_initial_modules(self):
        for package_name, package_info in self.packages.items():
            if package_info.get("load_at_start", False):
                self.run_module(package_name)

    def load_module(self, package_name):

        if package_name in self.loaded_modules:
            return
        
        # Verify package is installled 
        if package_name not in self.packages:
            raise ValueError(f"Package '{package_name}' is not installed.")
        
        package_dir = self.packages[package_name]["dir"]

        # Load "launch.py"
        spec = importlib.util.spec_from_file_location("launch_module", os.path.join(self.l2t_path, "packages", package_dir, "launch.py"))
        launch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch_module)
        
        # Save the package to loaded modules
        self.loaded_modules[package_name] = launch_module

    def run_module(self, package_name, *args):
        """
        Executes the previously loaded package.

        Args:
        - package_name (str): The name of the package you wish to run, e.g., "talk".
        - *args, **kwargs: Arguments and keyword arguments for the 'start' function of the module.
        """

        if package_name not in self.loaded_modules:
            self.load_module(package_name)
        
        package_dir = self.packages.get(package_name, {}).get("dir")
        if not package_dir:
            raise ValueError(f"Directory for package {package_name} has not been found.")

        # Launch run.py
        spec = importlib.util.spec_from_file_location("run_module", os.path.join(self.l2t_path, "packages", package_dir, "run.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__start__(*args)

package_loader = PackageLoader()