import sys
import os
import yaml

# Add src folder to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from utils.config import parse_args
from verifier.verification_cfg import multiple_experiments

if __name__ == '__main__':
    args = parse_args()
    
    # Resolve absolute path to config folder
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
    
    # Load global configuration
    global_cfg_file = os.path.join(config_path, "global_cfg.yaml")
    if not os.path.exists(global_cfg_file):
        raise FileNotFoundError(f"Global config file not found: {global_cfg_file}")
    
    with open(global_cfg_file, 'r') as file:
        global_cfg = yaml.safe_load(file)
    
    # Pass the absolute config path to the multiple_experiments function
    multiple_experiments(args, global_cfg, config_path=config_path)
