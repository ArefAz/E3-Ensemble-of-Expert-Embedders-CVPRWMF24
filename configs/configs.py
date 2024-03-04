import argparse
import yaml
from utils.config_utils import initalize_configs

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--config",
    "-c",
    type=str,
    required=False,
    help="config file path",
    default="configs/configs.yaml",
)
args = arg_parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

with open("configs/configs_ft.yaml", "r") as f:
    configs_ft = yaml.safe_load(f)

with open("configs/continual_configs.yaml", "r") as f:
    continual_configs = yaml.safe_load(f)

configs_dict = initalize_configs(config)
configs_ft = initalize_configs(configs_ft)
