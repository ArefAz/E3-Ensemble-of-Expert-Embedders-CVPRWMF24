import numpy as np


def create_configs_dict(configs):
    return configs

def initalize_configs(configs):
    """
    sets some of the configs based on other configs
    Also it sets some default values that I don't want to put in the yaml file
    for example, if I want to run the test with some variation that I don't want during training
    In that case, I can just set the config key in the corresponding test script to true
    """
    configs = create_configs_dict(configs)
    configs["data_size_exp"] = False
    configs["is_test"] = False
    configs["Data"]["patch_size"] = configs["Model"]["patch_size"]
    configs["Model"]["jpeg_num_classes"] = len(configs["Data"]["jpeg_quality"])
    # assert len(configs["Model"]["manipulation_ckpts"]) == len(configs["Model"]["manipulations"])
    if configs["Data"]["randomize_jpeg_quality"]:
        configs["Data"]["quality"] = list(np.arange(70, 100, 1))
    return configs
