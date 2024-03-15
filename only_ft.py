from train import train
from test import run_test as test
from configs import ft_configs, continual_configs as cl_configs
from utils.continual_utils import fill_configs_with_datasets

import pickle
import numpy as np
from copy import deepcopy


if __name__ == "__main__":
    cl_configs = deepcopy(cl_configs)
    ft_configs["Model"]["expert_ckpt"] = cl_configs["Model"]["ft_ckpt_paths"][0]
    ft_configs["Model"]["src_ckpts"].append(cl_configs["Model"]["ft_ckpt_paths"][0])
    ft_configs["Model"]["classifier"] = cl_configs["Model"]["backbone"]
    ft_configs["Model"]["fine_tune"] = True
    ft_configs["Model"]["model_type"] = "expert"
    ft_configs["Train"]["epochs"] = cl_configs["Train"]["epochs"]
    ft_configs["Train"]["lr"] = cl_configs["Train"]["lr"]
    ft_configs["General"]["check_val_every_n_epoch"] = cl_configs["General"][
        "check_val_every_n_epoch"
    ]
    assert (
        ft_configs["General"]["check_val_every_n_epoch"]
        <= ft_configs["Train"]["epochs"]
    ), "Check val every n epoch should be less than or equal to the number of epochs"
    acc_matrix = []
    auc_matrix = []
    seen_datasets = [cl_configs["Data"]["synthetic_dataset_names"][0]]
    ft_configs["Train"]["train_dataset_limit_per_class"] = (
                cl_configs["Data"]["memory_size"] // 2
            )
    ft_configs["Train"]["train_dataset_limit_real"] = (
                cl_configs["Data"]["memory_size"] // 2
                )

    for i, dataset in enumerate(cl_configs["Data"]["synthetic_dataset_names"]):
        ft_configs["Model"]["fine_tune"] = True
        ft_configs["Model"]["model_type"] = "expert"
        acc_matrix.append([])
        auc_matrix.append([])

        if i > 0:
            ft_configs = fill_configs_with_datasets(
                ft_configs, [dataset], cl_configs["Data"]["real_dataset_name"]
            )

            if (
                len(cl_configs["Model"]["ft_ckpt_paths"])
                == len(cl_configs["Data"]["synthetic_dataset_names"])
                and cl_configs["Model"]["load_from_ckpt"]
            ):
                last_expert_path = cl_configs["Model"]["ft_ckpt_paths"][i]
            else:
                print(f"Fine-tuning for dataset: {dataset}...")
                model_checkpoint_state_dict = train(ft_configs)
                last_expert_path = model_checkpoint_state_dict["last_model_path"]
                print(f"Finished fine-tuning for dataset {dataset}")
            print(f"Last expert path: {last_expert_path}")
            ft_configs["Model"]["src_ckpts"].append(last_expert_path)

            ft_configs = fill_configs_with_datasets(
                ft_configs, [dataset], cl_configs["Data"]["real_dataset_name"]
            )

            model_checkpoint_state_dict = train(ft_configs)
            ft_configs["Train"]["train_dataset_limit_per_class"] = None
            ft_configs["Train"]["train_dataset_limit_real"] = None
            ft_configs["Train"]["lr"] = cl_configs["Train"]["ft_lr"]
            ft_configs["Model"]["moe_ckpt"] = model_checkpoint_state_dict[
                "last_model_path"
            ]
            ft_configs["Model"]["transformer_ckpt"] = model_checkpoint_state_dict[
                "last_model_path"
            ]

    acc_matrix = np.array(acc_matrix)
    auc_matrix = np.array(auc_matrix)
    np.savetxt(f'acc_matrix.csv', np.round(acc_matrix, 4), delimiter=',')
    np.savetxt(f'auc_matrix.csv', np.round(auc_matrix, 4), delimiter=',')

    print("Expert file paths:")
    print(ft_configs["Model"]["src_ckpts"])
    filepath = f"expert_ckpt_{cl_configs['Model']['backbone']}.txt"
    with open(filepath, "w") as f:
        for path in ft_configs["Model"]["src_ckpts"]:
            f.write(f"{path}\n")
    print(f"Expert file paths saved to {filepath}")
