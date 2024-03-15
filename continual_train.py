from train import train
from test import run_test as test
from configs import ft_configs, continual_configs as cl_configs
from utils.continual_utils import fill_configs_with_datasets

import pickle
import numpy as np


if __name__ == "__main__":
    ft_configs["Model"]["expert_ckpt"] = cl_configs["Model"]["ft_ckpt_paths"][0]
    ft_configs["Model"]["src_ckpts"].append(cl_configs["Model"]["ft_ckpt_paths"][0])
    ft_configs["Model"]["classifier"] = cl_configs["Model"]["backbone"]
    ft_configs["Model"]["expert_n_features"] = 2048 if cl_configs["Model"]["backbone"] == "resnet50" else 200
    ft_configs["Model"]["fine_tune"] = True
    ft_configs["Model"]["model_type"] = "expert"
    ft_configs["Train"]["epochs"] = cl_configs["Train"]["epochs"]
    ft_configs["Train"]["max_steps"] = cl_configs["Train"]["max_steps"]
    ft_configs["Train"]["lr"] = cl_configs["Train"]["ft_lr"]
    ft_configs["Train"]["batch_size"] = cl_configs["Train"]["batch_size"]
    ft_configs["Train"]["train_dataset_limit_per_class"] = cl_configs["Train"]["train_dataset_limit_per_class"]
    per_class = ft_configs["Train"]["train_dataset_limit_per_class"]
    ft_configs["Train"]["train_dataset_limit_real"] = cl_configs["Train"]["train_dataset_limit_real"]

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
    # assert len(cl_configs["Model"]["ft_ckpt_paths"]) == len(cl_configs["Data"]["synthetic_dataset_names"]), \
    # "Number of ckpt paths should be equal to the number of synthetic datasets"

    for i, dataset in enumerate(cl_configs["Data"]["synthetic_dataset_names"]):
        seen_count = (i + 1) * per_class
        ft_configs["Model"]["fine_tune"] = True
        ft_configs["Model"]["model_type"] = "expert"
        ft_configs["Train"]["distill"] = (
            cl_configs["Train"]["distill"] if i > 1 else False
        )
        acc_matrix.append([])
        auc_matrix.append([])
        
        if i > 0:
            seen_datasets.append(dataset)
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
                ft_configs["Train"]["lr"] = cl_configs["Train"]["ft_lr"]
                model_checkpoint_state_dict = train(ft_configs)
                last_expert_path = model_checkpoint_state_dict["last_model_path"]
                print(f"Finished fine-tuning for dataset {dataset}")
            print(f"Last expert path: {last_expert_path}")
            ft_configs["Model"]["model_type"] = cl_configs["Model"]["model_type"]
            ft_configs["Train"]["lr"] = cl_configs["Train"]["cls_lr"]
            ft_configs["Model"]["fine_tune"] = False
            ft_configs["Model"]["src_ckpts"].append(last_expert_path)

            ft_configs = fill_configs_with_datasets(
                ft_configs, seen_datasets, cl_configs["Data"]["real_dataset_name"]
            )

            if cl_configs["Data"]["fixed_memory"]:
                limit = min(cl_configs["Data"]["memory_size"] // 2, seen_count)
                ft_configs["Train"]["train_dataset_limit_per_class"] = limit // (i + 1)
                ft_configs["Train"]["train_dataset_limit_real"] = limit
                print(f"Memory size: {cl_configs['Data']['memory_size']}")
                print(
                    f"Train dataset limit per class: {ft_configs['Train']['train_dataset_limit_per_class']}"
                )
            else:
                loss_weights = [float(len(seen_datasets)), 1.0]
                loss_weights = [l / sum(loss_weights) for l in loss_weights]
                ft_configs["Train"]["loss_weights"] = loss_weights
            print(
                f"Training MOE for dataset: {dataset}... with loss weights: {ft_configs['Train']['loss_weights']}"
            )
            ft_configs["Train"]["lr"] = cl_configs["Train"]["cls_lr"]
            model_checkpoint_state_dict = train(ft_configs)
            print(f"Finished training MOE for dataset {dataset}")
            ft_configs["Train"]["train_dataset_limit_per_class"] = cl_configs["Train"]["train_dataset_limit_per_class"]
            ft_configs["Train"]["train_dataset_limit_real"] = cl_configs["Train"]["train_dataset_limit_real"]
            ft_configs["Train"]["lr"] = cl_configs["Train"]["lr"]
            ft_configs["Model"]["moe_ckpt"] = model_checkpoint_state_dict[
                "last_model_path"
            ]
            ft_configs["Model"]["transformer_ckpt"] = model_checkpoint_state_dict[
                "last_model_path"
            ]

        for dataset in cl_configs["Data"]["synthetic_dataset_names"]:
            print(f"Testing MOE for dataset: {dataset}...")
            ft_configs = fill_configs_with_datasets(
                ft_configs, [dataset], cl_configs["Data"]["real_dataset_name"]
            )
            results = test(ft_configs)
            num = round(results["acc"], 4)
            acc_matrix[i].append(round(results["acc"], 4))
            auc_matrix[i].append(round(results["auc"], 4))

        print("ACC Matrix:")
        for j, row in enumerate(acc_matrix):
            print(f"{j}: {row}")
        print("AUC Matrix:")
        for j, row in enumerate(auc_matrix):
            print(f"{j}: {row}")
        print("ACC Averages:")
        for j, acc_row in enumerate(acc_matrix):
            avg = sum(acc_row[: j + 1]) / len(acc_row[: j + 1])
            print(f"{j}: {round(avg, 4)}")
        print("AUC Averages:")
        for j, auc_row in enumerate(auc_matrix):
            avg = sum(auc_row[: j + 1]) / len(auc_row[: j + 1])
            print(f"{j}: {round(avg, 4)}")

    acc_matrix = np.array(acc_matrix)
    auc_matrix = np.array(auc_matrix)

    new_gen_data_size = ft_configs["Train"]["train_dataset_limit_real"]
    np.savetxt(f'acc_matrix_generator_datasize_{new_gen_data_size}.csv', np.round(acc_matrix, 4), delimiter=',')
    np.savetxt(f'auc_matrix.csv_generator_datasize_{new_gen_data_size}.csv', np.round(auc_matrix, 4), delimiter=',')
