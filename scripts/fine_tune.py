from train import train
from test import run_test as test
from configs import ft_configs, continual_configs as cl_configs
from utils.continual_utils import fill_configs_with_datasets
import numpy as np


if __name__ == "__main__":
    ft_configs["Model"]["expert_ckpt"] = cl_configs["Model"]["ft_ckpt_paths"][0]
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

    for i, dataset in enumerate(cl_configs["Data"]["synthetic_dataset_names"]):
        ft_configs["Model"]["fine_tune"] = True
        ft_configs["Model"]["model_type"] = "expert"
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
                model_checkpoint_state_dict = train(ft_configs)
                last_expert_path = model_checkpoint_state_dict["last_model_path"]
                print(f"Finished fine-tuning for dataset {dataset}")
            print(f"Last expert path: {last_expert_path}")

        for dataset in cl_configs["Data"]["synthetic_dataset_names"]:
            # for dataset in seen_datasets:
            print(f"Testing Fine-tuned model on dataset: {dataset}...")
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
        np.savetxt(f'acc_matrix.csv', np.array(acc_matrix), delimiter=',')
        np.savetxt(f'auc_matrix.csv', np.array(auc_matrix), delimiter=',')
