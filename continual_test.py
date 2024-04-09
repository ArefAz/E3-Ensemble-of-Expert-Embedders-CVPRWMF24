from train import train
from test import run_test as test
from configs import ft_configs, continual_configs as cl_configs
from utils.continual_utils import fill_configs_with_datasets
import numpy as np


if __name__ == "__main__":
    ft_configs["Model"]["expert_ckpt"] = cl_configs["Model"]["ft_ckpt_paths"][0]
    ft_configs["Model"]["model_type"] = "fusion"
    acc_matrix = []
    auc_matrix = []
    seen_datasets = [cl_configs["Data"]["synthetic_dataset_names"][0]]

    # for i, dataset in enumerate(cl_configs["Data"]["synthetic_dataset_names"]):
    #     ft_configs["Model"]["model_type"] = "fusion"
    acc_matrix.append([])
    auc_matrix.append([])

    for dataset in cl_configs["Data"]["synthetic_dataset_names"]:
        # for dataset in seen_datasets:
        print(f"Testing Fine-tuned model on dataset: {dataset}...")
        ft_configs = fill_configs_with_datasets(
            ft_configs, [dataset], cl_configs["Data"]["real_dataset_name"]
        )
        results = test(ft_configs)
        num = round(results["acc"], 4)
        acc_matrix[0].append(round(results["acc"], 4))
        auc_matrix[0].append(round(results["auc"], 4))

    print("ACC Matrix:")
    for j, row in enumerate(acc_matrix):
        print(f"{j}: {row}")
    print("AUC Matrix:")
    for j, row in enumerate(auc_matrix):
        print(f"{j}: {row}")
    print("ACC Averages:")
    for j, acc_row in enumerate(acc_matrix):
        if len(acc_row) == 0:
            continue
        avg = sum(acc_row[: j + 1]) / len(acc_row[: j + 1])
        print(f"{j}: {round(avg, 4)}")
    print("AUC Averages:")
    for j, auc_row in enumerate(auc_matrix):
        if len(auc_row) == 0:
            continue
        avg = sum(auc_row[: j + 1]) / len(auc_row[: j + 1])
        print(f"{j}: {round(avg, 4)}")

    acc_matrix = np.array(acc_matrix)
    auc_matrix = np.array(auc_matrix)
    np.savetxt(f'acc_matrix-{cl_configs["Model"]["backbone"]}.csv', np.round(acc_matrix, 4), delimiter=',')
    np.savetxt(f'auc_matrix-{cl_configs["Model"]["backbone"]}.csv', np.round(auc_matrix, 4), delimiter=',')
