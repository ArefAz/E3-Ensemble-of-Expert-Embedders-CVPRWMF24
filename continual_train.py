from train import train
from test import run_test as test
from configs import configs_ft, continual_configs as configs
from utils.continual_utils import fill_configs_with_datasets

if __name__ == "__main__":
    configs_ft["Model"]["expert_ckpt"] = configs["Model"]["baseline_ckpt"]
    configs_ft["Model"]["src_ckpts"].append(configs["Model"]["baseline_ckpt"])
    fine_tuned_model_paths = []
    acc_matrix = []
    auc_matrix = []
    seen_datasets = []

    for i, dataset in enumerate(configs["Data"]["synthetic_dataset_names"]):
        task_id = i + 1 # the first task is the baseline therefore we skip it
        acc_matrix.append([])
        auc_matrix.append([])

        seen_datasets.append(dataset)
        configs_ft["Model"]["fine_tune"] = True
        configs_ft["Model"]["model_type"] = "expert"
        configs_ft = fill_configs_with_datasets(configs_ft, [dataset], configs["Data"]["real_dataset_name"])

        print(f"Fine-tuning for dataset: {dataset}...")
        model_checkpoint_state_dict = train(configs_ft)
        print(f"Finished fine-tuning for dataset {dataset}")

        last_model_path = model_checkpoint_state_dict["last_model_path"]
        fine_tuned_model_paths.append(last_model_path)
        configs_ft["Model"]["model_type"] = "moe"
        configs_ft["Model"]["fine_tune"] = False
        configs_ft["Model"]["src_ckpts"] = fine_tuned_model_paths

        # add gan dataset to the list of datasets for training and testing the MOE
        gan_included_datasets = ["dn-gan-500"] + seen_datasets
        configs_ft = fill_configs_with_datasets(configs_ft, gan_included_datasets , configs["Data"]["real_dataset_name"])
        
        if configs["Data"]["fixed_memory"]:
            configs_ft["Data"]["train_dataset_limit_per_class"] = configs["Data"]["memory_size"] // 2 // (task_id + 1)
            print(f"Memory size: {configs['Data']['memory_size']}")
            print(f"Train dataset limit per class: {configs_ft['Data']['train_dataset_limit_per_class']}")
            exit()
        else:
            loss_weights = [float(len(gan_included_datasets)), 1.0]
            loss_weights = [l / sum(loss_weights) for l in loss_weights]
            configs_ft["Train"]["loss_weights"] = loss_weights
        print(f"Training MOE for dataset: {dataset}... with loss weights: {configs_ft['Train']['loss_weights']}")
        model_checkpoint_state_dict = train(configs_ft)
        configs_ft["Data"]["train_dataset_limit_per_class"] = None
        print(f"Finished training MOE for dataset {dataset}")
        configs_ft["Model"]["moe_ckpt"] = model_checkpoint_state_dict["last_model_path"]
        configs_ft["Data"]["test_dataset_limit_per_class"] = configs["Data"]["memory_size"] // (task_id + 1)


        for j, dataset in enumerate(gan_included_datasets):
            print(f"Testing MOE for dataset: {dataset}...")
            configs_ft = fill_configs_with_datasets(configs_ft, [dataset], configs["Data"]["real_dataset_name"])
            results = test(configs_ft)
            print(f"Finished testing MOE for dataset {dataset}")
            print(f"Results for dataset {dataset}: {results}")
            acc_matrix[i].append(results["acc"])
            auc_matrix[i].append(results["auc"])

        print(f"Accuracy matrix: {acc_matrix}")
        print(f"AUC matrix: {auc_matrix}")
        for j, (acc_row, auc_row) in enumerate(zip(acc_matrix, auc_matrix)):
            print(f"Accuracy row avg for task {j + 1}: {sum(acc_row) / len(acc_row)}") 
            print(f"AUC row avg for task {j + 1}: {sum(auc_row) / len(auc_row)}")



        