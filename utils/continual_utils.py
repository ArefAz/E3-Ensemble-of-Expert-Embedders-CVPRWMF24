def dataset_name_to_paths(dataset_name):
    train_txt_path = f"datasets/dataset_file_paths/{dataset_name}/train.txt"
    val_txt_path = f"datasets/dataset_file_paths/{dataset_name}/val.txt"
    test_txt_path = f"datasets/dataset_file_paths/{dataset_name}/val.txt"
    return train_txt_path, val_txt_path, test_txt_path

def fill_configs_with_datasets(configs, dataset_names, real_name):
    train_txt_paths = [dataset_name_to_paths(real_name)[0]]
    val_txt_paths = [dataset_name_to_paths(real_name)[1]]
    test_txt_paths = [dataset_name_to_paths(real_name)[2]]
    configs_datasets = [real_name]
    
    for dataset_name in dataset_names:
        configs_datasets.append(dataset_name)
        train_txt_path, val_txt_path, test_txt_path = dataset_name_to_paths(dataset_name)
        train_txt_paths.append(train_txt_path)
        val_txt_paths.append(val_txt_path)
        test_txt_paths.append(test_txt_path)
    configs["Data"]["train_txt_paths"] = train_txt_paths
    configs["Data"]["val_txt_paths"] = val_txt_paths
    configs["Data"]["test_txt_paths"] = test_txt_paths
    configs["Data"]["datasets"] = configs_datasets
    return configs
