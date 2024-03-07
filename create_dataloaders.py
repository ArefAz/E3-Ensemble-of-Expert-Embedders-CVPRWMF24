import pickle
from train import train
from configs import configs
from utils.continual_utils import fill_configs_with_datasets
from utils.data_utils import get_dataloaders

if __name__ == "__main__":
    real_dataset_name = "dn-real-2k"
    
    dataset_name_list = [
        "dn-gan-2k",
        "dn-glide-2k",
        "dn-mj-2k",
        "dn-gansformer-2k",
        "dn-dallemini-2k",
        "dn-sd14-2k",
        "dn-tt-2k",
        "dn-sd21-2k",
        "dn-vqdiff-2k",
        "dn-ddf-2k",
    ]
    

    for dataset_name in dataset_name_list:
        configs = fill_configs_with_datasets(configs, [dataset_name], real_name=real_dataset_name)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(configs["Model"], configs["Data"], configs["Train"], save_format=True)
        file_name = f"dataloaders/{real_dataset_name}+{dataset_name}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump([train_dataloader, val_dataloader, test_dataloader], f)
        print(f"Saved dataloaders to {file_name}")
        print()
