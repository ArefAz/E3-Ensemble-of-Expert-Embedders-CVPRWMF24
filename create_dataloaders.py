import pickle
from train import train
from configs import configs
from utils.continual_utils import fill_configs_with_datasets
from utils.data_utils import get_dataloaders

if __name__ == "__main__":
    real_dataset_name = "dn-real"
    
    synthetic_dataset_names = [
        "dn-gan",
        "dn-sd14",
        "dn-glide",
        "dn-mj",
        "dn-dallemini",
        "dn-tt",
        "dn-sd21",
        "dn-cips",
        "dn-biggan",
        "dn-vqdiff",
        "dn-diffgan",
        "dn-sg3",
        "dn-gansformer",
        "dn-dalle2",
        "dn-ld",
        "dn-eg3d",
        "dn-projgan",
        "dn-sd1",
        "dn-ddg",
        "dn-ddpm",
    ]
    

    for dataset_name in synthetic_dataset_names:
        configs = fill_configs_with_datasets(configs, [dataset_name], real_name=real_dataset_name)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(configs["Model"], configs["Data"], configs["Train"], save_format=True)
        file_name = f"dataloaders/{real_dataset_name}+{dataset_name}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump([train_dataloader, val_dataloader, test_dataloader], f)
        print(f"Saved dataloaders to {file_name}")
        print()
