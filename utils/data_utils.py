from datasets import *
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch


def get_dataloaders(
    model_config: dict, data_config: dict, train_config: dict, save_format=False
):
    train_dataset, val_dataset, test_dataset = get_datasets(
        model_config, data_config, train_config, save_format=save_format
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        prefetch_factor=data_config["prefetch_factor"],
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        prefetch_factor=data_config["prefetch_factor"],
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        prefetch_factor=data_config["prefetch_factor"],
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader, test_dataloader


def get_datasets(
    model_config: dict, data_config: dict, train_config: dict, save_format=False
):
    assert (
        len(data_config["datasets"])
        == len(data_config["train_txt_paths"])
        == len(data_config["val_txt_paths"])
        == len(data_config["test_txt_paths"])
    ), "Number of datasets, train_txt_paths, val_txt_paths, test_txt_paths should be the same and in the same order"
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for i, dataset_name in enumerate(data_config["datasets"]):
        assert (
            dataset_name in data_config["train_txt_paths"][i]
        ), f"Dataset name mismatch: {dataset_name} not in {data_config['train_txt_paths'][i]}"
        assert dataset_name in data_config["val_txt_paths"][i], "Dataset name mismatch"
        assert dataset_name in data_config["test_txt_paths"][i], "Dataset name mismatch"
        if (
            dataset_name
            in [
                "coco",
                "midb",
                "easy-real",
                "db-real",
                "dn-real",
                "dn-real-500",
                "db-real-coco-lsun",
                "dn-real-coco-lsun-2k",
            ]
            or "real" in dataset_name
        ):
            label = 0
        elif dataset_name in [
            "dm",
            "gan",
            "db-final-gan",
            "stylegan",
            "stylegan2",
            "stylegan3",
            "stargan",
            "biggan",
            "projected_gan",
            "progan",
            "easy-progan",
            "tam_trans",
            "stable_diffusion",
            "db-gan",
            "db-sd",
            "du-gan",
            "dn-gan-500",
            "dn-gan-250",
            "dn-gan-2k",
            "dn-sd",
            "dn-sd-500",
            "dn-sd-250",
            "dn-tt",
            "dn-tt-2k",
            "dn-tt-250",
            "dn-gansformer-2k",
            "dn-dallemini-2k",
            "dn-vqdiff-2k",
            "dn-ddf-2k",
            "dn-eg3d",
            "dn-eg3d-250",
            "dn-glide",
            "dn-glide-2k",
            "dn-glide-250",
            "dn-dalle2",
            "dn-dalle2-250",
            "dn-glide-dif",
            "dn-mj",
            "dn-mj-2k",
            "dn-sd14",
            "dn-sd14-2k",
            "dn-sd21",
            "dn-sd21-2k",
            "dn-dalle-mini",
            "dn-final-gan-2k",
            "dn-cips-2k",
        ]:
            label = 1
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")

        if model_config["model_type"] == "jpeg":
            if dataset_name == "midb":
                dataset_class = HDF5JPEGDataset
            else:
                dataset_class = JpegDatasetFromTxt
        else:
            if dataset_name == "midb":
                dataset_class = HDF5Dataset
            else:
                dataset_class = ImageDatasetFromTxt

        print(f"Loading dataset {dataset_name}...")
        train_dataset = dataset_class(
            quality=data_config["jpeg_quality"],
            patch_size=data_config["patch_size"],
            label=label,
            num_classes=data_config["num_src_classes"],
            center_crop=False,
            txt_file_path=data_config["train_txt_paths"][i],
            hdf5_file_path=data_config["train_hdf5_paths"][i],
            save_format=save_format,
        )
        val_dataset = dataset_class(
            quality=data_config["jpeg_quality"],
            patch_size=data_config["patch_size"],
            label=label,
            num_classes=data_config["num_src_classes"],
            center_crop=True,
            txt_file_path=data_config["val_txt_paths"][i],
            hdf5_file_path=data_config["val_hdf5_paths"][i],
            save_format=save_format,
        )
        test_dataset = dataset_class(
            quality=data_config["jpeg_quality"],
            patch_size=data_config["patch_size"],
            label=label,
            num_classes=data_config["num_src_classes"],
            center_crop=True,
            txt_file_path=data_config["test_txt_paths"][i],
            hdf5_file_path=data_config["test_hdf5_paths"][i],
            save_format=save_format,
        )
        if train_config["train_dataset_limit_per_class"] and "real" not in dataset_name:
            train_dataset = Subset(
                train_dataset,
                torch.arange(train_config["train_dataset_limit_per_class"]),
            )
        if train_config["val_dataset_limit_per_class"] and "real" not in dataset_name:
            val_dataset = Subset(
                val_dataset, torch.arange(train_config["val_dataset_limit_per_class"])
            )
        if train_config["test_dataset_limit_per_class"] and "real" not in dataset_name:
            test_dataset = Subset(
                test_dataset, torch.arange(train_config["test_dataset_limit_per_class"])
            )
        if train_config["train_dataset_limit_real"] and "real" in dataset_name:
            train_dataset = Subset(
                train_dataset, torch.arange(train_config["train_dataset_limit_real"])
            )
        if train_config["val_dataset_limit_real"] and "real" in dataset_name:
            val_dataset = Subset(
                val_dataset, torch.arange(train_config["val_dataset_limit_real"])
            )
        if train_config["test_dataset_limit_real"] and "real" in dataset_name:
            test_dataset = Subset(
                test_dataset, torch.arange(train_config["test_dataset_limit_real"])
            )

        print("len(train_dataset):", len(train_dataset), end=", ")
        print("len(val_dataset):", len(val_dataset), end=", ")
        print("len(test_dataset):", len(test_dataset))
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)
    if train_config["train_dataset_hard_limit_num"]:
        indices = torch.randperm(len(train_dataset))[
            : train_config["train_dataset_hard_limit_num"]
        ]
        train_dataset = Subset(train_dataset, indices)
    if train_config["val_dataset_hard_limit_num"]:
        indices = torch.randperm(len(val_dataset))[
            : train_config["val_dataset_hard_limit_num"]
        ]
        val_dataset = Subset(val_dataset, indices)
    if train_config["test_dataset_hard_limit_num"]:
        indices = torch.randperm(len(test_dataset))[
            : train_config["test_dataset_hard_limit_num"]
        ]
        test_dataset = Subset(test_dataset, indices)

    print("Train dataset size:", len(train_dataset), end=", ")
    print("Val dataset size:", len(val_dataset), end=", ")
    print("Test dataset size:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset
