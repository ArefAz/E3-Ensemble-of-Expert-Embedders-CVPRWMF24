from datasets import *
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset


def get_dataloaders(model_config: dict, data_config: dict, train_config: dict):
    train_dataset, val_dataset, test_dataset = get_datasets(
        model_config, data_config, train_config
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


def get_datasets(model_config: dict, data_config: dict, train_config: dict):
    assert (
        len(data_config["datasets"])
        == len(data_config["train_txt_paths"])
        == len(data_config["val_txt_paths"])
        == len(data_config["test_txt_paths"])
    ), "Number of datasets, train_txt_paths, val_txt_paths, test_txt_paths should be the same and in the same order"
    train_datasets = []
    val_datasets = []
    test_datasets = []

    num_generators = len(data_config["datasets"]) - 1 # -1 for real dataset

    for i, dataset_name in enumerate(data_config["datasets"]):
        assert (
            dataset_name in data_config["train_txt_paths"][i]
        ), "Dataset name mismatch"
        assert dataset_name in data_config["val_txt_paths"][i], "Dataset name mismatch"
        assert dataset_name in data_config["test_txt_paths"][i], "Dataset name mismatch"
        if dataset_name in ["coco", "midb", "easy-real", "db-real", "dn-real"]:
            label = 0
        elif dataset_name in [
            "dm",
            "gan",
            "stylegan",
            "stylegan2",
            "stylegan3",
            "projected_gan",
            'progan',
            "easy-progan",
            "tam_trans",
            "stable_diffusion",
            "db-gan",
            'db-sd',
            "du-gan",
            "dn-sd",
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
        )
        val_dataset = dataset_class(
            quality=data_config["jpeg_quality"],
            patch_size=data_config["patch_size"],
            label=label,
            num_classes=data_config["num_src_classes"],
            center_crop=True,
            txt_file_path=data_config["val_txt_paths"][i],
            hdf5_file_path=data_config["val_hdf5_paths"][i],
        )
        test_dataset = dataset_class(
            quality=data_config["jpeg_quality"],
            patch_size=data_config["patch_size"],
            label=label,
            num_classes=data_config["num_src_classes"],
            center_crop=True,
            txt_file_path=data_config["test_txt_paths"][i],
            hdf5_file_path=data_config["test_hdf5_paths"][i],
        )
        # if label == 0:
        #     if train_config["train_dataset_limit_per_class"]:
        #         train_dataset = Subset(train_dataset, range(0, train_config["train_dataset_limit_per_class"]))
        #     if train_config["val_dataset_limit_per_class"]:
        #         val_dataset = Subset(val_dataset, range(0, train_config["val_dataset_limit_per_class"]))
        #     if train_config["test_dataset_limit_per_class"]:
        #         test_dataset = Subset(test_dataset, range(0, train_config["test_dataset_limit_per_class"]))
        # else:
        #     if train_config["train_dataset_limit_per_class"]:
        #         train_dataset, _ = random_split(
        #             train_dataset,
        #             [
        #                 train_config["train_dataset_limit_per_class"] // num_generators,
        #                 len(train_dataset) - train_config["train_dataset_limit_per_class"] // num_generators,
        #             ],
        #         )
        #     if train_config["val_dataset_limit_per_class"]:
        #         val_dataset, _ = random_split(
        #             val_dataset,
        #             [
        #                 train_config["val_dataset_limit_per_class"] // num_generators,
        #                 len(val_dataset) - train_config["val_dataset_limit_per_class"] // num_generators,
        #             ],
        #         )
        #     if train_config["test_dataset_limit_per_class"]:
        #         test_dataset, _ = random_split(
        #             test_dataset,
        #             [
        #                 train_config["test_dataset_limit_per_class"] // num_generators,
        #                 len(test_dataset) - train_config["test_dataset_limit_per_class"] // num_generators,
        #             ],
        #         )

        print("len(train_dataset):", len(train_dataset))
        print("len(val_dataset):", len(val_dataset))
        print("len(test_dataset):", len(test_dataset))
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)
    if train_config["train_dataset_hard_limit_num"]:
        train_dataset, _ = random_split(
            train_dataset,
            [
                train_config["train_dataset_hard_limit_num"],
                len(train_dataset) - train_config["train_dataset_hard_limit_num"],
            ],
        )

    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    return train_dataset, val_dataset, test_dataset
