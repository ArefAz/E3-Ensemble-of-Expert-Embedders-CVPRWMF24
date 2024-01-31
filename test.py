import lightning.pytorch as pl
import os
import pickle

from utils.train_utils import get_exp_name, get_model
from utils.data_utils import get_dataloaders
from lightning.pytorch.loggers import TensorBoardLogger
from eval import pretty_print_mat


def run_test(configs):
    configs["is_test"] = True
    pl.seed_everything(configs["General"]["seed"])
    exp_name = get_exp_name(configs)

    print(f"Running the tests for experiment: {exp_name}")
    model, model_version = get_model(configs, is_test=True)
    print("with model version:", model_version)
    print("with quality_list:", configs["Data"]["test_jpeg_qualities"])

    results = {}
    results_concatenated = {}
    results_jpeg = {}

    for quality in configs["Data"]["test_jpeg_qualities"]:
        # if configs["Model"]["model_type"] in ["analytics", "expert"]:
        #     model.reset()
        model, model_version = get_model(configs, is_test=True)
        configs["Data"]["jpeg_quality"] = quality
        logger = TensorBoardLogger("logs", name=exp_name, version=model_version)
        trainer = pl.Trainer(
            logger=logger,
            accelerator="gpu",
            devices=configs["General"]["num_devices"],
            inference_mode=configs["General"]["inference_mode"],
            limit_test_batches=configs["Train"]["test_dataset_limit"],
        )
        print(f"Quality: {configs['Data']['jpeg_quality']}")
        _, _, test_dataloader = get_dataloaders(
            configs["Model"], configs["Data"], configs["Train"]
        )
        results[quality] = trainer.test(
            model=model,
            dataloaders=test_dataloader,
        )
        if configs["Model"]["model_type"] in ["analytics", "expert"]:
            print()
            pretty_print_mat(
                model.get_conf_mat(),
                column_labels=["Real", "Synth"],
                row_labels=["Real", "Synth"],
            )
            print()

    save_filename = (
        os.path.join(*logger.log_dir.split("/")[:-1], model_version) + "/results.pkl"
    )
    with open(save_filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {save_filename}")


if __name__ == "__main__":
    from configs import configs

    if (
        isinstance(configs["Model"]["expert_manipulation"], list)
        and configs["Model"]["expert_task"]
        in ["manipulation", "src_test_with_manipulation"]
        and configs["Model"]["model_type"] == "expert"
    ):
        manipulations = configs["Model"]["expert_manipulation"]
        for manipulation in manipulations:
            configs["Model"]["expert_manipulation"] = manipulation
            print(f"Running experiment for manipulation: {manipulation}")
            run_test(configs)
    else:
        run_test(configs)
