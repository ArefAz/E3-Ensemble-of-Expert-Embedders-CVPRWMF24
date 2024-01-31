import os
import torch
from typing import Union
from lightning.pytorch.loggers import TensorBoardLogger
from models import ExpertClassifier, AnalyticsModel, JpegEstimator
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
torch.set_float32_matmul_precision("high")


class PrintCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        print()


def get_callbacks(configs):
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    if configs["Model"]["model_type"] == "jpeg":
        filename="{epoch:02d}-{step}-{v_loss:.4f}-{v_rmse:.4f}-{v_mae:.4f}"
    else:
        filename="{epoch:02d}-{step}-{v_loss:.4f}-{v_acc:.4f}"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor="v_loss",
        mode="min",
        filename=filename,
    )
    ModelCheckpoint.CHECKPOINT_NAME_LAST = filename + "-last"
    callbacks = [PrintCallback(), lr_monitor, checkpoint_callback]
    if configs["Train"]["early_stopping"]:
        early_stopping = EarlyStopping(monitor="v_loss", verbose=True)
        callbacks.append(early_stopping)
    return callbacks


def get_train_logger(configs):
    exp_name = get_exp_name(configs)
    logger = TensorBoardLogger(
        "logs", name=exp_name, version=configs["General"]["version"]
    )
    return logger


def get_model(configs, is_test: bool) -> Union[torch.nn.Module, str]:
    model_configs = configs["Model"]
    train_configs = configs["Train"]
    data_configs = configs["Data"]
    if model_configs["model_type"] == "expert":
        model_class = ExpertClassifier
    elif model_configs["model_type"] == "analytics":
        model_class = AnalyticsModel
    elif model_configs["model_type"] == "jpeg":
        model_class = JpegEstimator
    else:
        raise ValueError(f"Unknown model type {model_configs['model_type']}")

    if model_configs["model_type"] == "expert":
        ckpt_path = model_configs["expert_ckpt"]
    elif model_configs["model_type"] == "analytics":
        ckpt_path = model_configs["analytics_ckpt"]
    elif model_configs["model_type"] == "jpeg":
        ckpt_path = model_configs["jpeg_ckpt"]
    else:
        raise ValueError(f"Unknown model type {model_configs['model_type']}")
    if is_test:
        try:
            if model_configs["override_configs"]:
                model = model_class.load_from_checkpoint(
                    ckpt_path,
                    model_configs=model_configs,
                    train_configs=train_configs,
                    data_configs=data_configs,
                )
                print("Overriding configs...")
            else:
                model = model_class.load_from_checkpoint(ckpt_path)
        except Exception as e:
            raise ValueError(
                f"Error loading model from checkpoint {ckpt_path} possibly due to model and checkpoint mismatch."
            )
        print(f"Loaded model from checkpoint {ckpt_path}")
        version = ckpt_path.split("/")[2]
        return model, version
    else:
        if model_configs["fine_tune"] and model_configs["model_type"] == "expert":
            print("Fine tuning...")
            if model_configs["override_configs"]:
                model = model_class.load_from_checkpoint(
                    ckpt_path,
                    model_configs=model_configs,
                    train_configs=train_configs,
                    data_configs=data_configs,
                )
                print("Overriding configs...")
            else:
                model = model_class.load_from_checkpoint(ckpt_path)
            model.classifier.output = torch.nn.Linear(
                model.classifier.output.in_features, 2
            )
        else:
            model = model_class(model_configs, train_configs, data_configs)
            
        return model


def get_exp_name(configs):
    exp_name = ""
    if configs["Model"]["fine_tune"]:
        exp_name += "ft_"
    exp_name += f"{configs['Model']['model_type']}_"
    for dataset in configs["Data"]["datasets"]:
        exp_name += f"{dataset}_"

    if configs["Model"]["model_type"] == "expert":
        if not configs["is_test"]:
            if configs["Model"]["expert_task"] == "manipulation":
                exp_name += f"{configs['Model']['expert_manipulation']}_"
            elif configs["Model"]["expert_task"] == "src":
                exp_name += f"{configs['Model']['expert_task']}_"
        else:
            exp_name += (
                f"{configs['Model']['expert_ckpt'].split('/')[1].split('_')[2]}_"
            )
    elif configs["Model"]["model_type"] == "analytics":
        for manipulation in configs["Model"]["analytics_manipulations"]:
            exp_name += f"{manipulation}_"
    else: 
        pass

    exp_name += "q"
    if isinstance(quality := configs["Data"]["jpeg_quality"], int):
        exp_name += f"{quality}"
    elif isinstance(quality, list):
        for i in range(len(quality) - 1):
            exp_name += f"{quality[i]}-"
        exp_name += f"{quality[-1]}"

    if configs["is_test"]:
        if configs["Model"]["model_type"] == "expert":
            exp_name = configs["Model"]["expert_ckpt"].split("/")[1]
        elif configs["Model"]["model_type"] == "analytics":
            exp_name = configs["Model"]["analytics_ckpt"].split("/")[1]
    return exp_name
