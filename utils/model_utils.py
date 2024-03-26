import torch
from typing import Tuple
from data_pipes import Pipe
from torchvision.models import ResNet, DenseNet



def get_pipes(data_configs: dict) -> torch.nn.ModuleList:
    module_list = torch.nn.ModuleList(
        [
            Pipe(
                manipulation=manipulation,
                patch_size=data_configs["patch_size"],
            )
            for manipulation in data_configs["analytics_manipulations"] if manipulation != "jpeg"
        ]
    )
    module_list.append(torch.nn.Identity())
    return module_list
    

def get_experts(model_configs: dict, trim=True) -> torch.nn.ModuleList:
    from models import ExpertClassifier, MISLNet
    from models.srnet import SRNet
    expert_detectors = torch.nn.ModuleList(
        [
            ExpertClassifier.load_from_checkpoint(
                checkpoint_path=model_configs["src_ckpts"][i]
            ).eval()
            for i in range(len(model_configs["src_ckpts"]))
        ]
    )
    if trim:
        for detector in expert_detectors:
            if isinstance(detector.classifier, ResNet):
                detector.classifier.fc = torch.nn.Identity()
                print(detector)
                exit()
            elif isinstance(detector.classifier, MISLNet) or isinstance(detector.classifier, SRNet):
                detector.classifier.output = torch.nn.Identity()
            elif isinstance(detector.classifier, DenseNet):
                detector.classifier.classifier = torch.nn.Identity()
            else:
                raise NotImplementedError(f"{type(detector.classifier)} not implemented")
            detector.freeze()
    else:
        for detector in expert_detectors:
            detector.freeze()
    return expert_detectors

def get_detectors(model_configs: dict) -> Tuple[torch.nn.ModuleList, torch.nn.ModuleList]:
    """
    for now we assume all detectors are of the same type (ExpertClassifier)
    """
    from models import ExpertClassifier, JpegEstimator, MISLNet
    src_detector_list = torch.nn.ModuleList(
        [
            ExpertClassifier.load_from_checkpoint(
                checkpoint_path=model_configs["src_ckpts"][i]
            ).eval()
            for i in range(len(model_configs["src_ckpts"]))
        ]
    )
    manipulation_detector_list = torch.nn.ModuleList([])
    for i in range(len(model_configs["manipulation_ckpts"])):
        ckpt = model_configs["manipulation_ckpts"][i]
        if "jpeg" in ckpt:
            manipulation_detector_list.append(
                JpegEstimator.load_from_checkpoint(checkpoint_path=ckpt).eval()
            )
        else:
            manipulation_detector_list.append(
                ExpertClassifier.load_from_checkpoint(checkpoint_path=ckpt).eval()
            )
    # for detector in src_detector_list:
    #     detector.freeze()
    #     if isinstance(detector.classifier, ResNet):
    #         detector.classifier.fc = torch.nn.Identity()
    for detector in src_detector_list + manipulation_detector_list:
        if isinstance(detector.classifier, ResNet):
            detector.classifier.fc = torch.nn.Identity()
        elif isinstance(detector.classifier, MISLNet):
            detector.classifier.output = torch.nn.Identity()
        detector.freeze()
    return src_detector_list, manipulation_detector_list


def get_optimizer_dict(configs, parameters) -> dict:
    if configs["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=configs["lr"])
    elif configs["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(parameters, lr=configs["lr"])
    elif configs["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            parameters, lr=configs["lr"], momentum=configs["momentum"]
        )
    else:
        raise NotImplementedError(f"{configs['optimizer']} is not implemented.")
    if configs["scheduler"] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=configs["lr_step_size"],
            gamma=configs["lr_decay_rate"],
        )
    elif configs["scheduler"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=configs["epochs"],
            eta_min=0,
        )
    else:
        raise NotImplementedError(f"{configs['scheduler']} is not implemented.")
    return {
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
