import random
import lightning.pytorch as pl
import torch
from torchmetrics.classification import *

from data_pipes import Pipe
from .mislnet import MISLNet
from utils.model_utils import get_optimizer_dict
from torchvision.models import resnet50


class ExpertClassifier(pl.LightningModule):
    """
    Expert classifier, used for both source and manipulation detectors.
    if task == "src", then the classifier is a source detector
    if task == "manipulation", then the classifier is a manipulation detector. In this case, the label
    is randomly assigned to 0 or 1, and the image is correspondingly manipulated.
    """

    def __init__(
        self, model_configs: dict, train_configs: dict, data_configs: dict = None
    ):
        super().__init__()
        self.model_configs = model_configs
        self.train_configs = train_configs
        self.task = model_configs["expert_task"]
        # self.classifier = MISLNet(num_classes=2)
        self.classifier = resnet50(num_classes=2)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.acc = Accuracy("binary")
        self.v_acc = Accuracy("binary")
        self.f1 = F1Score(task="binary")
        self.prec = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.conf_mat = ConfusionMatrix("binary")
        self.conf_mat_accumalator = torch.zeros(2, 2).int()
        if self.task == "manipulation" or self.task == "src_test_with_manipulation":
            self.pipe = Pipe(
                manipulation=model_configs["expert_manipulation"],
                patch_size=model_configs["patch_size"],
            )
        self.save_hyperparameters()

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        loss = self.infer(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("t_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("t_acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.infer(batch, is_valid=True)
        self.log("v_acc", self.v_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("v_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.infer(batch, is_test=True)
        self.log("acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("f1", self.f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("prec", self.prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", self.recall, on_step=False, on_epoch=True, prog_bar=True)

    def infer(self, batch, is_valid=False, is_test=False):
        img, src_label = batch
        if self.task == "src":
            label = src_label
        elif self.task == "manipulation":
            label0 = torch.nn.functional.one_hot(torch.tensor(0), num_classes=2).float()
            label0 = label0.repeat(img.shape[0] // 2, 1)
            label1 = torch.nn.functional.one_hot(torch.tensor(1), num_classes=2).float()
            label1 = label1.repeat(img.shape[0] - img.shape[0] // 2, 1)
            label = torch.cat([label0, label1], dim=0)
            label = label.to(self.device)
            img[: img.shape[0] // 2] = self.pipe(img[: img.shape[0] // 2])
            indices = torch.randperm(img.shape[0])
            img = img[indices]
            label = label[indices]
        elif self.task == "src_test_with_manipulation":
            img[: img.shape[0] // 2] = self.pipe(img[: img.shape[0] // 2])
            label = src_label
            indices = torch.randperm(img.shape[0])
            img = img[indices]
            label = label[indices]
        else:
            raise ValueError(f"Unknown task {self.task}")
        logits = self(img)
        loss = self.loss(logits, label)
        label = label.argmax(dim=1)
        preds = logits.argmax(dim=1)
        self.acc.update(preds, label)
        if is_valid:
            self.v_acc.update(preds, label)
        if is_test:
            if self.conf_mat_accumalator.device != self.device:
                self.conf_mat_accumalator = self.conf_mat_accumalator.to(self.device)
                self.f1 = self.f1.to(self.device)
                self.prec = self.prec.to(self.device)
                self.recall = self.recall.to(self.device)
            self.conf_mat_accumalator += self.conf_mat(
                preds, label
            ).int()
            self.f1.update(preds, label)
            self.prec.update(preds, label)
            self.recall.update(preds, label)
        return loss

    def configure_optimizers(self):
        return get_optimizer_dict(self.train_configs, self.classifier.parameters())
    
    def get_conf_mat(self):
        return self.conf_mat_accumalator

