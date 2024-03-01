import lightning.pytorch as pl
import torch
from .classifier_head import ClassifierHead

from utils.model_utils import get_optimizer_dict, get_experts
from torchmetrics.classification import *


class FusionClassifier(pl.LightningModule):

    def __init__(
        self, model_configs: dict, train_configs: dict, data_configs: dict = None
    ):
        super().__init__()
        self.model_configs = model_configs
        self.train_configs = train_configs
        self.classifiers = get_experts(model_configs, trim=False)
        self.vacc = Accuracy("binary")
        self.acc = Accuracy("binary")
        self.conf_mat = ConfusionMatrix("binary")
        self.conf_mat_accumalator = torch.zeros(2, 2).int()
        self.auc = MulticlassAUROC(num_classes=2)
        self.f1 = F1Score("binary")
        self.prec = Precision("binary")
        self.recall = Recall("binary")
        self.save_hyperparameters()

    def forward(self, x):
        logits = [classifier.eval()(x) for classifier in self.classifiers]
        if self.model_configs["fusion_rule"] == "max":
            logits = torch.maximum(logits[0], logits[1])
        elif self.model_configs["fusion_rule"] == "avg":
            logits = (logits[0] + logits[1]) / 2
        else:
            raise ValueError(f"Unknown fusion rule: {self.model_configs['fusion_rule']}")
        return logits

    def infer(self, batch, is_valid=False, is_test=False):
        img, label = batch
        logits = self(img)
        self.acc.update(logits, label)
        if is_valid:
            self.vacc.update(logits, label)
        if is_test:
            if self.conf_mat_accumalator.device != self.device:
                self.conf_mat_accumalator = self.conf_mat_accumalator.to(self.device)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            label = label.argmax(dim=1)
            self.conf_mat_accumalator += self.conf_mat(
                preds, label
            ).int()
            self.auc.update(probs, label)
            self.prec.update(preds, label)
            self.recall.update(preds, label)
            self.f1.update(preds, label)

    def test_step(self, batch, batch_idx):
        self.infer(batch, is_test=True)
        self.log("acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("auc", self.auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("prec", self.prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", self.recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("f1", self.f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return get_optimizer_dict(self.train_configs, self.classifier_head.parameters())

    def get_conf_mat(self):
        return self.conf_mat_accumalator
