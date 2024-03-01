import lightning.pytorch as pl
import torch
from .classifier_head import ClassifierHead

from utils.model_utils import get_optimizer_dict, get_experts
from torchmetrics.classification import *


class MixtureOfExperts(pl.LightningModule):

    def __init__(
        self, model_configs: dict, train_configs: dict, data_configs: dict = None
    ):
        super().__init__()
        self.model_configs = model_configs
        self.train_configs = train_configs
        self.feature_extractors = get_experts(model_configs)
        n_features = len(self.feature_extractors) * model_configs["expert_n_features"]
        self.classifier_head = ClassifierHead(n_features, model_configs, num_outputs=2)
        self.loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.tensor(train_configs["loss_weights"])
        )
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
        with torch.no_grad():
            features = torch.cat(
                [extractor.eval()(x) for extractor in self.feature_extractors], dim=1
            )
        return self.classifier_head(features)

    def infer(self, batch, is_valid=False, is_test=False):
        img, label = batch
        logits = self(img)
        loss = self.loss(logits, label)
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
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.infer(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("t_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.infer(batch, is_valid=True)
        self.log("v_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("v_acc", self.vacc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.infer(batch, is_test=True)
        self.log("acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("auc", self.auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("prec", self.prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", self.recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("f1", self.f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return get_optimizer_dict(self.train_configs, self.classifier_head.parameters())

    def get_conf_mat(self):
        return self.conf_mat_accumalator
