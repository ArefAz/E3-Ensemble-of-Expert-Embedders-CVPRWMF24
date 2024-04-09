import lightning.pytorch as pl
import torch
import torch.nn.functional as F
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
        if train_configs["distill"]:
            train_configs["distill"] = False
            self.distill_loss = torch.nn.KLDivLoss(reduction="batchmean")
            self.teacher = MixtureOfExperts.load_from_checkpoint(
                model_configs["teacher_ckpt"], 
            )
            train_configs["distill"] = True
            self.teacher.freeze()
        self.feature_extractors = get_experts(model_configs)
        n_features = len(self.feature_extractors) * model_configs["expert_n_features"]
        self.classifier_head = ClassifierHead(n_features, model_configs, num_outputs=2)
        self.loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.tensor(train_configs["loss_weights"])
        )
        self.vacc = Accuracy("binary")
        self.acc = Accuracy("binary")
        self.test_acc = Accuracy("binary")
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
    
    def calc_distill_loss(self, x, student_logits, T=1.0):
        if not self.train_configs["distill"]:
            return 0.0
        with torch.no_grad():
            teacher_logits = self.teacher.eval()(x)
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        loss = (T ** 2) * self.distill_loss(student_log_probs, teacher_probs)
        return loss


    def infer(self, batch, is_valid=False, is_test=False):
        img, label = batch
        logits = self(img)
        dist_loss = self.calc_distill_loss(img, logits)
        loss = self.loss(logits, label) + dist_loss
        # if self.train_configs["distill"]:
        #     loss /= 2
        self.acc.update(logits.argmax(dim=1), label.argmax(dim=1))
        if is_valid:
            self.vacc.update(logits.argmax(dim=1), label.argmax(dim=1))
        if is_test:
            self.test_acc.update(logits.argmax(dim=1), label.argmax(dim=1))
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
        if self.train_configs["distill"]:
            return loss, dist_loss
        return loss

    def training_step(self, batch, batch_idx):
        if self.train_configs["distill"]:
            loss, dist_loss = self.infer(batch)
            self.log("dist_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.infer(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("t_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.train_configs["distill"]:
            loss, dist_loss = self.infer(batch, is_valid=True)
            self.log("v_dist_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            loss = self.infer(batch, is_valid=True)
        self.log("v_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("v_acc", self.vacc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _ = self.infer(batch, is_test=True)
        self.log("acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("auc", self.auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("prec", self.prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", self.recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("f1", self.f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return get_optimizer_dict(self.train_configs, self.classifier_head.parameters())

    def get_conf_mat(self):
        return self.conf_mat_accumalator
