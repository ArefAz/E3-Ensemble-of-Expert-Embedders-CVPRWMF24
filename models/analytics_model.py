import lightning.pytorch as pl
import torch
from .classifier_head import ClassifierHead

from utils.model_utils import get_optimizer_dict, get_detectors, get_pipes
from torchmetrics.classification import *
from torchvision.models import ResNet


def get_detector_dense(model, x):
    if not isinstance(model.classifier, ResNet):
        model.eval()(x)
        return model.classifier.get_dense()
    else:
        return model.classifier.eval()(x)



class AnalyticsModel(pl.LightningModule):
    def __init__(
        self, model_configs: dict, train_configs: dict, data_configs: dict = None
    ):
        super().__init__()
        self.model_configs = model_configs
        self.train_configs = train_configs
        self.src_detectors, self.manipulation_detectors = get_detectors(model_configs)
        self.num_src_neurons = len(self.src_detectors) + 1
        self.num_manipulation_neurons = len(self.manipulation_detectors) + 1
        n_features = (
            len(self.src_detectors) + len(self.manipulation_detectors)
        ) * model_configs["expert_n_features"]
        self.classifier_head = ClassifierHead(n_features, model_configs)
        self.src_loss = torch.nn.BCEWithLogitsLoss()
        self.manipulation_loss = torch.nn.BCEWithLogitsLoss()
        # if num_classes := len(model_configs["src_ckpts"]) < 2:
        #     self.src_acc = Accuracy("binary")
        #     self.vsrc_acc = Accuracy("binary")
        # else:
        #     self.src_acc = Accuracy("multiclass", num_classes=num_classes)
        self.vsrc_acc = Accuracy("binary")
        self.src_acc = Accuracy("binary")
        self.conf_mat = ConfusionMatrix("binary")
        self.conf_mat_accumalator = torch.zeros(2, 2).int()
        self.auc = MulticlassAUROC(num_classes=2)
        self.prec = Precision("binary")
        self.recall = Recall("binary")
        self.manip_acc_list = torch.nn.ModuleList(
            [
                Accuracy("binary")
                for _ in range(len(model_configs["manipulation_ckpts"]))
            ]
        )
        self.pipes = get_pipes(model_configs)
        self.use_jit = train_configs["use_jit"]
        self.save_hyperparameters()

    def forward(self, x):
        with torch.no_grad():
            if self.use_jit:
                futures = [
                    torch.jit.fork(get_detector_dense, detector, x)
                    for detector in [*self.src_detectors, *self.manipulation_detectors]
                ]
                results = [torch.jit.wait(future) for future in futures]
                src_features = results[: len(self.src_detectors)]
                manipulation_features = results[len(self.src_detectors) :]
            else:
                src_features = []
                manipulation_features = []
                for detector in self.src_detectors:
                    src_features.append(get_detector_dense(detector, x))
                for detector in self.manipulation_detectors:
                    manipulation_features.append(get_detector_dense(detector, x))

            src_features = torch.cat(src_features, dim=1)
            manipulation_features = torch.cat(manipulation_features, dim=1)

        return self.classifier_head(
            torch.cat([src_features, manipulation_features], dim=1)
        )

    def infer(self, batch, is_valid=False, is_test=False):
        img, src_label = batch
        manipulation_label = torch.zeros(
            img.shape[0], self.num_manipulation_neurons
        ).to(self.device)

        if len(self.pipes) > 1:
            for i, pipe in enumerate(self.pipes):
                img[i :: len(self.pipes)] = pipe(img[i :: len(self.pipes)])
                manipulation_label[i :: len(self.pipes)][:, i] = 1

            indices = torch.randperm(img.shape[0])
            img = img[indices]
            src_label = src_label[indices]
            manipulation_label = manipulation_label[indices]

        logits = self(img)
        src_logits, manipulation_logits = (
            logits[:, : self.num_src_neurons],
            logits[:, self.num_src_neurons :],
        )
        src_loss = (
            self.src_loss(src_logits, src_label) * self.train_configs["src_loss_coeff"]
        )
        manipulation_loss = 0.0
        self.src_acc.update(src_logits, src_label)
        if is_valid:
            self.vsrc_acc.update(src_logits, src_label)
        if len(self.pipes) > 1:
            for i in range(len(self.manipulation_detectors)):
                manipulation_loss += self.manipulation_loss(
                    manipulation_logits[:, i], manipulation_label[:, i]
                )
            manipulation_loss /= len(self.manipulation_detectors)
            manipulation_loss *= self.train_configs["manipulation_loss_coeff"]
            manipulation_accs = torch.tensor(
                [
                    acc(manipulation_logits[:, i], manipulation_label[:, i])
                    for i, acc in enumerate(self.manip_acc_list)
                ]
            )
        else:
            manipulation_accs = torch.tensor([0.0]).float()

        if is_test:
            if self.conf_mat_accumalator.device != self.device:
                self.conf_mat_accumalator = self.conf_mat_accumalator.to(self.device)
                self.auc = self.auc.to(self.device)
                self.prec = self.prec.to(self.device)
                self.recall = self.recall.to(self.device)

            preds = src_logits.argmax(dim=1)
            probs = torch.nn.functional.softmax(src_logits, dim=1)
            src_label = src_label.argmax(dim=1)
            self.conf_mat_accumalator += self.conf_mat(
                preds, src_label
            ).int()
            self.auc.update(probs, src_label)
            self.prec.update(preds, src_label)
            self.recall.update(preds, src_label)
            return src_loss, manipulation_loss, manipulation_accs
        return src_loss, manipulation_loss, manipulation_accs

    def training_step(self, batch, batch_idx):
        src_loss, manipulation_loss, manipulation_accs = self.infer(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("tl_src", src_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "tl_manip", manipulation_loss, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "t_loss",
            src_loss + manipulation_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("tacc_src", self.src_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "tacc_manip_avg",
            manipulation_accs.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return src_loss + manipulation_loss

    def validation_step(self, batch, batch_idx):
        src_loss, manipulation_loss, manipulation_accs = self.infer(batch, is_valid=True)
        self.log("v_loss", src_loss + manipulation_loss, on_epoch=True, prog_bar=True)
        self.log("v_acc", self.vsrc_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "vacc_manip_avg",
            manipulation_accs.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "vaccs_manip",
            manipulation_accs.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        src_loss, manipulation_loss, manipulation_accs = self.infer(
            batch, is_test=True
        )
        self.log("acc", self.src_acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(
        #     "acc_manip_avg",
        #     manipulation_accs.mean(),
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "accs_manip",
        #     manipulation_accs,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        # )
        # self.log("loss_src", src_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log(
        #     "loss_manip",
        #     manipulation_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        # self.log(
        #     "loss",
        #     src_loss + manipulation_loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )
        self.log("prec", self.prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("recall", self.recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("auc", self.auc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return get_optimizer_dict(self.train_configs, self.classifier_head.parameters())

    def get_conf_mat(self):
        return self.conf_mat_accumalator

