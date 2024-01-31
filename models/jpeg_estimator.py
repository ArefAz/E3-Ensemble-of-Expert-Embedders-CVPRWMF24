import lightning.pytorch as pl
import torch

from torchmetrics.regression import *
from utils.model_utils import get_optimizer_dict
from .mislnet import MISLNet


class JpegEstimator(pl.LightningModule):
    def __init__(self, model_configs: dict, train_configs: dict, data_configs: dict):
        super().__init__()
        self.model_configs = model_configs
        self.train_configs = train_configs
        self.classifier = MISLNet(num_classes=1)
        self.loss = torch.nn.MSELoss()
        self.tmse = MeanSquaredError()
        self.vmse = MeanSquaredError()
        self.mse = MeanSquaredError()
        self.tmae = MeanAbsoluteError()
        self.vmae = MeanAbsoluteError()
        self.mae = MeanAbsoluteError()
        self.tsmape = SymmetricMeanAbsolutePercentageError()
        self.vsmape = SymmetricMeanAbsolutePercentageError()
        self.smape = SymmetricMeanAbsolutePercentageError()
        self.save_hyperparameters()

    def forward(self, x):
        return self.classifier(x)

    def infer(self, batch, is_valid=False, is_test=False):
        img, quality = batch
        estimation = self(img)
        quality = quality.unsqueeze(1).float()
        loss = self.loss(estimation, quality)
        if not is_valid and not is_test:
            self.tmae.update(estimation, quality)
            self.tmse.update(estimation, quality)
            self.tsmape.update(estimation, quality)
        elif is_valid:
            self.vmae.update(estimation, quality)
            self.vmse.update(estimation, quality)
            self.vsmape.update(estimation, quality)
        elif is_test:
            self.mae.update(estimation, quality)
            self.mse.update(estimation, quality)
            self.smape.update(estimation, quality)
        else:
            raise ValueError("is_valid and is_test cannot be both True")
        return torch.sqrt(loss)

    def training_step(self, batch, batch_idx):
        loss = self.infer(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("mse", self.tmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mae", self.tmae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("smape", self.tsmape, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.infer(batch, is_valid=True)
        self.log("v_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("v_mse", self.vmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("v_mae", self.vmae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("v_smape", self.vsmape, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.infer(batch, is_test=True)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mse", self.mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mae", self.mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("smape", self.smape, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return get_optimizer_dict(self.train_configs, self.classifier.parameters())
