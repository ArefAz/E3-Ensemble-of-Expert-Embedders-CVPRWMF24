from typing import *

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy


class ConvBlock(torch.nn.Module):
	def __init__(
		self,
		in_chans,
		out_chans,
		kernel_size,
		stride,
		padding,
		activation: Literal["tanh", "relu"],
	):
		super().__init__()
		assert activation.lower() in ["tanh", "relu"], "The activation layer must be either Tanh or ReLU"
		self.conv = torch.nn.Conv2d(
			in_chans,
			out_chans,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
		)
		self.bn = torch.nn.BatchNorm2d(out_chans)
		self.act = torch.nn.Tanh() if activation.lower() == "tanh" else torch.nn.ReLU()
		self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2)

	def forward(self, x):
		return self.maxpool(self.act(self.bn(self.conv(x))))


class DenseBlock(torch.nn.Module):
	def __init__(
		self,
		in_chans,
		out_chans,
		activation: Literal["tanh", "relu"],
	):
		super().__init__()
		assert activation.lower() in ["tanh", "relu"], "The activation layer must be either Tanh or ReLU"
		self.fc = torch.nn.Linear(in_chans, out_chans)
		self.act = torch.nn.Tanh() if activation.lower() == "tanh" else torch.nn.ReLU()

	def forward(self, x):
		return self.act(self.fc(x))


class MISLNet(torch.nn.Module):
	arch = {
		256: {
			"conv1": (-1, 96, 7, 2, "valid", "tanh"),
			"conv2": (96, 64, 5, 1, "same", "tanh"),
			"conv3": (64, 64, 5, 1, "same", "tanh"),
			"conv4": (64, 128, 1, 1, "same", "tanh"),
			"fc1": (6 * 6 * 128, 200, "relu"),
			"fc2": (200, 200, "relu"),
		},
		128: {
			"conv1": (-1, 96, 7, 2, "valid", "tanh"),
			"conv2": (96, 64, 5, 1, "same", "tanh"),
			"conv3": (64, 64, 5, 1, "same", "tanh"),
			"conv4": (64, 128, 1, 1, "same", "tanh"),
			"fc1": (2 * 2 * 128, 200, "tanh"),
			"fc2": (200, 200, "tanh"),
		},
		96: {
			"conv1": (-1, 96, 7, 2, "valid", "tanh"),
			"conv2": (96, 64, 5, 1, "same", "tanh"),
			"conv3": (64, 64, 5, 1, "same", "tanh"),
			"conv4": (64, 128, 1, 1, "same", "tanh"),
			"fc1": (8 * 4 * 64, 200, "tanh"),
			"fc2": (200, 200, "tanh"),
		},
		64: {
			"conv1": (-1, 96, 7, 2, "valid", "tanh"),
			"conv2": (96, 64, 5, 1, "same", "tanh"),
			"conv3": (64, 64, 5, 1, "same", "tanh"),
			"conv4": (64, 128, 1, 1, "same", "tanh"),
			"fc1": (2 * 4 * 64, 200, "tanh"),
			"fc2": (200, 200, "tanh"),
		},
	}

	def __init__(
		self,
		patch_size: int=256,
		num_classes=0,
		num_filters=6,
		constrained_conv=True,
		save_features=False,
		**args,
	):
		super().__init__()

		self.chosen_arch = self.arch[patch_size]
		self.num_classes = num_classes
		self.constrained_conv = constrained_conv
		self.num_filters = num_filters
		self.save_features = save_features
		self.features = None
		self.dense = None

		self.weights_cstr = torch.nn.Parameter(torch.randn(num_filters, 3, 5, 5))

		self.conv_blocks = torch.nn.Sequential(
			*[
				ConvBlock(
					in_chans=(
						num_filters
						if self.chosen_arch[f"conv{i}"][0] == -1
						else self.chosen_arch[f"conv{i}"][0]
					),
					out_chans=self.chosen_arch[f"conv{i}"][1],
					kernel_size=self.chosen_arch[f"conv{i}"][2],
					stride=self.chosen_arch[f"conv{i}"][3],
					padding=self.chosen_arch[f"conv{i}"][4],
					activation=self.chosen_arch[f"conv{i}"][5],
				)
				for i in [1, 2, 3, 4]
			]
		)

		self.fc_blocks = torch.nn.Sequential(
			*[
				DenseBlock(
					in_chans=self.chosen_arch[f"fc{i}"][0],
					out_chans=self.chosen_arch[f"fc{i}"][1],
					activation=self.chosen_arch[f"fc{i}"][2],
				)
				for i in [1, 2]
			]
		)

		if self.num_classes > 0:
			self.output = torch.nn.Linear(self.chosen_arch["fc2"][1], self.num_classes)

		self.init_weights()
		self.constrain_conv()

	def __str__(self) -> str:
		return "MISLNet" + f" {self.chosen_arch}"

	def init_weights(self):
		for layer in self.children():
			if isinstance(layer, torch.nn.Conv2d):
				torch.nn.init.xavier_uniform_(layer.weight)
				torch.nn.init.constant_(layer.bias, 0.0)

	def constrain_conv(self):
		w = self.weights_cstr
		w = w * 10000
		w[:, :, 2, 2] = 0
		w = w.reshape([self.num_filters, 3, 1, 25])
		w = w / w.sum(3, keepdims=True)
		w = w.reshape([self.num_filters, 3, 5, 5])
		w[:, :, 2, 2] = -1
		self.weights_cstr.data = w

	def get_dense(self):
		return self.dense.detach().clone()

	def forward(self, x):
		if self.training and self.constrained_conv:
			self.constrain_conv()

		constr_conv = F.conv2d(x, self.weights_cstr, padding="valid")
		constr_conv = F.pad(constr_conv, (2, 3, 2, 3))
		if self.save_features:
			features = constr_conv[:, 0, :, :]
			self.features = features


		conv_out = self.conv_blocks(constr_conv)
		conv_out = conv_out.permute(0, 2, 3, 1).flatten(1, -1)

		dense_out = self.fc_blocks(conv_out)
		self.dense = dense_out
		if self.num_classes > 0:
			dense_out = self.output(dense_out)
		return dense_out


default_training_config = {
	"lr": 1e-3,
	"momentum": 0.95,
	"decay_rate": 0.70,
	"decay_step": 3,
}


class MISLnetPLWrapper(LightningModule):
	def __init__(
		self,
		patch_size,
		num_classes,
		num_filters=6,
		training_config: Dict[str, Any] = default_training_config,
	):
		super().__init__()
		self.model = MISLNet(patch_size, num_classes, num_filters)

		self.lr = training_config["lr"]
		self.momentum = training_config["momentum"]
		self.decay_rate = training_config["decay_rate"]
		self.decay_step = training_config["decay_step"]

		self.train_acc = MulticlassAccuracy(num_classes=num_classes)
		self.val_acc = MulticlassAccuracy(num_classes=num_classes)
		self.test_acc = MulticlassAccuracy(num_classes=num_classes)

		self.example_input_array = torch.randn(1, 3, patch_size, patch_size)

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.cross_entropy(logits, y)
		self.train_acc(logits, y)
		self.log("train_loss", loss, prog_bar=True)
		self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.cross_entropy(logits, y)
		self.val_acc(logits, y)
		self.log("val_loss", loss, on_epoch=True)
		self.log("val_acc", self.val_acc, on_epoch=True)

	def test_step(self, batch, batch_idx):
		x, y = batch
		logits = self(x)
		loss = F.cross_entropy(logits, y)
		self.test_acc(logits, y)
		self.log("test_loss", loss, on_epoch=True, on_step=False)
		self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

	def on_validation_epoch_end(self) -> None:
		self.log("val_acc_epoch", self.val_acc.compute(), prog_bar=True, on_epoch=True, on_step=False)
		return super().on_validation_epoch_end()

	def configure_optimizers(self):
		optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
		steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.decay_rate)
		# scheduler = {
		#     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
		#         optimizer, mode="max", patience=1, factor=0.5
		#     ),
		#     "monitor": "val_acc_epoch",
		# }
		return [optimizer], [steplr]