import torch


class ClassifierHead(torch.nn.Module):
    def __init__(self, n_features, configs, num_outputs=None):
        super().__init__()
        if num_outputs is None:
            num_outputs = len(configs["src_ckpts"]) + 1 + len(configs["manipulation_ckpts"])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_outputs),
        )

    def forward(self, x):
        return self.mlp(x)
