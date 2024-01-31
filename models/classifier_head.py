import torch


class ClassifierHead(torch.nn.Module):
    def __init__(self, n_features, configs):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(
                64,
                len(configs["src_ckpts"]) + 1 + len(configs["manipulation_ckpts"]),
            ),
        )

    def forward(self, x):
        return self.mlp(x)
