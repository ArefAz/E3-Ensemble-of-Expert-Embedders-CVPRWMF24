import torch
import kornia

class Pipe(torch.nn.Module):

    def __init__(self, manipulation: str, patch_size: int) -> None:
        super().__init__()
        if manipulation == "upsample":
            self.manipulation = torch.nn.Sequential(
                kornia.augmentation.Resize(patch_size * 2),
                kornia.augmentation.RandomCrop((patch_size, patch_size)),
            )
        elif manipulation == "medianblur":
            self.manipulation = kornia.filters.MedianBlur((3, 3))
        else:
            raise NotImplementedError(f"Unknown manipulation {manipulation}")

    def forward(self, x):
        return self.manipulation(x)