import torch
import random
from typing import Union, List
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms import RandomCrop, CenterCrop, Resize


class ParentDataset(Dataset):
    def __init__(
        self,
        quality: Union[List[int], int],
        patch_size: int,
        label: int,
        num_classes: int,
        center_crop: bool,
        save_format: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.file_paths: list = None
        self.quality = quality
        self.save_format = save_format
        if not self.save_format:
            self.label = torch.nn.functional.one_hot(
                torch.tensor(label), num_classes=num_classes
            ).float()
        else:
            self.label = label
        if isinstance(self.quality, int):
            self.quality = [self.quality]
        if center_crop:
            self.crop = CenterCrop(patch_size)
        else:
            self.crop = RandomCrop(patch_size)
        self.patch_size = patch_size

        self.resize = Resize(patch_size, antialias=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        try:
            img = io.read_image(self.file_paths[index])
        except:
            print(f"Error loading {self.file_paths[index]}")
            return self.__getitem__(random.randint(0, len(self.file_paths) - 1))
        if img.shape[1] < self.patch_size or img.shape[2] < self.patch_size:
            # print("Image too small:", self.file_paths[index], img.shape, "skipping...")
            # return self.__getitem__(random.randint(0, len(self.file_paths) - 1))
            print(f"Resized image {self.file_paths[index]} from {img.shape} to {self.patch_size}")
            img = self.resize(img)
        img = self.crop(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        quality_idx = index % len(self.quality)
        quality = self.quality[quality_idx]
        if quality != 100:
            img = io.decode_jpeg(io.encode_jpeg(img, quality=quality))
        img = img.float() / 255
        if self.save_format:
            return img, self.label, index
        else:
            return img, self.label
