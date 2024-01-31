import torch
import h5py
import random
from torchvision import io
from torchvision.transforms import RandomCrop, CenterCrop
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(
        self, quality, patch_size, label, num_classes, center_crop, hdf5_file_path, *args, **kwargs
    ):
        super().__init__()
        self.label = torch.nn.functional.one_hot(
            torch.tensor(label), num_classes=num_classes
        ).float()
        try:
            self.file_reader = h5py.File(hdf5_file_path, "r")
        except:
            print(f"Error loading {hdf5_file_path}")
            raise
        self.len = len(self.file_reader["patches"])
        self.quality = quality
        if isinstance(self.quality, int):
            self.quality = [self.quality]
        if center_crop:
            self.crop = CenterCrop(patch_size)
        else:
            self.crop = RandomCrop(patch_size)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.file_reader["patches"][idx]
        img = torch.from_numpy(img)
        img = self.crop(img)
        quality_idx = idx % len(self.quality)
        quality = self.quality[quality_idx]
        if quality != 100:
            img = io.decode_jpeg(io.encode_jpeg(img, quality=quality))
        img = img.float() / 255
        return img, self.label
