import torch
import random
from torchvision import io
from .hd5dataset import HDF5Dataset


class HDF5JPEGDataset(HDF5Dataset):
    def __init__(
        self,
        quality,
        patch_size,
        label,
        num_classes,
        center_crop,
        hdf5_file_path,
        *args,
        **kwargs
    ):
        super().__init__(
            quality,
            patch_size,
            label,
            num_classes,
            center_crop,
            hdf5_file_path,
            *args,
            **kwargs
        )

    # override
    def __getitem__(self, idx):
        img = self.file_reader["patches"][idx]
        img = torch.from_numpy(img)
        img = self.crop(img)
        quality = random.randint(self.quality[0], self.quality[1])
        img = io.decode_jpeg(io.encode_jpeg(img, quality=quality))
        img = img.float() / 255
        return img, float(quality)
