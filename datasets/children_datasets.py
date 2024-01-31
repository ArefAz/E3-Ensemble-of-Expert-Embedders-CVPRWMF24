import random
import pandas as pd
from .parent_dataset import ParentDataset
from torchvision import io


class ImageDatasetFromTxt(ParentDataset):
    def __init__(
        self,
        quality,
        patch_size,
        label,
        num_classes,
        center_crop,
        txt_file_path,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            quality, patch_size, label, num_classes, center_crop, *args, **kwargs
        )
        with open(txt_file_path, "r") as f:
            self.file_paths = [line.strip() for line in f]
        random.shuffle(self.file_paths)


class JpegDatasetFromTxt(ParentDataset):
    def __init__(
        self,
        quality,
        patch_size,
        label,
        num_classes,
        center_crop,
        txt_file_path,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            quality, patch_size, label, num_classes, center_crop, *args, **kwargs
        )
        with open(txt_file_path, "r") as f:
            self.file_paths = [line.strip() for line in f]
        random.shuffle(self.file_paths)

    def __getitem__(self, index):
        img = io.read_image(self.file_paths[index])
        if img.shape[1] < self.patch_size or img.shape[2] < self.patch_size:
            print("Image too small:", self.file_paths[index], "skipping...")
            return self.__getitem__(random.randint(0, len(self.file_paths) - 1))
        img = self.crop(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        quality = random.randint(self.quality[0], self.quality[1])
        img = io.decode_jpeg(io.encode_jpeg(img, quality=quality))
        img = img.float() / 255
        return img, float(quality) / 100


class CsvImgDataset(ParentDataset):
    def __init__(
        self,
        quality,
        patch_size,
        label,
        num_classes,
        center_crop,
        csv_path,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            quality, patch_size, label, num_classes, center_crop, *args, **kwargs
        )
        size_cond = (df["height"] >= patch_size) & (df["width"] >= patch_size)
        df = df[size_cond].reset_index(drop=True)
        self.df = df
        self.file_paths = df["path"].tolist()
        random.shuffle(self.file_paths)
