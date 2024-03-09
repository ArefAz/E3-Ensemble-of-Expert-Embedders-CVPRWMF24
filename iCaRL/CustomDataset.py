import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
# import io
from torchvision import io
import random
from torchvision.transforms import RandomCrop

class CustomDataset(Dataset):
	def __init__(self, txt_file_paths=None, transform=None, images_np=None, labels_np=None, patch_size=256):
		"""
		Initialize the dataset with either a list of .txt files containing image paths or numpy arrays of images and labels.
		
		Args:
			txt_file_paths (list of str): List of paths to .txt files, each containing image paths. Each .txt file represents a class.
			transform (callable, optional): Optional transform to be applied on a sample.
			images_np (numpy.ndarray, optional): Numpy array of images (used when not using .txt files for image paths).
			labels_np (numpy.ndarray, optional): Numpy array of labels (used when not using .txt files for image paths).
		"""
		self.transform = transform
		self.images_np = images_np
		self.labels_np = labels_np
		self.image_paths = []
		self.labels = []
		self.patch_size = patch_size
		self.crop = RandomCrop(patch_size)
		
		if txt_file_paths is not None:
			# Load image paths and labels from .txt files
			for label, txt_path in enumerate(txt_file_paths):
				if txt_path=="":
					continue
				with open(txt_path, 'r') as f:
					for line in f:
						self.image_paths.append(line.strip())  # Remove newline characters
						self.labels.append(label)  # The index of the .txt file is the label

		self.from_numpy = images_np is not None and labels_np is not None
		
	def __len__(self):
		if self.from_numpy:
			return len(self.images_np)
		return len(self.image_paths)
	
	def __getitem__(self, idx):

		if self.from_numpy:
			# Handle numpy array data
			image = torch.from_numpy(self.images_np[idx])
			label = self.labels_np[idx]

		else:
			# Load image and label from the list populated from .txt files
			label = self.labels[idx]
			try:
				image = io.read_image(self.image_paths[idx])
			except:
				print(f"Error loading {self.image_paths[idx]}")
				return self.__getitem__(random.randint(0, len(self.image_paths) - 1))

			if image.shape[1]<self.patch_size or image.shape[2]<self.patch_size:
				resize_transform = transforms.Resize(self.patch_size, antialias=True)
				image = resize_transform(image)
			
			if image.shape[0] == 1:
				image = image.repeat(3, 1, 1)

			# Apply JPEG compression
			image = io.decode_jpeg(io.encode_jpeg(image, quality=99))
			
			# Convert image to 0-1 range
			image = image.float() / 255

			# Random crop if the size of image is bigger than patch_size
			image = self.crop(image)
				
		return idx, image, label
		

class ExemplarDataset(Dataset):
	def __init__(self, exemplar_dict, transform=None):
		"""
		Args:
			exemplar_dict (dict): A dictionary with class labels as keys and tensors of images as values.
			transform (callable, optional): Optional transform to be applied on an image.
		"""
		self.labels = []
		self.images = []
		for label, images in exemplar_dict.items():
			for img in images:
				self.images.append(img)
				self.labels.append(label)
		
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx]
		
		if self.transform:
			image = self.transform(image)
		
		return idx, image, label

