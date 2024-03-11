import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import io

class OldCustomDataset(Dataset):
	def __init__(self, txt_file_paths=None, transform=None, images_np=None, labels_np=None):
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
			img_path = self.image_paths[idx]
			label = self.labels[idx]
			image = Image.open(img_path)  # Assuming these are paths to images

			if image.size[0]<256 or image.size[1]<256:
				resize_transform = transforms.Resize(256, antialias=True)
				image = resize_transform(image)
			
			if image.mode=='L':
				image = image.convert('RGB')


			# # Apply JPEG compression
			# buffer = io.BytesIO()
			# image.save(buffer, format='JPEG', quality=99)
			# buffer.seek(0)
			# image = Image.open(buffer)
				
			if self.transform:
				image = self.transform(image)
				
		return idx, image, label