import torch
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet50

from mislnet import MISLNet
from lib.CustomDataset import ExemplarDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This implementation deals with domain incremental learning in 2 classes setting
class iCaRLModel:
	def __init__(self, model_state_dict, neural_network='resnet50', total_memory=1000 , n_class=2, feature_size=200):
		'''
		memory_per_task:	  Defaults to 1000 per class
		model_state_dict:	 This is the state dict of pretrained model
		memory_size:		  Total memory buffer budget
		feature_size:		 Size of the feature extractor
		'''
		
		
		self.feature_size = feature_size
		self.exemplar_sets = {} # Dictionary to store exemplars for each class
		self.compute_exemplar_means = True
		self.n_class = 2
		self.total_memory = total_memory
		self.neural_network = neural_network

		# self.memory_per_task = total_memory/self.n_class


		if neural_network=='resnet50':
			self.classifier = resnet50(weights=None)
			
			self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 2)
			self.classifier.load_state_dict(model_state_dict)

			# Storing the last layer, will need this when updating representation 
			####################################################################
			self.last_layer = nn.Linear(self.classifier.fc.in_features, 2)
			self.last_layer.weight.data.copy_(self.classifier.fc.weight.data)
			self.last_layer.bias.data.copy_(self.classifier.fc.bias.data)
			####################################################################

			self.feature_extractor = self.classifier
			self.feature_extractor.fc = nn.Identity()
			self.feature_extractor.to(device)

			for i, (name, param) in enumerate(self.feature_extractor.named_parameters()):
				# Freezing almost 50% of the parameters. Upto 137th layer there are 12.4M parameters
				if i < 137:
					param.requires_grad = False

			self.feature_extractor.eval()

			
		elif neural_network=='mislnet':
			self.classifier = MISLNet(num_classes=2)
			self.classifier.load_state_dict(model_state_dict)

			# Storing the last layer, will need this when updating representation 
			self.last_layer = self.classifier.output
			
			self.feature_extractor = self.classifier
			self.feature_extractor.output = nn.Identity()

			# Transfer feature extractor to cuda
			self.feature_extractor.to(device)
			self.feature_extractor.eval()

		else:
			raise NotImplementedError("Currently only resnet50 and MISLNet is supported.")

		
	def reduce_exemplar_sets(self):
		# Reduce the number of exemplars per class to fit in memory
		# exemplar for real stays constant.
		print(f'Number of classes in icarl: {self.n_class}')
		for i in range(1, self.n_class):
			self.exemplar_sets[i] = self.exemplar_sets[i][:self.memory_per_task]
		


	def assign_exemplars(self, exemplars, selected_class, append=False):
		'''
		Assigns the calculated exemplars to the exemplar sets dictionary
		exemplars: N number of images that is closest to the mean of entire training set of that class (Nx256x256x3 tensor)
		selected_class: Class ID of the exemplars passed
		append: False if just need to assign. If True, need to append to the dictionary for that key.
		'''
		if not append:
			self.exemplar_sets[selected_class] = exemplars
		else:
			self.exemplar_sets[selected_class].extend(exemplars)

	
	def select_exemplars(self, class_dataset):
		'''
		Select exemplars for a class using herding
		Total number of exemplars per task: self.memory_per_task
		class_dataset: consists label and images for the given class: idx, img, label
					   should only have one class in the dataset   
		'''
		
		selected_class = class_dataset[0][2]
	
		with torch.no_grad():
			print("Calculating current class mean")
			features_list = []
			
			# Use DataLoader for batch processing to speed up feature extraction
			dataloader = DataLoader(class_dataset, batch_size=32, num_workers=2, shuffle=False, pin_memory=True if device == 'cuda' else False)
			
			for _, images, _ in tqdm(dataloader, desc="Processing"):
				images = images.to(device)
				features = self.feature_extractor(images)  # Extract features for the batch
				features = F.normalize(features, p=2, dim=1)  # L2 Normalize the features
				features_list.append(features.cpu())  # Move features to CPU to reduce GPU memory usage
	
			# Concatenate all features and compute the mean, and then normalize
			all_features = torch.cat(features_list, dim=0)
			class_mean_feature = all_features.mean(dim=0, keepdim=True)
			class_mean_feature = F.normalize(class_mean_feature, p=2, dim=1)  # Normalize the mean feature
			
			print(f"Class mean for {selected_class} -> Calculated")
	
			exemplar_set = []
			selected_image = []
	
			# Pre-compute and store features for all images to avoid recomputing
			precomputed_features = all_features.clone().to(device)  # Move precomputed features to device for comparison
			

			# Real image stays constant, all else is divided up 
			self.memory_per_task = int((self.total_memory/2)/(self.n_class - 1))

			print(f"New memory budget per task: {self.memory_per_task}")
			
			for k in tqdm(range(self.memory_per_task), desc="Calculating exemplars"):
				if k > 0:
					# Use precomputed features to avoid redundant computation
					exemplar_features = precomputed_features[selected_image]  # Use indexing to fetch selected exemplar features
					exemplar_feature_sum = exemplar_features.sum(dim=0, keepdim=True)
				else:
					exemplar_feature_sum = torch.zeros((1, self.feature_size)).to(device)
	
				# Calculate the intermediate mean, termed "mu_tmp" here
				mu_tmp = (exemplar_feature_sum + precomputed_features) / (k + 1)
				mu_tmp = F.normalize(mu_tmp, p=2, dim=1)
	
				# Find the image index with the minimum distance
				distances = torch.norm(class_mean_feature - mu_tmp.to('cpu'), dim=1)
				image_idx_min = torch.argmin(distances).item()
	
				# Track selected images
				selected_image.append(image_idx_min)
	
				# Append the corresponding image to the exemplar set
				exemplar_set.append(class_dataset[image_idx_min][1])
	
			return exemplar_set, selected_image
			

	def calculate_mean_exemplars(self):
		mean_exemplar_set = []
		for key, images in self.exemplar_sets.items():
			print(f"Calculating mean of exemplars for the class {key}")
			running_feature_sum = torch.zeros((1, self.feature_size), device=device)

			for image in images:
				image = image.unsqueeze(0).to(device)
				features = self.feature_extractor(image)
				features = F.normalize(features, p=2, dim=1)  # L2 Normalize the features
				running_feature_sum += features

			mean_feature = running_feature_sum / len(images)
			mean_feature = F.normalize(mean_feature, p=2, dim=1)
			mean_exemplar_set.append(mean_feature)

		print("Mean of exemplar sets calculated")
		self.mean_exemplar_set = torch.cat(mean_exemplar_set, dim=0)
		self.compute_exemplar_means = False

	
	def classify(self, image_batch):
		image_batch = image_batch.to(device)
		
		with torch.no_grad():
			if self.compute_exemplar_means:
				self.calculate_mean_exemplars()

			batch_feature = self.feature_extractor(image_batch)
			batch_feature = F.normalize(batch_feature, p=2, dim=1)

			batch_expanded = batch_feature.unsqueeze(1)  # Shape becomes [batch_size, 1, feature_size]
			mean_exemplar_set_expanded = self.mean_exemplar_set.unsqueeze(0) # Shape becomes [1, classes, feature_size]
			
			distances = torch.norm(batch_expanded - mean_exemplar_set_expanded, p=1, dim=2).detach().cpu()  # Shape becomes [32, num_classes]
			
			distances_std = (distances - distances.mean()) / distances.std()
			pseudo_probabilities = F.softmax(-distances_std, dim=1).detach().cpu().numpy()
			# Find the index of the closest reference  for each element in the batch
			closest_indices = distances.argmin(dim=1)

			return closest_indices, pseudo_probabilities


	def expanded_network(self):
		# This method increases the last layer of network by one every time it is called.
		# Keeps weights and biases constant for the previous known classes
		
		# Get the existing weights and bias from the last layer
		existing_weights = self.last_layer.weight.data
		existing_bias = self.last_layer.bias.data
		
		# Create a new layer with out_features increased by 1
		new_out_features = self.n_class  # Increase the number of output features by 1
		new_layer = nn.Linear(in_features=self.feature_size, out_features=new_out_features)
		
		# Initialize the new layer weights and bias with existing values and add zeros for the new node
		new_weights = torch.zeros((new_out_features, self.feature_size))
		new_bias = torch.zeros((new_out_features))
		
		# Copy the existing weights and bias to the new weights and bias
		new_weights[:existing_weights.size(0), :existing_weights.size(1)] = existing_weights
		new_bias[:existing_bias.size(0)] = existing_bias
		
		# Assign the new weights and bias to the new layer
		new_layer.weight.data = new_weights
		new_layer.bias.data = new_bias
		print(new_layer)
		return new_layer

	
	def update_representation(self, new_dataset, num_epochs = 50, new_class=1, learning_rate=1e-5, temperature=2, distil_gamma = 0.5):
		''' 
		Combine new data with exemplars from memory
		new_dataset:	   dataset with new data
		new_class_id:   class ID of the new dataset
		'''


		def kl_divergence_with_temperature(teacher_logits, student_logits, temperature):
			"""
			Calculate the KL divergence with temperature scaling between teacher and student logits.
		
			Parameters:
			- teacher_logits: Logits from the teacher model (torch.Tensor).
			- student_logits: Logits from the student model (torch.Tensor).
			- temperature: Temperature parameter to soften the probability distributions (float).
		
			Returns:
			- KL divergence loss (torch.Tensor).
			"""
			# Soften the probabilities of both teacher and student outputs
			teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
			student_probs = F.softmax(student_logits / temperature, dim=-1)
			
			# Calculate the log probabilities of student outputs
			student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
			
			# Calculate the KL divergence
			kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
			
			return kl_div


		
		exemplar_dataset = ExemplarDataset(self.exemplar_sets)  # This creates dataset from dictionary exemplar_sets
		combined_dataset = ConcatDataset([new_dataset, exemplar_dataset]) # Combines the new_dataset and exemplar dataset

		combined_dataloader = DataLoader(combined_dataset, batch_size=32, num_workers=2, shuffle=True, pin_memory=True if device == 'cuda' else False)
		
		########################################################
		# First, get ouput logits for all combined dataset. Store them for distillation loss.
		self.n_class+=new_class

		if self.neural_network=='mislnet':
			self.feature_extractor.output = self.expanded_network()
		else:
			self.feature_extractor.fc = self.expanded_network()
		

		# print(self.feature_extractor)
		
		self.feature_extractor.to(device)
		self.feature_extractor.eval()
		

		q = torch.zeros(len(combined_dataset), self.n_class).cuda()

		print("Getting logits for teacher model")
		with torch.no_grad():
			for indices, images, labels in tqdm(combined_dataloader):
				indices = indices.cuda()
				images = images.cuda()
				g = F.sigmoid(self.feature_extractor(images.to(device)))
				q[indices] = g.data
			
			q = q.cuda()

		
		########################################################
		# Now, train the model to update representation of classifier 

		# Define optimizers and loss functions here
		optimizer = optim.SGD(self.feature_extractor.parameters(), lr=learning_rate, weight_decay=0.00001)
		classification_loss = nn.CrossEntropyLoss()
		distillation_loss = nn.BCELoss()
		
		self.feature_extractor.train()

		best_train_loss = 1e5
		
		print("Training model with classification and distillation loss")
		for epoch in tqdm(range(num_epochs)):
			# with tqdm(combined_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
			for i, (indices, images, labels) in enumerate(combined_dataloader):
				images = images.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				g = self.feature_extractor(images)
				
				# Classification loss
				cl_loss = classification_loss(g, labels)

				# Distillation loss
				g = F.sigmoid(g)
				q_i = q[indices]
				
				# dist_loss = distillation_loss(g, q_i)
				dist_loss = kl_divergence_with_temperature(q_i, g, temperature=temperature)
				
				# loss = cl_loss + dist_loss
				loss = (1-distil_gamma) * cl_loss + distil_gamma * dist_loss

				loss.backward()
				optimizer.step()

				if loss.item()<best_train_loss:
					best_train_epoch = epoch
					best_train_loss = loss.item()
					best_loss_model_dump_name = f'best_feature_extractor_icarl_{learning_rate}.pt'
					torch.save(self.feature_extractor, best_loss_model_dump_name)
					# pbar.set_postfix(loss=loss.item(), cl_loss=cl_loss.item(), dist_loss=dist_loss.item())
				
				
		
		########################################################
		# First, load the saved performing model on training data
		# Now set the classifer back to feature extractor, last layer would be identity
		if self.neural_network=='mislnet':
			original_fc = self.feature_extractor.output
		else:
			original_fc = self.feature_extractor.fc
		
		# Storing this last layer for future use
		self.last_layer = nn.Linear(in_features=original_fc.in_features,
									out_features=original_fc.out_features,
									bias=(original_fc.bias is not None))
		
		# Copy the weights and biases from the original layer
		self.last_layer.weight.data = original_fc.weight.data.clone()
		if original_fc.bias is not None:
			self.last_layer.bias.data = original_fc.bias.data.clone()

		#########################################################

		# Convert the final layer to be feature extractor, instead of classifier
		if self.neural_network=='mislnet':
			self.feature_extractor.output = nn.Identity()
		else:
			self.feature_extractor.fc = nn.Identity()

		# # Transfer feature extractor to eval mode
		self.feature_extractor.eval()
		
		# Set this flag to True, since exemplar mean needs to be recalculated
		self.compute_exemplar_means = True
		
